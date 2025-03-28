import torch
torch.manual_seed(0) 
import torch.nn as nn

import copy
from collections import OrderedDict
from typing import Union, List
from pkg_resources import packaging

from VitaCLIP_text_encoder_utils import SimpleTokenizer as _Tokenizer
from kapt_head import ContextualPromptLearner

from typing import List

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    _tokenizer = _Tokenizer()

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, maple_prompts=None):
        if maple_prompts:
            num_prompts = maple_prompts[0].shape[0]
            for i, blk in enumerate(self.resblocks):
                if i == 0:
                    x = blk(x)
                else:
                    prefix = x[:1, :, :]
                    suffix = x[1 + num_prompts:, :, :]
                    # Create/configure learnable tokens of this layer
                    textual_context = maple_prompts[i-1]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    # Add the learnable tokens of this layer with the input, replaced by previous
                    # layer learnable tokens
                    x = torch.cat([prefix, textual_context, suffix], dim=0)

                    # then do forward pass from transformer
                    x = blk(x)                    
        else:
            for blk in self.resblocks:
                x = blk(x)
        return x

class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
    ):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, prompts, tokenized_prompts, maple_prompts=None,):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if maple_prompts:
            x = self.transformer(x, maple_prompts)
        else:
            x = self.transformer(x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), torch.where(tokenized_prompts==self.vocab_size-1,)[-1]] @ self.text_projection

        return x


class TextPromptLearner(nn.Module):
    def __init__(self, classnames, 
                 text_model, 
                 num_prompts, 
                 prompts_init='', 
                 CSC=False, 
                 ctx_pos='end', 
                 cls_type='updrs', 
                 knowledge_version:List[str]=['v0'],
                 use_descriptor:bool=False,
                 token_wise_mlp:bool=False,
                 ):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        n_ctx = num_prompts
        ctx_init = prompts_init.lower()
        assert ctx_init == '' or set(ctx_init.split('_')).issubset({'split', 'uni', 'cntn', 'disc'}), "Invalid prompt initialization"
        ctx_dim = text_model.ln_final.weight.shape[0]
        self.knowledge_aware_prompt = False
        self.use_descriptor = use_descriptor
        
        if ctx_init!='':
            self.knowledge_aware_prompt  = True
            self.context_prompt_learner = ContextualPromptLearner(
                use_cntn=True if 'cntn' in ctx_init else False,
                cntn_split=True if 'split' in ctx_init else False,
                uni_mlp=True if 'uni' in ctx_init else False,
                use_disc=True if 'disc' in ctx_init else False,
                emb_dim=ctx_dim//4,
                out_dim=ctx_dim,
                n_cls=n_cls,
                n_tokens=n_ctx,
                cls_type=cls_type,
                knowledge_version=knowledge_version,
                use_descriptor=self.use_descriptor,
                token_wise_mlp=token_wise_mlp,
                )

            print("Initializing class-specific contexts with kapt")

            # initialize class-specific learnable parameters
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            nn.init.zeros_(ctx_vectors)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        else:
            if ctx_init=='fixed':
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = tokenize(ctx_init)
                with torch.no_grad():
                    embedding = text_model.token_embedding(prompt)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization with normal distribution
                if CSC:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
                else:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # get the integral text prompts (to be tokenized)
        if self.knowledge_aware_prompt:
            # concatenate with discrete knowledge
            prompts = [[] for _ in range(n_cls)]
            for idc in range(n_cls):
                if self.use_descriptor:
                    cls_descriptor = self.context_prompt_learner.cls_disc[idc]
                    prompts[idc].extend([descriptor + " " + classnames[idc] for descriptor in cls_descriptor]) # extend a list
                else:
                    for ik in range(len(knowledge_version)):
                        prompts[idc].append(self.context_prompt_learner.cls_disc[idc][ik] + " " + classnames[idc])
        # elif len(prompt_prefix) > 0:
        #    prompts = []
        #    for idx, name in enumerate(classnames):
        #        prompts.append(prompt_prefix[idx] + " " + name)
        else:
            prompts = [[prompt_prefix + " " + name + "."] for name in classnames]
        
        assert isinstance(prompts[0], list)
        tokenized_prompts = []
        for prompt in prompts:
            # process prompt for each class
            tokenized_prompts.append(torch.cat([tokenize(p) for p in prompt]))
        # tokenized_prompts = torch.stack(tokenized_prompts)
        # else:
        #    tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        #    tokenized_prompts = [tokenized_prompts]

        # check the length of the raw text
        assert max([torch.where(tokenized_prompts[idc]==49407)[-1].max().item() \
            for idc in range(n_cls)]) <= 77, "The tokenized prompt is too long"
        # print(tokenized_prompts.shape)
        embedding = []
        token_prefix, token_suffix = [], []
        for idc in range(n_cls):
            with torch.no_grad():
                embedding.append(text_model.token_embedding(tokenized_prompts[idc]))

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
        
            # self.register_buffer("token_prefix", embedding[0][:, :1, :].unsqueeze(1))  # SOS
            # if self.knowledge_aware_prompt:
            #     self.register_buffer("token_suffix", embedding[0][:, 1: -n_ctx, :].unsqueeze(1))  # CLS, EOS
            # else:
            #     self.register_buffer("token_suffix", embedding[0][:, 1 + n_ctx :, :].unsqueeze(1))  # CLS, EOS

            token_prefix.append(embedding[idc][:, :1, :])  # SOS
            if self.knowledge_aware_prompt:
                token_suffix.append(embedding[idc][:, 1: -n_ctx, :])  # CLS, EOS
            else:
                token_suffix.append(embedding[idc][:, 1 + n_ctx :, :])
                
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self):
        if self.knowledge_aware_prompt:
            ctx = self.context_prompt_learner(self.ctx)
        else:
            ctx = self.ctx
            # if ctx.dim() == 2:
            #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx = ctx.unsqueeze(1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = []
        for idc in range(self.n_cls):
            if self.class_token_position == "end":
                prompts.append(torch.cat(
                    [
                        prefix[idc].to(ctx[idc].device),  # (n_kv, 1, dim)
                        ctx[idc],     # (n_kv, n_ctx, dim)
                        suffix[idc].to(ctx[idc].device),  # (n_kv, *, dim)
                    ],
                    dim=-2,
                ))
            # elif self.class_token_position == "middle":
            #     half_n_ctx = self.n_ctx // 2
            #     prompts = []
            #     for i in range(self.n_cls):
            #         name_len = self.name_lens[i]
            #         prefix_i = prefix[i : i + 1, :, :]
            #         class_i = suffix[i : i + 1, :name_len, :]
            #         suffix_i = suffix[i : i + 1, name_len:, :]
            #         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
            #         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
            #         prompt = torch.cat(
            #             [
            #                 prefix_i,     # (1, 1, dim)
            #                 ctx_i_half1,  # (1, n_ctx//2, dim)
            #                 class_i,      # (1, name_len, dim)
            #                 ctx_i_half2,  # (1, n_ctx//2, dim)
            #                 suffix_i,     # (1, *, dim)
            #             ],
            #             dim=-2,
            #         )
            #         prompts.append(prompt)
            #     prompts = torch.cat(prompts, dim=0)

            # elif self.class_token_position == "front":
            #     prompts = []
            #     for i in range(self.n_cls):
            #         name_len = self.name_lens[i]
            #         prefix_i = prefix[i : i + 1, :, :]
            #         class_i = suffix[i : i + 1, :name_len, :]
            #         suffix_i = suffix[i : i + 1, name_len:, :]
            #         ctx_i = ctx[i : i + 1, :, :]
            #         prompt = torch.cat(
            #             [
            #                 prefix_i,  # (1, 1, dim)
            #                 class_i,   # (1, name_len, dim)
            #                 ctx_i,     # (1, n_ctx, dim)
            #                 suffix_i,  # (1, *, dim)
            #             ],
            #             dim=-2,
            #         )
            #         prompts.append(prompt)
            #     prompts = torch.cat(prompts, dim=0)

            else:
                raise NotImplementedError(f"Unsupported class token position: {self.class_token_position}")

        return prompts        



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])