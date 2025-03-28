#!/usr/bin/env python
import os.path as osp
import pickle

from typing import Tuple
import numpy as np

import torch
torch.manual_seed(0) 
import torch.nn as nn
import torch.nn.functional as F

from VitaCLIP_vision_encoder import CLIPVisionEncoder
from VitaCLIP_text_encoder import CLIPTextEncoder, TextPromptLearner, tokenize
# from memory_head import MemoryPromptLearner
from video_dataset import NUM_COMB

from typing import List

MAX_EMBED_NUM = 10000 # maximum number of embeddings for each class

class VitaCLIP(nn.Module):

    def __init__(
        self,
        # load weights
        backbone_path: str = '',
        # data shape
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        use_fp16: bool = False,
        # exper setting
        cls_type: str = 'updrs',
        num_classes: int = 4,   
        # model def
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        embed_dim: int = 512,
        # use summary token
        use_summary_token: bool = False,
        # use local prompts
        use_local_prompts: bool = False,
        # use global prompts
        use_global_prompts: bool = False,
        num_global_prompts: int = 8,
        # use text prompt learning
        use_text_prompt_learning: bool = False,
        text_context_length: int = 77,
        text_vocab_size: int = 49408,
        text_transformer_width: int = 512,
        text_transformer_heads: int = 8,
        text_transformer_layers: int = 12,
        text_num_prompts: int = 8,
        text_prompt_pos: str = 'end',
        text_prompt_init: str = '',
        text_prompt_CSC: bool = False,
        text_prompt_classes_path: str = '',
        knowledge_version: List[str] = ['v0'],
        use_descriptor: bool = False,
        token_wise_mlp:bool = False,
        # zeroshot eval
        zeroshot_evaluation: bool = False,
        zeroshot_text_features_path: str = '',
        # support memory
        use_support_memory: bool = False,
        detach_features: bool=False,
        memory_batch_size: int=64,
        add_nte: bool=False,
        # loss opt
        use_sigmoid_loss: bool = False,
        ):
        super().__init__()


        # general paremeters
        self.fp16 = use_fp16

        # frames and tubelet
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.text_context_length = text_context_length
        self.text_transformer_width = text_transformer_width

        # use summary token
        self.use_summary_token = use_summary_token

        # loss parameters
        self.use_sigmoid_loss = use_sigmoid_loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = None

        # zeroshot text_features
        self.zeroshot_evaluation = zeroshot_evaluation
        if self.zeroshot_evaluation:
            self.text_features = torch.load(zeroshot_text_features_path, map_location='cpu')['text_features']
            
        
        # visual model
        self.visual = CLIPVisionEncoder(
            # data shape
            input_size=input_size,
            num_frames=num_frames,
            # model def
            feature_dim=feature_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_factor=mlp_factor,
            embed_dim=embed_dim,
            # use summary token
            use_summary_token=use_summary_token,
            # use local prompts
            use_local_prompts=use_local_prompts,
            # use global prompts
            use_global_prompts=use_global_prompts,
            num_global_prompts=num_global_prompts,
        )
        
        self.use_text_prompt_learning = use_text_prompt_learning

        # text prompt learning
        if self.use_text_prompt_learning:
            self.textual = CLIPTextEncoder(
                embed_dim=embed_dim,
                context_length=text_context_length,
                vocab_size=text_vocab_size,
                transformer_width=text_transformer_width,
                transformer_heads=text_transformer_heads,
                transformer_layers=text_transformer_layers,
            )
        
        if backbone_path:
            ckpt = torch.load(backbone_path)
            self.load_state_dict(ckpt, strict=False)

        # support memory
        self.use_support_memory = use_support_memory
        self.memoty_batch_size = memory_batch_size
        self.detach_features = detach_features
        self.add_nte = add_nte # add contrastive learning btw video efatures and NTE

        # if self.add_nte or self.use_support_memory:
        #     self.proj_layer = nn.Linear(embed_dim, embed_dim)

        if self.add_nte:
            # self.proj_layer = nn.Linear(embed_dim, embed_dim)
            # with torch.no_grad():
            #     self.nte_prompts = tokenize(" ".join(["X"]*NUM_COMB))
            #     nte_tokens = self.textual.token_embedding(self.nte_prompts)
            # self.nte_prefix = nte_tokens[:,:1,:]
            # self.nte_postfix = nte_tokens[:,(1+NUM_COMB):,:]
            self.sum_proj = nn.Linear(feature_dim, embed_dim)
            if self.use_sigmoid_loss:
                self.logit_scale_vm = nn.Parameter(torch.ones([]) * np.log(10.))
            else:
                self.logit_scale_vm = nn.Parameter(torch.ones([]) * 100.)
            

        if self.use_support_memory: 
            # project KEPLER gait param embeddings to the same space as the text features
            # self.memory_head = MemoryPromptLearner(
            #     self.textual, num_classes, batch_size=memory_batch_size, splitMLP=class_wise_mlp,
            # )
            self.tf_project  = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//4),
                nn.Tanh(),
                nn.Linear(embed_dim//4, embed_dim//8),
            )
            # project text features
            # self.tf_project = nn.Linear(embed_dim, embed_dim)
            ## =====> Obsolete <===== ##
            # =====> Only one memory projection layer (MLP)
            # self.memory_project = nn.Sequential(
            #         nn.Linear(embed_dim, embed_dim//4),
            #         nn.Tanh(),
            #         nn.Linear(embed_dim//4, embed_dim),
            #     )
            # self.memory_project = nn.Linear(embed_dim, embed_dim//4)
            self.memory_project = nn.ModuleList([])
            for _ in range(num_classes):
                self.memory_project.append(
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim//4),
                        nn.Tanh(),
                        nn.Linear(embed_dim//4, embed_dim//8),
                    )
                )
            if self.use_sigmoid_loss:
                self.logit_scale_mt = nn.Parameter(torch.ones([]) * np.log(10.))
                self.logit_bias_mt = nn.Parameter(torch.ones([])* -10.)
            else:
                self.logit_scale_mt = nn.Parameter(torch.ones([]) * 100.)
                self.logit_bias_mt = None

        if self.use_sigmoid_loss: # modify loss params initialization
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(np.log(10.)))
            self.logit_bias = nn.Parameter(torch.ones([])* -10.)

        if self.use_text_prompt_learning:
            with open(text_prompt_classes_path, 'r') as f:
                classes = f.read().strip().split('\n')
            classes = [x for x in classes if x[0] != '*']
            
            self.prompt_learner = TextPromptLearner(
                            classnames=classes,
                            text_model=self.textual,
                            num_prompts=text_num_prompts,
                            prompts_init=text_prompt_init,
                            CSC=text_prompt_CSC,
                            ctx_pos=text_prompt_pos,
                            cls_type=cls_type,
                            knowledge_version=knowledge_version,
                            use_descriptor=use_descriptor,
                            token_wise_mlp=token_wise_mlp,
                            )
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # freeze encoders
        self._freeze_visual_except_prompts_time_embed()
        try:
            self._freeze_textual()
        except:
            pass


    
    def _freeze_visual_except_prompts_time_embed(self):
        for name, param in self.visual.named_parameters():
                if 'summary' in name or 'local' in name or 'global' in name or 'time_embed' in name:
                    pass
                else:
                    param.requires_grad = False
    
    def _freeze_textual(self):
        for name, param in self.textual.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, 
                memory=None, 
                video_nte=None, 
                desc_wise=False):
        B, C, T, H, W = x.size()

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        # used in training
        if self.use_text_prompt_learning:
            # vision side
            video_features, summary = self.visual(x)
            # normalize video features
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            # text side
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            assert isinstance(prompts, list) and isinstance(tokenized_prompts, list)
            # if desc_wise, assert that model is in eval mode
            #    output the per-description logits for each class
            # convert to half16 float
            with torch.cuda.amp.autocast(self.fp16):
                # calculate the similarity per-class
                if desc_wise:
                    assert self.training == False
                    logits = []
                    # get all text features
                    for i in range(len(prompts)):
                        prompts[i] = prompts[i].to(x.device)
                        tokenized_prompts[i] = tokenized_prompts[i].to(x.device)
                        text_features = self.textual(prompts[i], tokenized_prompts[i])
                        # normalize text features
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        similarity = logit_scale * video_features @ text_features.t()
                        logits.append(similarity)
                else:
                    logits = torch.empty(B, 0).to(x.device)
                    # define the eventual per-class text features as mean of {tf_i}
                    text_features = torch.empty(0, self.text_transformer_width).to(x.device)
                    # process the knowledge / description by class
                    for i in range(len(prompts)):
                        prompts[i] = prompts[i].to(x.device)
                        tokenized_prompts[i] = tokenized_prompts[i].to(x.device)
                        _text_features = self.textual(prompts[i], tokenized_prompts[i])
                        # normalize text features
                        _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
                        similarity = logit_scale * video_features @ _text_features.t()
                        logits = torch.cat((logits, similarity.mean(-1, keepdim=True)), dim=-1)
                        text_features = torch.cat((text_features, _text_features.mean(0, keepdim=True)), dim=0)
                    text_features = text_features/text_features.norm(dim=-1, keepdim=True)
                    
            self.text_features = text_features

        # used in zeroshot evaluation
        else:
            # vision side
            video_features, summary = self.visual(x)
            # text side
            text_features = self.text_features.to(video_features.device)

            # normalized features
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * video_features @ text_features.t()

        if self.logit_bias is not None:
            logits += self.logit_bias

        if self.add_nte and video_nte is not None:
            # get valid nte
            sum_proj = self.sum_proj(summary)
            sum_proj = sum_proj / sum_proj.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                valid_idx = ((video_nte.sum(dim=-1).sum(dim=-1))!=0).float()
                # create the valid matrix from the valid index
                valid_mat = valid_idx.unsqueeze(1) * valid_idx.unsqueeze(0)
                # choose_id = torch.zeros((B, NUM_COMB)).to(x.device)
                # choose_id[:,torch.randint(0, NUM_COMB, (1,)).item()] = 1               
            #     # permute the order of combinaison in video_nte
            #     order = torch.randperm(NUM_COMB)
            #     video_nte = video_nte[:,order]
            #     nte_prompts = self.nte_prompts.to(x.device).expand(B, -1)
            #     nte_prefix =  self.nte_prefix.expand(B, -1, -1).to(x.device)
            #     nte_postfix = self.nte_postfix.expand(B, -1, -1).to(x.device)
            #     # nte_proj = self.proj_layer(video_nte)

            #     nte_tokens = torch.cat([
            #         nte_prefix,
            #         video_nte,
            #         nte_postfix,
            #     ], dim=1)
            #     nte_feature = self.textual(nte_tokens, nte_prompts).to(sum_proj.dtype)

            # nte_proj = self.proj_layer(nte_feature).to(sum_proj.dtype)
            # normalize the features
            # nte_feature = nte_feature / nte_feature.norm(dim=-1, keepdim=True)
            # calculate the similarity matrix betweem video features and NTE
            video_nte = video_nte / video_nte.norm(dim=-1, keepdim=True)
            similarity = torch.bmm(sum_proj.unsqueeze(0).expand(NUM_COMB,-1,-1), video_nte.permute(1, 2, 0)).mean(0)
            logits_mat = self.logit_scale_vm * (similarity * valid_mat)
            logits_vm = F.log_softmax(logits_mat, dim=-1) + F.log_softmax(logits_mat, dim=-2)
        else:
            logits_vm = None

        if self.use_support_memory and memory is not None:
            # assert self.add_nte, "Integrating support memory for text encoder requires video-paired NTE !"
            # project the embeddings to the same space 768 -> 512 
            # feed the concatenated embeddings to the text encoder (frozen)
            # memory_features = self.memory_head(memory, values)
            # memory_features = self.mem_encoder(memory.mean(dim=1))
            # memory_features /= memory_features.norm(dim=-1, keepdim=True)
            # text_features = self.tf_project(self.text_features)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # if memory_features.ndim==3: # splitMLP
            #     logits_mt = []
            #     for idm, mem_feat in enumerate(memory_features):
            #         logits_mt.append((self.logit_scale_mt*self.text_features[idm]@mem_feat.t()).unsqueeze(-1))
            #     logits_mt = torch.concat(logits_mt, dim=-1)
            # elif memory_features.ndim==2:
            #     logits_mt = (self.logit_scale_mt * text_features @ memory_features.t()).t()
            # else:
            #     raise NotImplementedError('Invalid dimension size for `memory_features`!')

            ## =====> Obsolete <===== ##
            # compute cosine similarity between text features and memory
            if self.detach_features:
                text_features = self.text_features.detach()
            else:
                text_features = self.text_features
            memory = memory.mean(dim=1)
            logits_mt = torch.empty(memory.size(0),0).to(memory.device)
            for cid, mproj in enumerate(self.memory_project):
                tf = self.tf_project(text_features[cid])
                tf = tf / tf.norm(dim=-1, keepdim=True)
                memo = mproj(memory)
                memo = memo / memo.norm(dim=-1, keepdim=True)
                logits_mt = torch.concat([logits_mt, (self.logit_scale_mt*memo@tf.t()).unsqueeze(-1)], dim=1)
            logits_mt = F.log_softmax(logits_mt, dim=-1)
            ## =====> Only one memory projection layer (MLP) 
            # tf = self.tf_project(text_features)
            # tf = tf / tf.norm(dim=-1, keepdim=True)
            # memo = self.memory_project(memory)
            # memo = memo / memo.norm(dim=-1, keepdim=True)
            # nte_proj = nte_proj / nte_proj.norm(dim=-1, keepdim=True)
            # mt_sim = torch.bmm(nte_proj.permute(1,0,2), text_features.t().unsqueeze(0).expand(NUM_COMB,-1,-1)).mean(0)
            # mt_sim = mt_sim * valid_idx
            # logits_mt = F.log_softmax(self.logit_scale_mt * memo@tf.t(), dim=-1)
            if self.logit_bias_mt is not None:
                logits_mt += self.logit_bias_mt
            # if not self.use_sigmoid_loss:
            #     logits_mt = torch.softmax(logits_mt, dim=-1)
            # attn = self.attn_regressor(memo)
            # attn_memory = (attn * memo).sum(dim=-2)
            # attn_memory = attn_memory / attn_memory.norm(dim=-1, keepdim=True)
        else:
            logits_mt = None
            
        
        return logits, logits_mt, logits_vm