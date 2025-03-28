import sys
sys.path.append('./')
import os
sys.path.insert(0, os.getcwd())
import os.path as osp
import json
import pickle
from typing import Tuple
import shutil
from tqdm import tqdm

import argparse
import time
from datetime import datetime
import builtins
from collections import OrderedDict

import numpy as np
import torch
torch.manual_seed(0) 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from VitaCLIP_text_encoder import tokenize, CLIPTextEncoder, _Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2LMHeadModel
from loss_utils import categorical_ordinal_focal_weight

class ClipGaitDataset(Dataset):
    """
    Variant of `DeCap.ClipCocoDataset` for gait analysis. \n
    data_path: list of paths to the pkl or json files. 
    If pkl, then the tokens are loaded directly. If json, then the tokens are computed on the fly. \n
    """
    def __init__(self, data_path:list, device:str='cuda:0'):
        super().__init__()
        self.clip_tokenizer = CLIPTextEncoder(
                embed_dim=512,
                context_length=77,
                vocab_size=49408,
                transformer_width=512,
                transformer_heads=8,
                transformer_layers=12,
            )
        # load pretrained model
        ckpt = torch.load('pretrained/clip_pretrained.pth')
        new_ckpt = OrderedDict()
        for n, param in ckpt.items():
            if 'textual' in n:
                new_ckpt[n.replace('textual.','')] = param
        self.clip_tokenizer.load_state_dict(new_ckpt, strict=True)
        self.clip_tokenizer.to(device)
        self.clip_tokenizer.eval()

        assert isinstance(data_path, list), 'data_path must be a list of paths'
        self.tokens = []
        self.embeds = []
        # load natural language descriptions
        for dpath in data_path:
            if dpath.endswith('.json'):
                with open(dpath, 'r') as f:
                    data = json.load(f)
                assert isinstance(data, list)
                # tokenize all the descriptions
                ## load pre-computed tokens
                if isinstance(data[0], str) or isinstance(data[0], np.ndarray):
                    has_tokenized = isinstance(data[0], np.ndarray)
                    # construct tokenized texts
                    tokens = []
                    embeds = []
                    for text in tqdm(data):
                        try:
                            tokenized_text = torch.from_numpy(text).long() if has_tokenized else tokenize(text)
                        except:
                            continue
                        else:
                            with torch.no_grad():
                                prefix_embedding = self.clip_tokenizer.token_embedding(tokenized_text.to(device).reshape(1,-1)) # extend dimension to 1x77x512
                                # encode the prefix embedding to 512*1 text embedding
                                text_embedding = self.clip_tokenizer(prefix_embedding, tokenized_text) # 1x512
                            tokens.append(tokenized_text.long().cpu().reshape(1,-1)) # tensor
                            embeds.append(text_embedding.float().cpu().reshape(1,-1)) # tensor
                    self.tokens.extend(tokens)
                    self.embeds.extend(embeds)
                    tokens = torch.cat(tokens, dim=0)
                    embeds = torch.cat(embeds, dim=0)
                    dict_data = {'tokens': tokens.numpy(), 'embeds': embeds.numpy()}
                    with open(dpath.replace('.json', '.pkl'), 'wb') as f:
                        pickle.dump(dict_data, f)
                else:
                    raise NotImplementedError
            elif dpath.endswith('.pkl'):
                with open(dpath, 'rb') as f:
                    data = pickle.load(f)
                assert 'tokens' in data.keys() and 'embeds' in data.keys()
                assert isinstance(data['tokens'][0], np.ndarray) and isinstance(data['embeds'][0], np.ndarray)

                self.tokens.extend([torch.from_numpy(x).long().reshape(1,-1) for x in data['tokens']])
                self.embeds.extend([torch.from_numpy(x).float().reshape(1,-1) for x in data['embeds'].mean(-2)])
            else:
                raise NotImplementedError    
        
        assert len(self.tokens) == len(self.embeds)
                
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index: int,):
        # compute on 'cpu'
        # load token
        tokenized_text = self.tokens[index] # 1x77 int64 tensor
        # load embedding
        text_embedding = self.embeds[index] # 1x512 float32 tensor
        
        return tokenized_text.squeeze(0), text_embedding.squeeze(0) # 77, 512

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
class DeCap(nn.Module):

    def __init__(self,
                 prefix_size: int = 512,
                 cfg_path=None,
                #  pretrained='decap/pretrained/epoch-049.pt',
                #  pretrained='decap/pretrained/coco_prefix-reduced.pt',
                 pretrained='',
                ):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        if cfg_path is None or not osp.isfile(cfg_path):
            cfg_path = './decap/config.pkl'
        with open(cfg_path,'rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config) # add 500 numbers into token vocab. (only 200 effective data points)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1] # 768
        # self.clip_project = MLP((prefix_size, self.embedding_size)) # 512 --> 768
        # use a bottle-neck MLP to project the prefix embedding to text embedding
        self.clip_project = MLP((prefix_size, self.embedding_size//3, self.embedding_size)) # 512-->256-->768
        
        # load pretrained model if exists
        if osp.isfile(pretrained):
            ckpt = torch.load(pretrained, map_location= torch.device('cpu'))
            self.load_state_dict(ckpt, strict=False)
        
    def forward(self, clip_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out
    
def setup_print(is_master: bool):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    
def train_decoder(args, lr: float = 1e-5, warmup_steps: int = 1000,):

    # dist.init_process_group(backend='nccl', init_method='env://')
    logdir = f"./decap/logs/{time.strftime('%Y%m%d-%H%M')}"
    os.makedirs(logdir, exist_ok=True)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:16006',
                            rank=0,
                            world_size=1,)
    setup_print(dist.get_rank() == 0)

    cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    device = torch.device('cuda:'+str(cuda_device_id))
    torch.cuda.set_device(cuda_device_id)
    SEED=42
    torch.cuda.manual_seed_all(SEED)
    # training hyperparameters
    batch_size = args.bs
    epochs = args.epochs
    vocab_size = args.vocab_size
    writer = None
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=logdir)
        args.output_dir = osp.join(args.output_dir, f"{time.strftime('%Y%m%d-%H%M')}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
    model = DeCap()
    
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction='none')
    loss_ordinal = categorical_ordinal_focal_weight(alpha=0.0, gamma=2.0, beta=0.2) # only ordinal loss
    model.to(device)

    model = DDP(
        model,
        device_ids=[cuda_device_id],
        output_device=cuda_device_id,
        find_unused_parameters=False, #True,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # construct the datasets
    train_dataset = ClipGaitDataset(data_path=args.train_data,)
    train_sampler = DistributedSampler(train_dataset)    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=batch_size,drop_last=True)

    if len(args.valid_data)>0:
        val_dataset = ClipGaitDataset(data_path=args.valid_data,)
        val_sampler = DistributedSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler,batch_size=batch_size,drop_last=True)
    else:
        val_dataloader = None
    
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    # set variable for best accuracy
    best_val_acc = 0.
    best_train_acc = 0.
    for epoch in range(epochs):
        loss_token_save, loss_num_save, ac_save = 0, 0, 0
        sys.stdout.flush()
        if dist.get_rank() == 0:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader)/10), desc="Training",)

        dist.barrier()
        for idx,(clip_tokens, text_embedding) in enumerate(train_dataloader):
            clip_tokens, text_embedding = clip_tokens.to(device), text_embedding.to(device)

            # get the 512*1 text embedding
            with torch.no_grad():
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True) 
            
            outputs = model(text_embedding.float(), clip_tokens)
            
            logits = outputs.logits

            logits = logits[:,: -1]
            clip_tokens = clip_tokens.flatten() # 4928 = 64*77
            logits = logits.reshape(-1, logits.shape[-1])
            num_ids = torch.where(clip_tokens>=vocab_size)[0].detach().cpu().numpy()
            # pred_num_ids = torch.where(logits.argmax(1)>=vocab_size)[0].detach().cpu().numpy()
            # num_ids = np.intersect1d(real_num_ids, pred_num_ids)
            
            loss_token = loss_ce(logits, clip_tokens)
            # add ordinal loss for number-word tokens
            loss_number_weight = loss_ordinal(logits[num_ids, vocab_size:], clip_tokens[num_ids]-vocab_size)
            loss_number = loss_number_weight * loss_token[num_ids]
            loss_all = loss_token.mean() + loss_number.mean()
            
            ac=((logits.argmax(1)==clip_tokens)*(clip_tokens>0)).sum()/(clip_tokens>0).sum()
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            
            display = False
            if display:
                _tokenizer = _Tokenizer()
                # convert to vocab 49408
                cliptt = clip_tokens[:77].detach().cpu().numpy().copy()
                number_id = np.where(cliptt>=vocab_size)[0]
                number = cliptt[number_id].copy() - vocab_size
                cliptt[number_id] = 286
                original_text = _tokenizer.decode(cliptt) # `clip_tokens` already flattened
                original_text = original_text.split()
                for i, n in zip(number_id, number):
                    original_text[i-1] = str(n)
                original_text = ' '.join(original_text)
                llogits = logits[:77].detach()
                llogits = llogits.argmax(dim=-1).cpu().numpy().copy()
                number_id = np.where(llogits>=vocab_size)[0]
                number = llogits[number_id].copy() - vocab_size
                llogits[number_id] = 286
                decoded_text = _tokenizer.decode(llogits)
                decoded_text = decoded_text.split()
                for i, n in zip(number_id, number):
                    try:
                        decoded_text[i-1] = str(n)
                    except IndexError:
                        print(f"IndexError: {i}, {n}")
                        break
                decoded_text = ' '.join(decoded_text)
                print(f"Original text: {original_text}\nDecoded text: {decoded_text}")
                del _tokenizer
                
            if dist.get_rank() == 0:
                
                if(idx+1) %10 ==0:
                    progress.set_postfix({"loss_token": loss_token_save/10.0, \
                                          "loss_number": loss_num_save/10.0,"acc_token":ac_save/10.0})
                    progress.update()
                    loss_token_save, loss_num_save, ac_save= 0, 0, 0
                else:
                    loss_token_save += loss_token.mean().item()
                    loss_num_save += loss_number.mean().item()
                    ac_save += ac.item()
                    
                if writer is not None and dist.get_rank() == 0:
                    global_epoch = epoch*len(train_dataloader) + idx + 1
                    writer.add_scalar('train/loss token', loss_token.mean().item(), global_epoch)
                    writer.add_scalar('train/loss number', loss_number.mean().item(), global_epoch)
                    writer.add_scalar('train/accuracy', ac.item(), global_epoch)
        
        model.eval()
        eval_st = datetime.now()           
        dist.barrier() # wait all the process to finish the epoch
        
        # evaluate at the end of epoch
        if val_dataloader is not None:
            tot, hit1 = 0, 0
            for val_tokens, val_embedding in val_dataloader:
                val_tokens, val_embedding = val_tokens.to(device), val_embedding.to(device)
                
                with torch.no_grad():
                    val_embedding /= val_embedding.norm(dim=-1, keepdim=True)
                    outputs = model(val_embedding.float(), val_tokens)
                    
                logits = outputs.logits
                logits = logits[:,: -1]
                logits = logits.reshape(-1, logits.shape[-1])
                tot += (val_tokens>0).sum().item()
                hit1 += ((logits.argmax(1)==val_tokens.flatten())*(val_tokens.flatten()>0)).sum().item()
                
                if tot % 20 == 0:
                    print(f'[Evaluation] num_samples: {tot}  '
                        f'ETA: {(datetime.now() - eval_st) / tot * (len(val_dataloader) - tot)}  '
                        f'cumulative_acc1: {hit1 / tot * 100.:.6f}%')
                    
            print(f'[Evaluation] overall samples: {tot}  '
                f'ETA: {(datetime.now() - eval_st) / tot * (len(val_dataloader) - tot)}  '
                f'cumulative_acc1: {hit1 / tot * 100.:.6f}%')
            
            sync_tensor = torch.LongTensor([tot, hit1]).cuda()
            dist.all_reduce(sync_tensor)
            tot, hit1 = sync_tensor.cpu().tolist()
            if writer is not None and dist.get_rank() == 0:
                val_acc = hit1 / tot * 100.
                writer.add_scalar('valid/accuracy', val_acc, global_epoch)
            print(f'Accuracy on validation set: top1={val_acc:.6f}%')
        else:
            val_acc = -1.
            
        if dist.get_rank() == 0:
            # with open(osp.join(log_dir,'output.txt'),'a+') as f:
            #     f.writelines('epoch ' +str(epoch) +': '+ progress.postfix+'\r\n')
            progress.close()
            if val_acc > best_val_acc:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.output_dir, f"val_best.pt"),
                )
                best_val_acc = val_acc
            if ac.item() > best_train_acc:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.output_dir, f"train_best.pt"),
                )
                best_train_acc = ac.item()
            if epoch%args.save_freq==0 or epoch == epochs - 1:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.output_dir, f"{epoch:03d}.pt"),
                )
                
        model.train()
        dist.barrier()
        
    if dist.get_rank() == 0:
        writer.close()
        
    dist.destroy_process_group()
                
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', nargs='+', default='')
    parser.add_argument('--valid_data', nargs='+', default='') #./decap/valid_memory.json')
    parser.add_argument('--cfg_file', default='./decap/config.pkl')
    parser.add_argument('--output_dir', default='./train_output/decap/')
    parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--ord_start_epochs', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=49408)
    
    args = parser.parse_args()

    train_decoder(args)


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    main()
