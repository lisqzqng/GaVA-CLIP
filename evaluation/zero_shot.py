"""
Evaluate pre-trained VitaCLIP model on the target task.
Without prompt tuning.
"""
import argparse
from datetime import datetime
import builtins

import torch
torch.manual_seed(0) 
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import os.path as osp
import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, osp.join(os.getcwd(), './training/'))
import time
from collections import OrderedDict
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

import video_dataset
from VitaCLIP_model import VitaCLIP
from VitaCLIP_text_encoder import CLIPTextEncoder, tokenize

from train import setup_print

from typing import List
from collections import OrderedDict

def load_text_fetures_from_checkpoint(checkpoint_path):
    return

def knowledge_to_text_features(args:argparse.ArgumentParser, cls_names:List[str]):
    "Get the per-calss text features from NL descriptions"
    text_model = CLIPTextEncoder(
        embed_dim=args.embed_dim,
        context_length=args.text_context_length,
        vocab_size=args.text_vocab_size,
        transformer_width=args.text_transformer_width,
        transformer_heads=args.text_transformer_heads,
        transformer_layers=args.text_transformer_layers,
    )
    ckpt = torch.load(args.backbone_path, map_location='cpu')
    # change the name of the keys
    ckpt = {k.replace('textual.', ''): v for k, v in ckpt.items() if k.startswith('textual')}
    text_model.load_state_dict(ckpt, strict=True)

    text_model.cuda()
    text_model.eval()
    #--> preprocess the class names
    cls_names = [name.replace("_", " ") for name in cls_names]
    if args.use_discrete_prompt:
        # load discrete prompts from file
        disc_file = osp.join(args.info_dir, f'ke_{args.type}', f"simQdesc_{args.knowledge_version}.txt")
        assert osp.isfile(disc_file)
        cls_disc = []
        with open(disc_file, 'r') as f:
            for _, line in enumerate(f):
                cls_disc.append(line.strip())
        assert len(cls_disc) == len(cls_names)
        cls_names = [' '.join([cls_disc[i], cls_names[i]]) for i in range(len(cls_names))]
    
    # tokenize the class names
    tokenized_prompts = torch.cat([tokenize(name) for name in cls_names])
    with torch.no_grad():
        token_features = text_model.token_embedding(tokenized_prompts.cuda())
        text_features = text_model(token_features, tokenized_prompts.cuda())
    
    # save text features to file
    filename = osp.join(args.info_dir, f'ke_{args.type}', f"text_features_{args.knowledge_version}.npy")
    torch.save(text_features.cpu(), filename)

    del text_model

    return filename
    
def main():
    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser) # data_root, val_list_path, batch_size
    
    # vlm checkpoint path
    parser.add_argument('--backbone_path', type=str, default='./pretrained/clip_pretrained.pth')
    parser.add_argument('--pretrained_vlm', type=str, default='./pretrained/ckpt_k400.pth',)

    # model params
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch size of patch embedding')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='number of transformer heads')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='number of transformer layers')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='transformer feature dimension')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='clip projection embedding size')
    parser.add_argument('--mlp_factor', type=float, default=4.0,
                        help='transformer mlp factor')
    parser.add_argument('--cls_dropout', type=float, default=0.5,
                        help='dropout rate applied before the final classification linear projection')
    # text prompt learning
    parser.add_argument('--use_text_prompt_learning', action='store_true', dest='use_text_prompt_learning',
                        help='use coop text prompt learning')
    parser.add_argument('--text_context_length', type=int, default=77,
                        help='text model context length')
    parser.add_argument('--text_vocab_size', type=int, default=49408,
                        help='text model vocab size')
    parser.add_argument('--text_transformer_width', type=int, default=512,
                        help='text transformer width')
    parser.add_argument('--text_transformer_heads', type=int, default=8,
                        help='text transformer heads')
    parser.add_argument('--text_transformer_layers', type=int, default=12,
                        help='text transformer layers')
    parser.add_argument('--text_num_prompts', type=int, default=8,
                        help='number of text prompts')
    parser.add_argument('--text_prompt_pos', type=str, default='end',
                        help='postion of text prompt')
    parser.add_argument('--text_prompt_init', type=str, default='',
                        help='initialization option for text prompt. Leave empty for random')
    parser.add_argument('--use_text_prompt_CSC', action='store_true', dest='text_prompt_CSC',
                        help='use Class Specific Context in text prompt')
   
    # classification params
    parser.add_argument('--type', type=str, default='updrs', help='classification type')
    parser.add_argument('--text_prompt_classes_path', type=str, default='./classes/k400_classes.txt',
                        help='path of classnames txt file')
    parser.add_argument('--use_discrete_prompt', action='store_true', help='use discrete prompt')
    parser.add_argument('--info_dir', type=str, default='./data/',)
    parser.add_argument('--knowledge_version', type=str, default='v0',)
    parser.add_argument('--nfold', type=int, default=10, help='number of folds of cross-validation')

    args = parser.parse_args()
    
    # get the number of classes
    cls_names = []
    with open(args.text_prompt_classes_path, 'r') as f:
        # count line number
        for ele in f:
            global CLS_NUM
            if ele.strip()[0] != '*':continue
            CLS_NUM += 1
            cls_names.append(ele.strip()[1:])
    # generate text features from knowledge
    zeroshot_text_features_path = knowledge_to_text_features(args, cls_names)
    # initialize distributed training
    # dist.init_process_group('nccl',)
    dist.init_process_group('nccl',
                            init_method='tcp://127.0.0.1:16007',
                            rank=0,
                            world_size=1,) 
    setup_print(dist.get_rank() == 0)
    cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)

    # get checkpoint paths
    assert osp.isfile(args.pretrained_vlm)
    # load evaluation data
    eval_loader = video_dataset.create_eval_loader(args)
    conf_mat = torch.zeros((CLS_NUM, CLS_NUM)).cuda() # | true, -> prediction
    
    model = VitaCLIP(# load weights
        backbone_path=args.backbone_path,
        # data shape
        input_size=(args.spatial_size, args.spatial_size),
        num_frames=args.num_frames,
        # exper setting
        cls_type=args.type,
        num_classes=CLS_NUM,
        # model def
        feature_dim=args.feature_dim,
        patch_size=(args.patch_size, args.patch_size),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_factor=args.mlp_factor,
        embed_dim=args.embed_dim,
        # use summary token
        use_summary_token=True,
        # use local prompts
        use_local_prompts=True,
        # use global prompts
        use_global_prompts=True,
        num_global_prompts=8,
        # use text prompt learning
        use_text_prompt_learning=False,
        # zeroshot eval
        zeroshot_evaluation=True,
        zeroshot_text_features_path=zeroshot_text_features_path, 
    )
    if dist.get_rank() == 0:
        print(f'Loading checkpoint from {args.pretrained_vlm}')
    ckpt = torch.load(args.pretrained_vlm, map_location='cpu')['model']
    # keep only the visual part
    visual_ckpt = OrderedDict({k.replace('module.', ''): v for k, v in ckpt.items() \
                        if 'textual' not in k and 'prompt_learner' not in k})
    model.load_state_dict(visual_ckpt, strict=True)

    print('----------------------------------------------------')
    model.cuda()
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[cuda_device_id], output_device=cuda_device_id,
    # )

    tot, hit1, = 0, 0,
    eval_st = datetime.now()
    # neg_pairs = []
    # calculate sensitivity & specificity using confusion matrix
    for idx, (data, labels, _) in enumerate(eval_loader):
        data, labels = data.cuda(), labels.cuda()
        # assert data.size(0) == 1

        with torch.no_grad():
            logits, _ , _ = model(data)
            scores = logits.softmax(dim=-1).argmax(-1)

        tot += data.size(0)
        for ind in range(data.size(0)):
            conf_mat[labels[ind], scores[ind].topk(1)[1]] += 1
            hit1 += (scores[ind].topk(1)[1] == labels[ind]).sum().item()

        if tot % 20 == 0:
            print(f'[Evaluation] num_samples: {tot}  '
                f'ETA: {(datetime.now() - eval_st) / idx * (len(eval_loader) - idx)}  '
                f'cumulative_acc1: {hit1 / tot * 100.:.2f}%')
            
    sync_tensor = torch.LongTensor([tot, hit1]).cuda()
    dist.all_reduce(sync_tensor)
    tot, hit1 = sync_tensor.cpu().tolist()
    print(f'Evaluation accuracy: top1={hit1 / tot * 100:.2f}%')
    performance = hit1 / tot

    # calculate F1-score per class from confusion matrix
    # use weighted average F1-score, the weight of which are determined by the number of samples available in that class
    f1_score = np.zeros(CLS_NUM)
    wf1_score = np.zeros(CLS_NUM)
    weights = conf_mat.sum(axis=1) / conf_mat.sum()
    for ci in range(CLS_NUM):
        f1_score[ci] = 2 * conf_mat[ci, ci] / (conf_mat[ci, :].sum() + conf_mat[:, ci].sum())
        wf1_score[ci] = f1_score[ci] * weights[ci]
        
    f1_score = np.nan_to_num(f1_score, nan=0.)
    wf1_score = np.nan_to_num(wf1_score, nan=0.)
    # write the overall evaluation accuracy
    if dist.get_rank() == 0:
        print(f'Overall accuracy: {np.mean(np.array(performance)) * 100:.2f}%')
        
        print('----------------------------------------------------')
        print('Overall confusion matrix:')
        print(conf_mat)
        os.makedirs('./eval_output', exist_ok=True)
        output_file = osp.join('./eval_output',  f'disc_{args.knowledge_version}.txt' if \
                               args.use_discrete_prompt else 'class_name.txt')
        with open(output_file, 'w') as f:
            f.write(f'Overall accuracy: {np.mean(np.array(performance)) * 100:.2f}%\n')
            f.write('Overall confusion matrix:\n')
            # write matrix line by line
            for i in range(CLS_NUM):
                f.write(' '.join([str(int(conf_mat[i, j])) for j in range(CLS_NUM)]) + '\n')
            f.write('----------------------------------------------------\n')
            # write the f1-score
            f.write('\nF1-score per class: ' + ' '.join([f'{x:.4f}' for x in f1_score]))
            f.write(f'\nAverage F1-score: {f1_score.mean():.4f}')
            # write the weighted f1-score
            f.write('\nWeighted F1-score per class: ' + ' '.join([f'{x:.4f}' for x in wf1_score]))
            f.write(f'\nAverage weighted F1-score: {wf1_score.sum():.4f}')
        
    
    return

if __name__ == '__main__':
    global CLS_NUM
    CLS_NUM = 0 # the number of classes
    main()