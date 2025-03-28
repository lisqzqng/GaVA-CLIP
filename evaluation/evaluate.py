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

import sys
sys.path.append('./')
sys.path.append('./training/')
import os
sys.path.insert(0, os.getcwd())
import os.path as osp
import glob
import time
import shutil
from collections import OrderedDict
import pandas as pd
import yaml

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

import video_dataset
import training.checkpoint as checkpoint
from training.VitaCLIP_model import VitaCLIP

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
    
def main():
    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    # evaluate settings
    parser.add_argument('--type', choices=['updrs', 'updrs_3cls', 'diag', 'diag_3cls'], default='diag',
                        help='type of the task',)

    # backbone and checkpoint paths
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights', default='')
    
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

    # zeroshot evaluation
    parser.add_argument('--zeroshot_evaluation', action='store_true', dest='zeroshot_evaluation',
                        help='set into zeroshot evaluation mode')
    parser.add_argument('--zeroshot_text_features_path', type=str, default='./ucf101_text_features_B16/class-only.pth',
                        help='path to saved clip text features to be used for zeroshot evaluation')
    
    #fp16
    parser.add_argument('--use_fp16', action='store_true', dest='fp16',
                        help='disable fp16 during training or inference')
    parser.set_defaults(fp16=False)

    # use summary token attn
    parser.add_argument('--use_summary_token', action='store_true', dest='use_summary_token',
                        help='use summary token')
    # use local prompts
    parser.add_argument('--use_local_prompts', action='store_true', dest='use_local_prompts',
                        help='use local (frame-level conditioned) prompts')
    # use global prompts
    parser.add_argument('--use_global_prompts', action='store_true', dest='use_global_prompts',
                        help='use global (video-level unconditioned) prompts')
    parser.add_argument('--num_global_prompts', type=int, default=8,
                        help='number of global prompts')
    # set defaults
    parser.set_defaults(use_summary_token=False, use_local_prompts=False, use_global_prompts=False)

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
    parser.add_argument('--text_num_prompts', type=int, default=16,
                        help='number of text prompts')
    parser.add_argument('--text_prompt_pos', type=str, default='end',
                        help='postion of text prompt')
    parser.add_argument('--text_prompt_init', type=str, default='',
                        help='initialization option for text prompt. Leave empty for random')
    parser.add_argument('--use_text_prompt_CSC', action='store_true', dest='text_prompt_CSC',
                        help='use Class Specific Context in text prompt')
    parser.add_argument('--text_prompt_classes_path', type=str, default='./classes/k400_classes.txt',
                        help='path of classnames txt file')
    # ------> advanced options for contextualPromptLeaner
    parser.add_argument('--knowledge_version', action='append', type=str, default=[],
                        help='knowledge version(s) used in text prompt learning')
    parser.add_argument('--use_descriptor', action='store_true', help='use class-wose descriptors \
        for text prompt learning')
    parser.add_argument("--token_wise_mlp", action='store_true', help='Operate token-wise projecction for all classes.')

    # loss params
    parser.add_argument("--use_focal_ordinal_loss", action='store_true', dest='focal_ordinal_loss',)
    
    parser.add_argument("--use_sigmoid_loss", action='store_true', dest='sigmoid_loss',)

    # support memory
    parser.add_argument("--clLoss_nte_video", dest='add_nte', action='store_true', help='Make contrastive learning between NTE and video',)
    parser.add_argument("--use_support_memory", action='store_true', 
                        help='use support memory for class-specific text prompts learning',)
    parser.add_argument("--memory_data_path", type=str, default='./data/gait/data_dict_part4.pkl',
                        help='file path to the precomputed support memory',)
    parser.add_argument("--mem_batch_size", type=int, default=32,
                        help='batch size for support memory during training',)
    parser.add_argument("--class_wise_mlp", action='store_true',
                        help='apply class-wise MLP projection for support memory during training',)
    
    parser.add_argument("--memory_loss_weight", type=float, default=0.1,)
    parser.add_argument("--vnte_loss_weight", type=float, default=0.05,)

    parser.add_argument("--detach", action='store_true', help='detach features (text/video) from graph in support memory training',)

    
    args = parser.parse_args()
    
    # get the number of classes
    cls_labels = []
    is_find = False
    with open(args.text_prompt_classes_path, 'r') as f:
        # count line number
        for ele in f:
            global CLS_NUM
            if ele.strip()[0] != '*':continue
            CLS_NUM += 1
            cls_labels.append(ele.strip()[1:])
    # initialize distributed training
    # dist.init_process_group('nccl', rank=0, world_size=1,)
    dist.init_process_group('nccl',
                            init_method='tcp://127.0.0.1:16007',
                            rank=0,
                            world_size=1,) 
    setup_print(dist.get_rank() == 0)
    cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)

    # get checkpoint paths
    assert osp.isdir(args.checkpoint_dir)
    nfold = len(glob.glob(osp.join(args.checkpoint_dir, 'fold*')))
    checkpoint_format = osp.join(args.checkpoint_dir, 'fold_{:d}/fold-{:d}-best.pth')
    # load config
    config_path = osp.join(args.checkpoint_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # # update the args if commun setting in config
    for key, value in config.items():
        if key in args:
            if 'data_root' in key:
                continue
            elif 'list_path' in key:
                continue
            elif 'checkpoint' in key:
                continue
            setattr(args, key, value)
    # load evaluation data
    eval_loader = video_dataset.create_val_loader(args)
    performance = []
    conf_mat = torch.zeros((CLS_NUM, CLS_NUM)).cuda() # | true, -> prediction
    for nf in range(nfold):
        checkpoint_path = checkpoint_format.format(nf, nf)
        if not osp.isfile(checkpoint_path):continue
        if dist.get_rank() == 0:
            print(f'Loading checkpoint from {checkpoint_path}')
        
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
            use_summary_token=args.use_summary_token,
            # use local prompts
            use_local_prompts=args.use_local_prompts,
            # use global prompts
            use_global_prompts=args.use_global_prompts,
            num_global_prompts=args.num_global_prompts,
            # use text prompt learning
            use_text_prompt_learning=args.use_text_prompt_learning,
            text_context_length=args.text_context_length,
            text_vocab_size=args.text_vocab_size,
            text_transformer_width=args.text_transformer_width,
            text_transformer_heads=args.text_transformer_heads,
            text_transformer_layers=args.text_transformer_layers,
            text_num_prompts=args.text_num_prompts,
            text_prompt_pos=args.text_prompt_pos,
            text_prompt_init=args.text_prompt_init,
            text_prompt_CSC=args.text_prompt_CSC,
            text_prompt_classes_path=args.text_prompt_classes_path,
            knowledge_version=args.knowledge_version,
            use_descriptor=args.use_descriptor,
            token_wise_mlp=args.token_wise_mlp,
            # zeroshot eval
            zeroshot_evaluation=True,
            zeroshot_text_features_path=checkpoint_path,
            # support memory
            use_support_memory=False,
            detach_features=args.detach,            
        )
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        renamed_ckpt = OrderedDict((k[len("module."):], v) for k, v in ckpt['model'].items() if k.startswith("module."))
        # remove class-knowledge related modules
        key_list = list(renamed_ckpt.keys())
        for key in key_list:
            if 'tf_project' in key or 'sum_proj' in key or 'memory_project' in key or 'logit_scale_mt' in key:
                del renamed_ckpt[key]    
        model.load_state_dict(renamed_ckpt, strict=True)

        print('----------------------------------------------------')
        model.cuda()
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[cuda_device_id], output_device=cuda_device_id,
        # )

        tot, hit1, = 0, 0,

        # neg_pairs = []
        # calculate sensitivity & specificity using confusion matrix
        for data, labels, _ in eval_loader:
            data, labels = data.cuda(), labels.cuda()
            # assert data.size(0) == 1
            # if data.ndim == 6:
            #     data = data[0] # now the first dimension is number of views

            with torch.no_grad():
                logits, _ , _ = model(data)
                scores = logits.softmax(dim=-1)

            tot += data.size(0)
            for ind in range(data.size(0)):
                conf_mat[labels[ind], scores[ind].topk(1)[1]] += 1
                hit1 += (scores[ind].topk(1)[1] == labels[ind]).sum().item()

            if tot % 20 == 0:
                print(f'[Evaluation] num_samples: {tot}  '
                    f'cumulative_acc1: {hit1 / tot * 100.:.2f}%')


        sync_tensor = torch.LongTensor([tot, hit1]).cuda()
        dist.all_reduce(sync_tensor)
        tot, hit1 = sync_tensor.cpu().tolist()
        if dist.get_rank() == 0:
            print(f'Accuracy on evaluation set fold-{nf}: top1={hit1 / tot * 100:.2f}%')
            performance.append(hit1 / tot)
    
    # write the overall evaluation accuracy
    if dist.get_rank() == 0:
        conf_mat = conf_mat.detach().cpu().numpy()
        print(f'Overall accuracy: {np.mean(np.array(performance)) * 100:.2f}%')
        
        print('----------------------------------------------------')
        print('Overall confusion matrix:')
        print(conf_mat)
        output_file = osp.join(args.checkpoint_dir,  f"eval_{args.data_root.split('datasets/')[-1].replace('/','_')}.txt")
        # calculate the F1-score
        f1_score = np.zeros(CLS_NUM)
        for ci in range(CLS_NUM):
            tp = conf_mat[ci, ci]
            f1_score[ci] = 2 * tp / (conf_mat[ci].sum() + conf_mat[:, ci].sum()+1e-8)
        print("Per-class F1-score:\n")
        f1_str = " ".join([f'{x:.4f}' for x in f1_score.tolist()])
        print(f1_str)
        average_f1 = np.mean(f1_score)
        print(f'Average F1-score: {average_f1:.4f}')
        # count the number of sequences for each class
        seq_num = conf_mat.sum(1)
        with open(output_file, 'w') as f:
            f.write(f'Overall accuracy: {np.mean(np.array(performance)) * 100:.2f}%\n')
            f.write(f'Overall F1-score: {f1_str}\n')
            f.write(f'Average F1-score: {average_f1:.4f}\n')
            f.write('Per-class sequence number:\n')
            f.write(" ".join([str(int(x)) for x in seq_num.tolist()]) + '\n')
            f.write('Overall confusion matrix:\n')
            # write matrix line by line
            for i in range(CLS_NUM):
                f.write(' '.join([str(int(conf_mat[i, j])) for j in range(CLS_NUM)]) + '\n')
        # save confusion matrix figure
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_mat.astype(int), annot=False, fmt='d', cmap='Blues', cbar=True)
        # increase the font size in heatmap
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)  
        # increase cbar font size
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)      
        plt.savefig(output_file.replace('.txt', '.png'))
    
    return

if __name__ == '__main__':
    global CLS_NUM
    CLS_NUM = 0 # the number of classes
    main()