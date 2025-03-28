#!/usr/bin/env python

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
import os
sys.path.insert(0, os.getcwd())
import os.path as osp
import time
from collections import OrderedDict
import yaml

from sklearn import metrics
import matplotlib.pyplot as plt

import video_dataset
import checkpoint
from VitaCLIP_model import VitaCLIP


from loss_utils import categorical_ordinal_focal_weight, sigmoid_focal_loss

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
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    # train settings
    parser.add_argument('--nfold', type=int, default=1,)
    parser.add_argument('--type', choices=['updrs', 'updrs_3cls', 'diag', 'diag_3cls'], default='diag',
                        help='type of the task',)
    parser.add_argument('--num_steps', type=int,
                        help='number of training steps')
    parser.add_argument('--eval_only', action='store_true',
                        help='run evaluation only')
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='save a checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='evaluate every N steps')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print log message every N steps')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.2,
                        help='optimizer weight decay')
    parser.add_argument('--batch_split', type=int, default=1,
                        help='optionally split the batch into smaller shards and forward/backward one shard '
                             'at a time to avoid out-of-memory error.')
    parser.add_argument('--for_zero_shot', action='store_true',
                        help='train a model for zero-shot evaluation')
    parser.add_argument('--early_stop_steps', type=int, default=10000,
                        help='will stop if accumulated steps is bigger',)
    
    # backbone and checkpoint paths
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights', default='')
    parser.add_argument('--checkpoint_path', type=str,
                        help='path to pretrained checkpoint weights', default='')
    
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
    parser.add_argument("--mem_batch_size", type=int, default=64,
                        help='batch size for support memory during training',)
    parser.add_argument("--class_wise_mlp", action='store_true',
                        help='apply class-wise MLP projection for support memory during training',)
    
    parser.add_argument("--memory_loss_weight", type=float, default=0.1,)
    parser.add_argument("--vnte_loss_weight", type=float, default=0.05,)

    parser.add_argument("--detach", action='store_true', help='detach features (text/video) from graph in support memory training',)

    
    args = parser.parse_args()
    
    # get the number of classes
    cls_labels = []

    with open(args.text_prompt_classes_path, 'r') as f:
        # count line number
        for ele in f:
            global CLS_NUM
            if ele.strip()[0] != '*':continue
            CLS_NUM += 1
            cls_labels.append(ele.strip()[1:])
    # initialize distributed training
    # dist.init_process_group('nccl',)
    dist.init_process_group('nccl',
                            init_method='tcp://127.0.0.1:16006',
                            rank=0,
                            world_size=1,) 
    setup_print(dist.get_rank() == 0)
    cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)
    # create folder for logging
    if dist.get_rank() == 0:
        if args.eval_only:
            all_eval_conf_mat = np.zeros((CLS_NUM, CLS_NUM))
            perf_list = []
        else:
            postfix = '_'+args.data_root+'_' if args.data_root else ''
            if len(args.text_prompt_init)>0:
                postfix += args.text_prompt_init.replace('_', '-') + '_' + '-'.join(args.knowledge_version)
            postfix += '_NTE' if args.use_support_memory else ''
            postfix += '_clL' if args.add_nte else ''
            if len(postfix) > 0 and postfix[0] != '_':
                postfix = '_' + postfix
            logdir = f"./logs/{args.type.lower()}{'-zs' if args.for_zero_shot else ''}_{time.strftime('%m%d-%H%M')}{postfix}/"
            os.makedirs(logdir, exist_ok=True)
            result_file = osp.join(logdir, 'results.txt')
            all_conf_mat = np.zeros((CLS_NUM, CLS_NUM))
            # save argparse to config file
            with open(osp.join(logdir, 'config.yaml'), 'w') as f:
                yaml.dump(vars(args), f)
            
        
    performances = []
    for n in range(args.nfold):
        bad_steps = 0
        pre_eval_perf = 0.
        # use different train/val csv files for each fold
        if not args.eval_only:
            if args.for_zero_shot:
                args.data_root = f'datasets/hospital/chunks_{int(n)}/'
            elif 'park' in args.data_root: # pure synthetic data
                args.data_root = f'datasets/parkinson_cv/'
            elif 'mix' in args.data_root:
                args.data_root = f'datasets/mix/'
            elif 'real' in args.data_root:
                args.data_root = f'datasets/real_3cls/train/'
            elif 'miccai' in args.data_root:
                args.data_root = f'datasets/miccai_10_fold/chunks_{int(n)}'
            elif 'tulip' in args.data_root:
                args.data_root = f'datasets/tulip/chunks_{int(n)}'
            args.train_list_path = osp.join(args.data_root, f'train_{args.type}.csv')
            args.val_list_path = osp.join(args.data_root, f'val_{args.type}.csv')
            if 'sep' in args.data_root:
                args.data_root = ''
                args.train_data_root = f'datasets/mix/'
                args.val_data_root = f'datasets/real_3cls/train/'
                args.train_list_path = osp.join(args.train_data_root, f'train_{args.type}_sep.csv')
                args.val_list_path = osp.join(args.val_data_root, f'val_{args.type}_sep.csv')


        # initialize best performance
        best_perf = torch.tensor(0.0).cpu()
        best_acc= torch.tensor(0.0).cpu()
        # initialize summary writer
        writer = None
        save_conf_mat = None
        try:
            args.checkpoint_path = osp.join(args.checkpoint_dir, f"fold-{n}-best.pth")
            if not osp.isfile(args.checkpoint_path):
                args.checkpoint_path = osp.join(args.checkpoint_dir, f"fold_{n}", f"fold-{n}-best.pth")
        except: pass
        if args.eval_only:
            assert osp.isfile(args.checkpoint_path), 'Checkpoint file not found.'
        else:
            if dist.get_rank() == 0:
                sub_logdir = osp.join(logdir, f'fold_{n}')
                writer = SummaryWriter(log_dir=sub_logdir)
                args.checkpoint_dir = sub_logdir
        # initialize model
        model = VitaCLIP(
            # load weights
            backbone_path=args.backbone_path,
            # data shape
            input_size=(args.spatial_size, args.spatial_size),
            num_frames=args.num_frames,
            use_fp16=args.fp16,
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
            zeroshot_evaluation=args.zeroshot_evaluation,
            zeroshot_text_features_path=args.zeroshot_text_features_path,
            # support memory
            use_support_memory=args.use_support_memory,
            detach_features=args.detach,
            memory_batch_size=args.mem_batch_size,
            add_nte=args.add_nte,
            # loss params
            use_sigmoid_loss=args.sigmoid_loss,
        )

        if osp.isfile(args.checkpoint_path):
            print('loading checkpoint')
            ckpt = torch.load(args.checkpoint_path, map_location='cpu')
            renamed_ckpt = OrderedDict((k[len("module."):], v) for k, v in ckpt['model'].items() if k.startswith("module."))
            model.load_state_dict(renamed_ckpt, strict=True)
        
        
        print(model)
        print('----------------------------------------------------')
        print('Trainable Parameters')
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('----------------------------------------------------')
        model.cuda()
        
        calculate_params = False
        if calculate_params:
            # calculate total trainable paramaters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if dist.get_rank() == 0:
                print(f'Total trainable parameters: {total_params}')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cuda_device_id], output_device=cuda_device_id,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # if args.sigmoid_loss:
        #     # reduce the learning rate when the loss reaches a plateau
        #     lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,\
        #                                                         min_lr=1e-10, verbose=True)
        # else:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
        loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        fo_criterion = categorical_ordinal_focal_weight(gamma=2., alpha=0.25, \
                                                        beta=0.2 if 'updrs' in args.type else 0., scale=1.0)
        if args.sigmoid_loss:
            # criterion = sigmoid_focal_loss(use_focal=args.focal_ordinal_loss, scale=1.0)
            mt_criterion = sigmoid_focal_loss(use_focal=False, scale=args.memory_loss_weight)
        else:
            mt_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        resume_step = checkpoint.resume_from_checkpoint(model, optimizer, lr_sched, loss_scaler, args)

        val_loader = video_dataset.create_val_loader(args)
        if args.eval_only:
            print('Running in eval_only mode.')
            model.eval()
            eval_perf, eval_conf_mat = evaluate(args, model, val_loader)
            eval_perf = torch.Tensor([eval_perf]).cuda()
            handle_mat = dist.all_reduce(eval_conf_mat, op=dist.ReduceOp.SUM, async_op=True)
            handle_mat.wait()
            handle = dist.all_reduce(eval_perf, op=dist.ReduceOp.MAX, async_op=True)
            handle.wait()
            eval_conf_mat = eval_conf_mat.cpu().numpy()
            eval_perf = eval_perf.cpu()
            # save current performance
            perf_list.append(eval_perf.item())
            all_eval_conf_mat += eval_conf_mat
            del model
            continue
        else:
            assert args.train_list_path is not None, 'Train list path must be specified if not in eval_only mode.'
            train_loader = video_dataset.create_train_loader(args, resume_step=resume_step)
            memory_loader = video_dataset.create_memory_loader(args, resume_step=resume_step,)

        assert len(train_loader) == args.num_steps - resume_step
        batch_st, train_st = datetime.now(), datetime.now()

        for i, ((data, labels, vid_nte), (memo, memolabel),) in enumerate(zip(train_loader, memory_loader), start=resume_step):
        # for i, (data, labels, vid_nte) in enumerate(train_loader, start=resume_step):
            # if bad_steps >= args.early_stop_steps:
            #     if dist.get_rank() == 0:
            #         print('Early stopping.')
            #     break
            if args.use_support_memory:
                # data, labels, memo, value, memolabel = data.cuda(), labels.cuda(), memo.cuda(), value.cuda(), memolabel.cuda()
                memo, memolabel = memo.cuda(), memolabel.cuda()
            else:
                memo, memolabel = None, None
            if args.add_nte:
                assert vid_nte is not None, 'NTE associated with video must be provided for contrastive learning.'
                vid_nte = vid_nte.cuda()
            else:
                vid_nte = None

            data, labels = data.cuda(), labels.cuda()
                
                
            data_ed = datetime.now()

            optimizer.zero_grad()

            assert data.size(0) % args.batch_split == 0
            split_size = data.size(0) // args.batch_split
            if args.use_support_memory:
                memo_split_size = memo.size(0) // args.batch_split
                # value_split_size = value.size(0) // args.batch_split
            hit1, loss_value, loss_mt_value, loss_vm_value = 0, 0, 0, 0
            for j in range(args.batch_split):
                data_slice = data[split_size * j: split_size * (j + 1)]
                labels_slice = labels[split_size * j: split_size * (j + 1)]
                if args.add_nte:
                    vid_nte_slice = vid_nte[split_size * j: split_size * (j + 1)]
                else:
                    vid_nte_slice = None
                if args.use_support_memory:
                    memo_slice = memo[memo_split_size * j: memo_split_size * (j + 1)]
                    # value_slice = value[value_split_size * j: value_split_size * (j + 1)]
                    mt_label_slice = memolabel[memo_split_size * j: memo_split_size * (j + 1)]
                else:
                    # memo_slice, value_slice, mt_label_slice = None, None, None
                    memo_slice, mt_label_slice = None, None

                with torch.cuda.amp.autocast(args.fp16):
                    
                    logits, logits_mt, logits_vm = model(data_slice, memory=memo_slice, video_nte=vid_nte_slice) #values=value_slice,

                    
                    loss = criterion(logits, labels_slice) # no reduction, loss per-sample
                    # calculate weights if necessary
                    if args.focal_ordinal_loss: # and not args.sigmoid_loss:
                        weights = fo_criterion(logits, labels_slice)
                        loss *= weights
                    
                    loss = loss.mean()

                    if args.use_support_memory:
                        # loss_memory = args.memory_loss_weight * criterion(logits_memory, mt_label_slice).mean()
                        # InfoNCE loss
                        #loss_mt = args.memory_loss_weight*tm_critertion(logits_mt, mt_label_slice)
                        if args.sigmoid_loss:
                            loss_mt = args.memory_loss_weight*mt_criterion(logits_mt, mt_label_slice)
                        else:
                            loss_mt = args.memory_loss_weight*(criterion(logits_mt, mt_label_slice))
                                #+ fo_criterion(logits_mt, mt_label_slice))
                        loss_mt = loss_mt.mean()
                        # loss_vm = args.memory_loss_weight*vm_critertion(logits_vm, mt_label_slice, labels_slice)
                        loss_tot = loss + loss_mt  #+ loss_vm # + loss_cls
                    else:
                        loss_mt = None
                        loss_tot = loss
                    
                    if args.add_nte:
                        # contrastive loss that encourage the positive pairs to have higher similarity and minimize the negative pairs
                        loss_vm = - args.vnte_loss_weight * torch.diag(logits_vm).mean()
                        loss_tot += loss_vm
                    else:
                        loss_vm = None
                    
                if labels.dtype == torch.long: # no mixup, can calculate accuracy
                    hit1 += (logits.topk(1, dim=1)[1] == labels_slice.view(-1, 1)).sum().item()
                loss_value += loss.item() / args.batch_split

                # cooperative contrastive learning loss 
                try: loss_mt_value += loss_mt.item() / args.batch_split
                except: pass
                
                try: loss_vm_value += loss_vm.item() / args.batch_split
                except: pass
                loss_scaler.scale(loss_tot / args.batch_split).backward()
            
            loss_scaler.step(optimizer)
            loss_scaler.update()
            if args.sigmoid_loss:
                lr_sched.step(loss_value)
            else:
                lr_sched.step()

            batch_ed = datetime.now()

            if i % args.print_freq == 0:
                sync_tensor = torch.Tensor([loss_value, hit1 / data.size(0)]).cuda()
                dist.all_reduce(sync_tensor)
                sync_tensor = sync_tensor.cpu() / dist.get_world_size()
                loss_value, acc1 = sync_tensor.tolist()

                print_text = f'batch_time: {(batch_ed - batch_st).total_seconds():.3f}  '+\
                    f'data_time: {(data_ed - batch_st).total_seconds():.3f}  '+\
                        f'ETA: {(batch_ed - train_st) / (i - resume_step + 1) * (args.num_steps - i - 1)}  |  '+\
                            f'lr: {optimizer.param_groups[0]["lr"]:.6f}  '+\
                                f'loss: {loss_value:.6f}' +\
                                    (f'  acc1: {acc1 * 100:.2f}%' if labels.dtype == torch.long else '')
                
                if loss_mt_value != 0:
                    print_text += f'  loss_mt: {loss_mt_value:.6f}' #  loss_vm: {loss_vm_value:.6f}'
                if loss_vm_value != 0:
                    print_text += f'  loss_vm: {loss_vm_value:.6f}'
                
                print(print_text)

                if writer is not None and dist.get_rank() == 0:
                    writer.add_scalar('train/accuracy', acc1, i + 1)
                    writer.add_scalar('train/loss', loss_value, i + 1)
                    if loss_mt_value != 0:
                        writer.add_scalar('train/loss_mt', loss_mt_value, i + 1)
                    if loss_vm_value != 0:
                        writer.add_scalar('train/loss_vm', loss_vm_value, i + 1)
            
            if (i + 1) % args.eval_freq == 0:
                print('Start model evaluation at step', i + 1)
                model.eval()
                eval_acc, conf_mat = evaluate(args, model, val_loader, writer=writer, global_step=i + 1)
                eval_acc = torch.Tensor([eval_acc]).cuda()
                handle_mat = dist.all_reduce(conf_mat, op=dist.ReduceOp.SUM, async_op=True)
                handle_mat.wait()
                handle = dist.all_reduce(eval_acc, op=dist.ReduceOp.MAX, async_op=True)
                handle.wait()
                conf_mat = conf_mat.cpu()
                eval_acc = eval_acc.cpu()
                # calculate the per-class F1 score
                f1_score = torch.zeros(CLS_NUM)
                for ci in range(CLS_NUM):
                    f1_score[ci] = 2 * conf_mat[ci, ci] / (conf_mat[ci, :].sum() + conf_mat[:, ci].sum()+1e-8)
                eval_perf = f1_score.mean()
                # update the global best performance
                # if dist.get_rank() == 0 and ((i > args.num_steps//10-2) or 'miccai' in args.data_root):
                if dist.get_rank() == 0:
                    if eval_perf >= best_perf:
                        best_perf = eval_perf
                        best_acc = eval_acc
                        # register the negative videos to the global list
                        # negpair_global = neg_pairs
                        # negpair_global.insert(0, ['vidname', 'ground-truth', 'prediction'])
                        save_conf_mat = conf_mat.numpy()
                        checkpoint.save_checkpoint(model, optimizer, lr_sched, loss_scaler, i + 1, args, \
                            is_best=True, name='fold-{}'.format(n))
                        
                if eval_perf <= pre_eval_perf:
                    bad_steps += args.eval_freq
                else:
                    bad_steps = 0
                # synchronize the `bad_steps` values

                pre_eval_perf = eval_perf

                model.train()
                

            if (i + 1) % args.save_freq == 0 and dist.get_rank() == 0:
                checkpoint.save_checkpoint(model, optimizer, lr_sched, loss_scaler, i + 1, args)
            
            batch_st = datetime.now()
        
        if dist.get_rank()==0:
            if writer is not None:
                writer.close()
            # add performance to the list
            # calculate the accuracy 
            performances.append(best_acc.numpy())
            perf_string = ' '.join([f'fold-{ind} ' + str(x) for ind, x in enumerate(performances)])
            with open(result_file, 'w') as f:
                f.write(perf_string)
            # save confusion matrix
            all_conf_mat += save_conf_mat
            # save the confusion matrix as text file
            np.savetxt(osp.join(osp.dirname(result_file), f'confusion_matrix_fold-{n}.txt'), save_conf_mat, fmt='%d')
            # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=save_conf_mat, display_labels=cls_labels)
            # cm_display.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal', values_format=None)
            # plt.savefig(os.path.join(osp.dirname(result_file), f'confusion_matrix_fold-{n}.png'))
            # plt.close()
        
        del model

    
    if dist.get_rank() == 0:
        if args.eval_only:
            output_file_base = f"{args.type.split('_')[0]}_{args.checkpoint_dir.split('/')[-1]}_{args.val_data_root.split('/')[-1]}"
            avg_str = f'Eval top-1 accuracy: {sum(perf_list)/len(perf_list):.4f}%.'
            print(avg_str)

            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=eval_conf_mat, display_labels=cls_labels)
            _, ax = plt.subplots(figsize=(10, 10))
            cm_display.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal', values_format=None,)
            plt.savefig(os.path.join('eval_output/', f'{output_file_base}.png'))
            plt.close()
            # write to txt file
            with open(os.path.join('eval_output/', f'{output_file_base}.txt'), 'w') as f:
                f.write('  '.join(['fold-'+str(fi)+' '+str(x) for fi, x in enumerate(perf_list)]) + '\n')
                f.write(avg_str + '\n')
        else:
            # write final accuracy to file
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=all_conf_mat, display_labels=cls_labels)
            cm_display.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal', values_format=None)
            plt.savefig(os.path.join(osp.dirname(result_file), f'confusion_matrix_fold-all.png'))
            plt.close()
            result = f'\nTotal average accuracy for {args.nfold}-fold {args.type}: {np.array(performances).mean():.4f}'
            print(result)
            with open(result_file, 'a') as f:
                f.write(result)
            # calculate F1-score per class from confusion matrix
            # use weighted average F1-score, the weight of which are determined by the number of samples available in that class
            f1_score = np.zeros(CLS_NUM)
            wf1_score = np.zeros(CLS_NUM)
            weights = all_conf_mat.sum(axis=1) / all_conf_mat.sum()
            for ci in range(CLS_NUM):
                f1_score[ci] = 2 * all_conf_mat[ci, ci] / (all_conf_mat[ci, :].sum() + all_conf_mat[:, ci].sum())
                wf1_score[ci] = f1_score[ci] * weights[ci]
            precision = np.nanmean(np.diag(all_conf_mat) / all_conf_mat.sum(axis=0))
            recall = np.nanmean(np.diag(all_conf_mat) / all_conf_mat.sum(axis=1))
            f1_score = np.nan_to_num(f1_score, nan=0.)
            wf1_score = np.nan_to_num(wf1_score, nan=0.)
            # calculate the difference between the lowest and the highest accuracy
            min_max = np.array(performances).max() - np.array(performances).min()
            #assert (f1_score>0).all() and (wf1_score>0).all()
            with open(result_file, 'a') as f:
                # write the f1-score
                f.write('\nF1-score per class: ' + ' '.join([f'{x:.4f}' for x in f1_score]))
                f.write(f'\nPrecision: {precision:.4f}')
                f.write(f'\nRecall: {recall:.4f}')
                f.write(f'\nAverage F1-score: {f1_score.mean():.4f}')
                # write the weighted f1-score
                f.write('\nWeighted F1-score per class: ' + ' '.join([f'{x:.4f}' for x in wf1_score]))
                f.write(f'\nAverage weighted F1-score: {wf1_score.sum():.4f}')
                f.write(f'\nMin-Max difference: {min_max:.4f}')
        
        
    return 

def evaluate(args: argparse.ArgumentParser, model: torch.nn.Module, loader: torch.utils.data.DataLoader,
             writer=None, global_step=0,):
    tot, hit1, = 0, 0,
    eval_st = datetime.now()
    # neg_pairs = []
    # calculate sensitivity & specificity using confusion matrix
    conf_mat = torch.zeros((CLS_NUM, CLS_NUM)).cuda() # | true, -> prediction
    for idx, (data, labels, _) in enumerate(loader):
        data, labels = data.cuda(), labels.cuda()
        # assert data.size(0) == 1
        # vidname = names[0]
        # if data.ndim == 6:
        #     data = data[0] # now the first dimension is number of views

        with torch.no_grad():
            logits, _ , _ = model(data)
            # if args.sigmoid_loss:
            #     scores = torch.sigmoid(logits)
            # else:
            scores = logits.softmax(dim=-1)

        tot += data.size(0)
        for ind in range(data.size(0)):
            conf_mat[labels[ind], scores[ind].topk(1)[1]] += 1
            hit1 += (scores[ind].topk(1)[1] == labels[ind]).sum().item()

        # if scores.topk(1)[1] != labels:
        #     # [name, ground-truth, label]
        #     neg_pairs.append([vidname, labels.item(), scores.topk(1)[1].item()])

        if tot % 20 == 0:
            print(f'[Evaluation] num_samples: {tot}  '
                f'ETA: {(datetime.now() - eval_st) / idx * (len(loader) - idx)}  '
                f'cumulative_acc1: {hit1 / tot * 100.:.2f}%')

    sync_tensor = torch.LongTensor([tot, hit1]).cuda()
    dist.all_reduce(sync_tensor)
    tot, hit1 = sync_tensor.cpu().tolist()
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar('test/accuracy', hit1 / tot * 100, global_step)
    print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%')
            
    return hit1 / tot * 100, conf_mat


if __name__ == '__main__': 
    global CLS_NUM
    CLS_NUM = 0 # the number of classes
    main()
        

