# implementation of importance-weighted aggregation
# try aggregate pretrained +kapt models for better results

import argparse
import yaml

import torch
torch.manual_seed(0) 
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import sys
sys.path.append('./')
sys.path.append('./training/')
sys.path.append('./utils/')
import os
sys.path.insert(0, os.getcwd())
import os.path as osp
import time
from collections import defaultdict, OrderedDict
import yaml

from tqdm import tqdm

import video_dataset
from training.VitaCLIP_model import VitaCLIP
from training.VitaCLIP_vision_encoder import CLIPVisionEncoder
from training.VitaCLIP_text_encoder import CLIPTextEncoder, TextPromptLearner

import utils.aux_numpy as aux_np

def main(args):
    # =========> Setup the device <========= #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # =========> Load the checkpoints with configurations <========= #
    # -----> Text encoder
    # construct CLIP Text Encoder (frozen in training across all configurations)
    textEncoder = CLIPTextEncoder(
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    # load pretrained pretrained CLIP weights
    assert osp.isfile(args.clip_checkpoint)
    ckpt = torch.load(args.clip_checkpoint)
    te_state_dict = {}
    for n, params in ckpt.items():
        if 'textual' in n:
            te_state_dict[n.replace('textual.', '')] = params
    textEncoder.load_state_dict(te_state_dict, strict=True)
    textEncoder.eval()

    # -----> Video Encoder
    # construct CLIP Vision Encoder (will load the weights from the corresponding checkpoint)
    videoEncoder = CLIPVisionEncoder(
            # data shape
            input_size=(224, 224),
            num_frames=args.num_frames,
            # model def
            feature_dim=768,
            patch_size=(16, 16),
            num_heads=12,
            num_layers=12,
            mlp_factor=4.0,
            embed_dim=512,
            # use summary token
            use_summary_token=True,
            # use local prompts
            use_local_prompts=True,
            # use global prompts
            use_global_prompts=True,
            num_global_prompts=8,
        )

    # -----> Vita-CLIP Baseline
    # construct baseline model
    if args.use_text_features:
        vtclip = VitaCLIP(
            backbone_path=args.clip_checkpoint,
            # data shape
            input_size=(224, 224),
            num_frames=args.num_frames,
            # exper setting
            cls_type='diag',
            num_classes=args.num_classes,
            # use summary token
            use_summary_token=True,
            # use local prompts
            use_local_prompts=True,
            # use global prompts
            use_global_prompts=True,
            num_global_prompts=8,
            # use text prompt learning
            use_text_prompt_learning=False,
            text_prompt_classes_path='classes/diag_classes.txt',
            text_prompt_CSC=True,
            # zeroshot eval
            zeroshot_evaluation=False,
            zeroshot_text_features_path='',
            # support memory
            use_support_memory=False, 
        )
            
    nfold = 10
    info_dict = OrderedDict({
        'model_path': defaultdict(dict),
        'text_features': defaultdict(dict),
        'G_vector': defaultdict(dict),
        'F_scalar': defaultdict(dict),
        'F_vector': defaultdict(dict),
        'label_val': {},
        'vf_val': defaultdict(dict),
    })
    performance = []
    conf_mat = None
    n_classes = 5
    cls_type = 'diag'
    # get the predictions on the source (training) and target (validation) data
    # load the precompted text features / matrices if exits
    import joblib
    if osp.isfile('info_dict.json'):
        info_dict = joblib.load('info_dict.json')
    dict_names = []
    for nf in tqdm(range(nfold)):      
        for model_id, md in enumerate(args.model_dir):
            name_in_dict = osp.basename(md)
            if nf==0:
                dict_names.append(name_in_dict)
            if conf_mat is None:
                conf_mat = torch.zeros(n_classes, n_classes).to(device)
            if len(info_dict['text_features'][name_in_dict])==nfold: continue
            if not osp.isdir(md):
                if nf==0: dict_names.pop(-1)
                continue
            with open(osp.join(md, 'config.yaml'), 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            assert config['nfold']==nfold, 'All models must have the same number of folds!'
            # ----> initialize the text encoder
            text_prompt_classes_path = config['text_prompt_classes_path']
            with open(text_prompt_classes_path, 'r') as f:
                classes = f.read().strip().split('\n')
            classes = [x for x in classes if x[0] != '*']
            assert len(classes)==n_classes
            assert config['type']==cls_type
            try:knowledge_version = config['knowledge_version']
            except:pass
            promptLearner = TextPromptLearner(
                classnames=classes,
                text_model=textEncoder.cpu(),
                num_prompts=config['text_num_prompts'],
                prompts_init=config['text_prompt_init'],
                CSC=config['text_prompt_CSC'],
                ctx_pos=config['text_prompt_pos'],
                cls_type=cls_type,
                knowledge_version=knowledge_version,
            )
            # load the model state_dict
            ckpt_path = osp.join(md, f'fold_{nf}/fold-{nf}-best.pth')
            assert osp.isfile(ckpt_path), 'checkpoint of pretrained Vita-CLIP not found!'
            ckpt = torch.load(ckpt_path, map_location=device)
            # -----> initialize Vita-CLIP prompt learner
            # load pretraiend weights &
            state_dict = ckpt['model']
            # retrieve the learned per-class embeddings (8 tokens per class)
            pl_state_dict = {}
            for n, params in state_dict.items():
                if 'prompt_learner' in n:
                    pl_state_dict[n.replace('module.prompt_learner.', '')] = params
            assert len(pl_state_dict)>0
            promptLearner.load_state_dict(pl_state_dict, strict=True)
            promptLearner.eval()
            promptLearner.to(device)
            # forward the text encoder to generate eventual tokens
            with torch.no_grad():
                prompts = promptLearner()
                tokenized_prompts = promptLearner.tokenized_prompts.to(device)
                textEncoder = textEncoder.to(device)
                text_features = textEncoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # ----> load the weights of the video encoder
            ve_state_dict = {}
            # baseline_ckpt = osp.join(args.vitaclip_checkpoint_dir, f'fold_{nf}/fold-{nf}-best.pth')
            # state_dict = torch.load(baseline_ckpt, map_location=device)['model']
            for n, params in state_dict.items():
                if 'visual' in n:
                    ve_state_dict[n.replace('module.visual.', '')] = params
            assert len(ve_state_dict)>0
            videoEncoder.load_state_dict(ve_state_dict, strict=True)
            videoEncoder.eval()
            videoEncoder.to(device)
            # get the logits scale
            # logit_scale = state_dict['module.logit_scale'].clone().detach().to(device)
            # ----> calculate the estimated probability on training set
            # load data of that fold
            args.data_root = f"datasets/miccai_10_fold/chunks_{int(nf)}/"
            args.val_list_path = osp.join(args.data_root, f'train_{cls_type}.csv')
            args.batch_size = 8 # to accelerate the process
            source_loader = video_dataset.create_val_loader(args)
            # initialize the F matrix
            F_mat = torch.empty((0, n_classes)).to(device)
            for idx, (data, labels, _) in enumerate(source_loader):
                data, labels = data.to(device), labels.to(device)
                # assert data.size(0) == labels.size(0)==1
                with torch.no_grad():
                    video_features = videoEncoder(data)
                    video_features = video_features / video_features.norm(dim=-1, keepdim=True)
                    logits = (video_features @ text_features.t()).reshape(-1, n_classes)
                    one_hot_labels = F.one_hot(labels, n_classes).float().reshape(-1, n_classes)
                    F_mat = torch.cat([F_mat, 1.* F.softmax(logits, dim=-1) * one_hot_labels], dim=0)
                if idx>0 and idx%50==0:
                    print(f'Processed {idx}/{len(source_loader)}')
            # ----> calculate the estimated probability on validation set     
            args.val_list_path = osp.join(args.data_root, f'val_{cls_type}.csv')
            args.batch_size = 8
            target_loader = video_dataset.create_val_loader(args)
            G_mat = torch.empty((0, n_classes)).to(device)
            # save the labels
            label_val = torch.empty((0,)).to(device)
            vf_val = torch.empty((0, video_features.size(-1))).to(device)
            for idx, (data, labels, _) in enumerate(target_loader):
                data, labels = data.to(device), labels.to(device)
                with torch.no_grad():
                    video_features = videoEncoder(data)
                    video_features = video_features / video_features.norm(dim=-1, keepdim=True)
                    logits = (video_features @ text_features.t()).reshape(-1, n_classes)
                    G_mat = torch.cat([G_mat, logits], dim=0)
                    label_val = torch.cat([label_val, labels], dim=0)
                    vf_val = torch.cat([vf_val, video_features], dim=0)
            # save the results to the info dict
            info_dict['text_features'][name_in_dict][f'fold{nf}'] = text_features.detach().cpu().numpy()
            info_dict['model_path'][name_in_dict][f'fold{nf}'] = md
            info_dict['G_vector'][name_in_dict][f'fold{nf}'] = G_mat.detach().cpu().numpy()
            info_dict['F_scalar'][name_in_dict][f'fold{nf}'] = (F_mat.detach()/F_mat.shape[0]).sum(-1).sum(0).cpu().numpy()
            info_dict['F_vector'][name_in_dict][f'fold{nf}'] = F_mat.detach().cpu().numpy()
            info_dict['label_val'][f'fold{nf}'] = label_val.cpu().numpy().astype(int)
            info_dict['vf_val'][name_in_dict][f'fold{nf}'] = vf_val.cpu().numpy()

        # =========> Perform per-fold evaluation <========= #
        num_model = model_id + 1
        # load pre-trained baseline vita-clip
        if args.use_text_features and not args.use_separate_video_encoder:
            ckpt = torch.load(osp.join(args.vitaclip_checkpoint_dir, f'fold_{nf}/fold-{nf}-best.pth'), map_location='cpu')
            renamed_ckpt = OrderedDict((k[len("module."):], v) for k, v in ckpt['model'].items() if k.startswith("module."))
            vtclip.load_state_dict(renamed_ckpt, strict=False)
            vtclip.eval()
            vtclip.to(device)
        else:
            vtclip = None
        matrix_G = np.zeros((num_model, num_model))
        num_sample = len(info_dict['G_vector'][dict_names[0]][f'fold{nf}'])
        for i, iname in enumerate(dict_names):
            for j, jname in enumerate(dict_names):
                # calculate matrix G of that fold
                matrix_G[i,j] = (info_dict['G_vector'][iname][f'fold{nf}'] * info_dict['G_vector'][jname][f'fold{nf}']).sum(-1).sum(0)/num_sample
        # inverse matrix G using singular value analysis
        matrix_G_inv = aux_np.pinv_with_singular_values(matrix_G, num_singular_values=-1, rcond=args.rcond)
        # concatenate vectors to make the matrix F
        matrix_F = np.array([info_dict['F_scalar'][x][f'fold{nf}'] for x in dict_names])
        aggregate_weights = matrix_G_inv @ matrix_F
        #aggregate_weights = np.exp(aggregate_weights)/np.exp(aggregate_weights).sum()
        # get unit vector
        # aggregate_weights = aggregate_weights / np.linalg.norm(aggregate_weights)
        if args.use_text_features:
            text_features = aggregate_weights[:,None,None] * np.concatenate([info_dict['text_features'][x][f'fold{nf}'].reshape(1,n_classes,-1)\
                                                                for x in dict_names], axis=0)
            # =====> zero-shot baseline using the aggregated text features <===== #
            text_features = torch.from_numpy(text_features).float().to(device).sum(0)/aggregate_weights.sum()
            if vtclip is not None:
                vtclip.zeroshot_evaluation = True
                vtclip.text_features = text_features
                # load (validation) data of that fold
                args.batch_size = 8
                args.data_root = f"datasets/miccai_10_fold/chunks_{int(nf)}/"
                args.val_list_path = osp.join(args.data_root, f'val_{cls_type}.csv')
                data_loader = video_dataset.create_val_loader(args)
                tot, hit1 = 0, 0
                for data, labels, _ in data_loader:
                    data, labels = data.to(device), labels.to(device)
                    with torch.no_grad():
                        logits, _, _ = vtclip(data)
                        scores = logits.softmax(dim=-1).argmax(-1)
                    
                    tot += scores.shape[0]
                    for ind in range(scores.shape[0]):
                        hit1 += (scores[ind] == labels[ind])
                        conf_mat[labels[ind], scores[ind]] += 1
            else:
                # use precomputed video features and calculate the cosine similarity with text features
                video_features = aggregate_weights[:,None,None] * np.concatenate([info_dict['vf_val'][x][f'fold{nf}'].reshape(1,-1,video_features.size(-1)) \
                    for x in dict_names], axis=0)
                video_features = torch.from_numpy(video_features.sum(0)/aggregate_weights.sum()).float().to(device)
                with torch.no_grad():
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    video_features = video_features / video_features.norm(dim=-1, keepdim=True)
                    logits = (video_features @ text_features.t())
                    scores = logits.softmax(dim=-1).argmax(-1).cpu().numpy()
                gts = info_dict['label_val'][f'fold{nf}']
                hit1 = (scores == gts).sum()
                tot = len(gts)
                # update confusion matrix
                for ind in range(tot):
                    conf_mat[gts[ind], scores[ind]] += 1
        else:
            # calculate linear combination of the predicted probabilities
            scores = aggregate_weights[:,None,None] * np.concatenate([info_dict['G_vector'][x][f'fold{nf}'].reshape(1,-1,n_classes) \
                for x in dict_names], axis=0)
            scores = scores.sum(0)
            preds = scores.argmax(-1)
            # use precomputed ground-truth label to get the performance
            gts = info_dict['label_val'][f'fold{nf}']
            hit1 = (preds == gts).sum()
            tot = len(gts)
            # update confusion matrix
            for ind in range(tot):
                conf_mat[gts[ind], preds[ind]] += 1
            
        try:perf = (hit1/tot).cpu().numpy()
        except: perf = hit1/tot
        performance.append(perf)
        print(f'Fold {nf} accuracy: {perf}')
    
    # calculate overall F1-score from confusion matrix
    conf_mat = conf_mat.cpu().numpy()
    precision = np.diag(conf_mat)/conf_mat.sum(0)
    recall = np.diag(conf_mat)/conf_mat.sum(1)
    f1 = 2*precision*recall/(precision+recall)
    # weighted F1-score
    f1[np.isnan(f1)] = 0
    wf1 = (f1*conf_mat.sum(1)).sum()/conf_mat.sum()
    if args.use_text_features:
        file_name = 'aggregate_results'
        if args.use_separate_video_encoder:
            file_name += '_sepVE'
    else:
        file_name = 'aggregate_results_noTF'

    with open(file_name+'.txt', 'a') as f:
        # add the datetime of the test
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        f.write(f'F1-score: {f1.mean()}\n')
        f.write(f'Weighted F1-score: {wf1}\n')
        f.write(f'Accuracy: {np.array(performance).mean()}\n')
    
    # save confusion matrix with 
    print('F1-score:', f1.mean())
    print('Weighted F1-score:', wf1)
    print('Accuracy:', np.array(performance).mean())
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    video_dataset.setup_arg_parser(parser)
    # checkpoints
    parser.add_argument('--model_dir', action='append', type=str, default=[], help='path to the model directory')
    parser.add_argument('--clip_checkpoint', type=str, default='pretrained/clip_pretrained.pth',)
    parser.add_argument('--vitaclip_checkpoint_dir', type=str, default='pretrained/baseline/',)

    # parameter for aggregation
    parser.add_argument('--use_text_features', action='store_true', help='use text features for aggregation')
    parser.add_argument('--num_classes', type=int, default=5,)
    parser.add_argument('--use_separate_video_encoder', action='store_true', \
                        help='use separate video encoder for each model')
    # parser.add_argument('--num_singular_values', type=int, default=800,)
    parser.add_argument('--rcond', type=float, default=1e-1,)
    parser.add_argument('--eps', type=float, default=0.02,)

    args = parser.parse_args()

    main(args)
