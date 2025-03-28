# analysis the segmentation knowledge
# the associated classification accuracy

import os, sys
import os.path as osp
sys.path.insert(0, os.getcwd())
import pandas as pd
import yaml
from tqdm import tqdm

import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

import numpy as np
import torch

import video_dataset
from VitaCLIP_model import VitaCLIP

from collections import OrderedDict, defaultdict

CLS_NUM = 0

keywords = {
    'updrs': [
        ['Normal Gait Pattern',
        'No Shuffling or Dragging of Feet',
        'Normal Arm Swing',
        'No Balance Issues',
        'No Assistive Device Required',
        'Normal Speed and Rhythm',
        'No Freezing of Gait',],
        ['Minimal Impairment', 'Occasional Slight Issues', 'Normal Speed and Rhythm', 'No Assistive Device Required', 'No Falling', ],
        ['Mild Impairment','Possible Reduced Arm Swing','Slight Slowness or Shuffling',
         'Mildly Irregular Steps','No Need for Assistive Device', 'No Falling'],
        ['Moderate Impairment', 'Slowness or Shuffling', 'Frequent Freezing Episodes', 'Use of Assistive Devices', 
         'Irregular Steps and Reduced Arm Swing', 'Possible Balance Problems', 'Possible Independent Walking'],
    ],
    'diag':[
        ['General Stable Gait Patterns', 'Longer Stride Length than in DLB and AD', 'Regular and Consistent Cadence',
         'Faster and More Consistent Speed than in AD and DLB', 'Even Weight Distribution and Movement','Consistent Rhythm',
         'Arm Swing Naturally Synchronized with Leg Movements'],
        ['More Noticeable Gait Changes than in Early AD', 'Slight Speed Reduction', 'Minor Balance Issues', 'Less Fluidity than Normal',
         'Occasional Hesitations in Initiating Movement', 'Slightly Reduced Arm Swing', 'Less Severe Mobility Impairment than Severe DLB',],
        ['Less Pronounced Gait Changes than in Early DLB', 'Slight Speed Reduction', 'Minor Decrease in Fluidity', 'Mild Balance Problems in Complex Conditions',
         'Less Pronounced Changes Compared with Early DLB', 'Less Severe Gait Impairment than Severe AD'],
        ['More Severe than Early DLB', 'Shuffling Gait', 'En Bloc Turning', 'Significant Balance Issues', 
         'Freezing Movement or Frequent Hesitation'],
        ['More Severe than Early AD', 'Greatly Reduced Speed and Irregular steps', 'Significant Balance Issue',
         'Loss the Ability of Independent Walking', 'Fewer freezing episodes Compared with DLB', 'Profound Mobility Impairment and caregiver dependence'],
    ]
}

# Function to wrap text based on bar width
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

def main(args):
    "Load the 10-fold checkpoints and eval the per-descriptor classification precision"
    config_fp = osp.join(args.model_dir, 'config.yaml')
    assert osp.isfile(config_fp), 'config file not found'
    # load the model configuration
    with open(config_fp, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.type = config['type']
    assert osp.isfile(config['text_prompt_classes_path']), 'class name file not found'
    # load the class names with per-class descriptors into dictionary
    descriptor_dir = osp.join('data', f"ke_{config['type']}")
    if config['use_descriptor']:
        desc_format = 'descriptor_{:01d}.txt'
    else:
        desc_format = 'simQdesc_v{:01d}.txt'
    
    with open(config['text_prompt_classes_path'], 'r') as f:
        for ele in f:
            global CLS_NUM
            if ele.strip()[0] == '*':
                continue
            CLS_NUM += 1

    class_descriptors = {ncls: {} for ncls in range(CLS_NUM)}
    num_classes = CLS_NUM if config['use_descriptor'] else 5
    for ncls in range(num_classes):
        if config['use_descriptor']:
            descriptor_file = osp.join(descriptor_dir, desc_format.format(ncls))
        else:
            descriptor_file = osp.join(descriptor_dir, desc_format.format(ncls+1))
        
        with open(descriptor_file, 'r') as f:
            for idl, _ in enumerate(f):
                if config['use_descriptor']:
                    class_descriptors[ncls][idl] = [] # store per-fold precision
                else:
                    class_descriptors[idl][ncls] = []# store per-fold precision

    model = VitaCLIP(
            # load weights
            backbone_path=config['backbone_path'],
            # data shape
            input_size=(config['spatial_size'], config['spatial_size']),
            num_frames=config['num_frames'],
            use_fp16=config['fp16'],
            # exper setting
            cls_type=config['type'],
            num_classes=CLS_NUM,
            # model def
            feature_dim=config['feature_dim'],
            patch_size=(config['patch_size'], config['patch_size']),
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            mlp_factor=config['mlp_factor'],
            embed_dim=config['embed_dim'],
            # use summary token
            use_summary_token=config['use_summary_token'],
            # use local prompts
            use_local_prompts=config['use_local_prompts'],
            # use global prompts
            use_global_prompts=config['use_global_prompts'],
            num_global_prompts=config['num_global_prompts'],
            # use text prompt learning
            use_text_prompt_learning=config['use_text_prompt_learning'],
            text_context_length=config['text_context_length'],
            text_vocab_size=config['text_vocab_size'],
            text_transformer_width=config['text_transformer_width'],
            text_transformer_heads=config['text_transformer_heads'],
            text_transformer_layers=config['text_transformer_layers'],
            text_num_prompts=config['text_num_prompts'],
            text_prompt_pos=config['text_prompt_pos'],
            text_prompt_init=config['text_prompt_init'],
            text_prompt_CSC=config['text_prompt_CSC'],
            text_prompt_classes_path=config['text_prompt_classes_path'],
            knowledge_version=config['knowledge_version'],
            use_descriptor=config['use_descriptor'],
            token_wise_mlp=config['token_wise_mlp'],
            # zeroshot eval
            # support memory
            use_support_memory=config['use_support_memory'],
            detach_features=config['detach'],
            memory_batch_size=config['mem_batch_size'],
            class_wise_mlp=config['class_wise_mlp'],
            # loss params
            use_sigmoid_loss=config['sigmoid_loss'],
    )

    nfold = config['nfold']
    checkpoint_format = osp.join(args.model_dir, 'fold_{:01d}', 'fold-{:01d}-best.pth')
    for nf in tqdm(range(nfold)):
        checkpoint_path = checkpoint_format.format(nf, nf)
        try:
            assert osp.isfile(checkpoint_path)
        except AssertionError:
            print('checkpoint file not found for fold {:01d}!!'.format(nf))
            continue
        ckpt = torch.load(checkpoint_path, map_location='cpu')['model']
        renamed_ckpt = OrderedDict((k[len("module."):], v) for k, v in ckpt.items() if k.startswith("module."))
        model.load_state_dict(renamed_ckpt, strict=False)

        print('loading model from checkpoint: {}'.format(checkpoint_path))
        print('----------------------------------------------------')
        model.cuda()
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        # load evaluation data
        args.eval_list_path = osp.join(args.data_dir, 'chunks_{:01d}'.format(nf), f"val_{config['type']}.csv")
        args.eval_data_root = osp.join(args.data_dir, 'chunks_{:01d}'.format(nf))
        eval_loader = video_dataset.create_eval_loader(args)

        precisions = {k:{sk:[] for sk in range(len(class_descriptors[k]))} for k in range(CLS_NUM)}
        for _, (data, labels, _) in enumerate(eval_loader):
            data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():
                logits, _ , _ = model(data, desc_wise=True,)
                assert isinstance(logits, list)
                pred = torch.zeros((data.size(0),CLS_NUM)).to(logits[0].device)
                scores = torch.zeros((data.size(0),CLS_NUM)).to(logits[0].device)
                for i in range(len(logits)): # enumerate over classes
                    pred[:, i] = torch.argmax(logits[i], dim=-1)
                    scores[:,i] = logits[i].max(dim=-1)[0]
                scores = torch.argmax(scores, dim=-1).detach().cpu().numpy()
                # calculate true positive of that class / prediction of that class
                pred = pred.detach().cpu().numpy()
                for b in range(data.size(0)):
                    if scores[b] == labels[b]:
                        precisions[scores[b]][pred[b, scores[b]]].append(1)
                    else:
                        precisions[scores[b]][pred[b, scores[b]]].append(0)
        # calculate the per-class precision
        for ncls in range(CLS_NUM):
            for ind, _ in enumerate(class_descriptors[ncls]):
                if len(precisions[ncls][ind]) > 0:
                    class_descriptors[ncls][ind].append(np.mean(precisions[ncls][ind]))
                else:
                    class_descriptors[ncls][ind].append(0)
    # average over folds
    os.makedirs(args.output_dir, exist_ok=True)
    FS = 30
    for ncls in range(CLS_NUM):
        precisions = [round(np.mean(class_descriptors[ncls][i]),4)*100 for i in range(len(class_descriptors[ncls]))]
        if config['use_descriptor']:
            cls_keywords = keywords[config['type']][ncls]
        else:
            cls_keywords = [f'Segment {i}' for i in range(len(class_descriptors[ncls]))]

        # Create a horizontal bar chart using seaborn
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(y=cls_keywords, x=precisions, palette='pastel', ax=ax, width=0.8)

        # get the smallest width
        avg_width = np.max([p.get_width() for p in ax.patches if p.get_width() > 0])

        for i, p in enumerate(ax.patches):
            # if p.get_width() > 0:
            #     bar_width = p.get_width()
            #     temp_text = ax.text(0, 0, cls_keywords[i], fontsize=10, fontweight='bold')
            #     renderer = fig.canvas.get_renderer()
            #     bbox = temp_text.get_window_extent(renderer)
            #     temp_text.remove()
                
            #     # Estimate the number of characters that fit in the bar width
            #     char_width = bbox.width / len(cls_keywords[i])
            #     max_chars_per_line = int(bar_width / char_width)
            #     wrapped_text = wrap_text(cls_keywords[i], max_chars_per_line)
                
            #     ax.annotate(wrapped_text, 
            #                 (bar_width / 2, p.get_y() + p.get_height() / 2), 
            #                 ha='center', va='center', 
            #                 xytext=(0, 0),  # No offset
            #                 textcoords='offset points', 
            #                 color='black', fontsize=10, fontweight='bold')
            # Add keywords inside the bars
            if p.get_width() > 10:
                ax.annotate(f'{cls_keywords[i]}', 
                            (p.get_width() / 2, p.get_y() + p.get_height() / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0),  # No offset
                            textcoords='offset points', 
                            color='black', fontsize=FS,)
            else:
                ax.annotate(f'{cls_keywords[i]}', 
                            (avg_width / 2, p.get_y() + p.get_height() / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0),  # No offset
                            textcoords='offset points', 
                            color='black', fontsize=FS,)

        # Add precision values to the outside right of the bars
        for i, p in enumerate(ax.patches):
            if p.get_width() > 10:
                ax.annotate(f'{p.get_width():.2f}%', 
                            (p.get_width(), p.get_y() + p.get_height() / 2), 
                            ha='left', va='center', 
                            xytext=(5, 0),  # Offset to the right
                            textcoords='offset points', 
                            color='black', fontsize=FS,)
            else:
                ax.annotate(f'{p.get_width():.2f}%', 
                            (avg_width, p.get_y() + p.get_height() / 2), 
                            ha='left', va='center', 
                            xytext=(5, 0),  # Offset to the right
                            textcoords='offset points', 
                            color='black', fontsize=FS,) # use simple fold, not blidface

        # Remove x and y axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.show()
        plt.savefig(osp.join(args.output_dir, f"{config['type']}_{ncls}_per_descriptor_precision.png"))
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis the segmentation knowledge')
    video_dataset.setup_arg_parser(parser)
    # model directory
    parser.add_argument('--model_dir', type=str, default=None, help='model directory')
    # dataset directory
    parser.add_argument('--data_dir', type=str, default='./datasets/miccai_10_fold', help='dataset directory')
    # output xlsx file
    parser.add_argument('--output_dir', type=str, default='segmentation_analysis/', help='output xlsx file')

    parser.add_argument('--nfold', type=int, default=1, help='fold number')

    parser.add_argument('--type', type=str, default='', help='classification type')

    args = parser.parse_args()

    main(args)