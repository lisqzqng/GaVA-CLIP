# d.wang@unistra.fr
# prepare embedding
import os
import os.path as osp
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, osp.join(os.getcwd(), 'training/'))
import argparse

import torch
import numpy as np

import yaml
import json
import pickle
from tqdm import tqdm
import copy
from collections import OrderedDict, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from video_dataset.dataset import VideoDataset
from training.VitaCLIP_vision_encoder import CLIPVisionEncoder
from training.VitaCLIP_text_encoder import tokenize, CLIPTextEncoder
from training.kapt_head import ContextualPromptLearner

MIN_VID_NUM = 20

ke_path = 'data/figure.npy'
with open(ke_path, 'rb') as f:
    KE = np.load(f)

def visualize_number_distance(N=100):
    "visualize the distance between numeraical words"
    assert N<1000 and N>0
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dictionary of number words
    digits = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
               'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
               'eighteen', 'nineteen', 'twenty']
    tens_digits = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    hundred = ['hundred']
    
    # construct the texts from one to 101
    text_list = []
    for i in range(1, N):
        if i<21:
            text_list.append(digits[i-1])
        elif i<100:
            text_list.append(tens_digits[i//10-2]+'-'+digits[i%10-1])
        else:
            text_list.append(digits[i//100-1]+' '+hundred[0]+' '+tens_digits[i%100//10-2]+'-'+digits[i%10-1])

    text_format = {
        # 'speed':['the walking speed is {:s}', 'the speed normalized is {:s}'],
        # 'cadence': ['the number of steps per minute is {:s}',],
        'distance':['the difference in distance covered between a left step and a right step is {:s}',\
                    'the distance covered from the first contact of right foot to the first contact of the left foot is {:s}'],
        'percentage':['the percentage of the duration when only the left foot contacts the ground within one gait cycle is {:s}',\
                      'the percentage of the duration when the left foot is off the ground within the left walk cycle is {:s}'],
    }
    level_words = {
                # time, short v.s. long
                'speed':['very slow', 'slow', 'quick', 'fast', ], # 2:2 # 'snail-paced',  ==> slow v.s.fast
                'cadence': ['few', 'limited', 'minimal', 'numerous', 'maximal', 'frequent'], # 4:3 ==> minimal v.s. maximal
                'distance':['close', 'short', 'small', 'large', 'long', 'far'], # 2:2 ==> close v.s. long
                'percentage':['low', 'small', 'minor', 'large', 'high', 'major' ], # 3:3 ==> minor v.s. major
                }
    
    # encode the text
    clip_tokenizer = CLIPTextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    # load pretrained weights
    ckpt = torch.load('pretrained/clip_pretrained.pth')
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer.to(device)
    clip_tokenizer.eval()
    del ckpt, new_ckpt
    
    # sentence_base = 'the walking speed is'
    # base_embeds = clip_tokenizer.token_embedding(tokenize(sentence_base).to(device).reshape(1, -1))
    sentence_base = ' dogs' # 'The walking speed is'
    base_embeds = clip_tokenizer.token_embedding(tokenize(sentence_base).to(device).reshape(1, -1))
    embeds = []
    for idx, t in tqdm(enumerate(text_list)):
        tokenized_text = tokenize('This image has '+t+' dogs').to(device).reshape(1, -1)
        with torch.no_grad():
            prefix_embedding = clip_tokenizer.token_embedding(tokenized_text)
            # prefix_embedding = torch.cat([base_embeds, prefix_embedding], dim=0)
            embedding = clip_tokenizer(prefix_embedding, tokenized_text)
        embeds.append(embedding.cpu().numpy().reshape(-1))
    
    embeds = np.vstack(embeds)

    # compute the distance using cosine similarity
    normed_embeds = embeds/np.linalg.norm(embeds, axis=1, keepdims=True)
    similarity = normed_embeds @ normed_embeds.T
    
    # visualize the similarity as image
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(similarity, interpolation='nearest',cmap='gray', 
                origin='lower', vmin=similarity.min(), vmax=1)
    plt.title("Cosine Sim")
    # plt.savefig('number_similarity_gpt.png')
    plt.show()
    
    for k,v in text_format.items():
        for tf in v:
            text_list = []
            for lw in level_words[k]:
                text_list.append(tf.format(lw))
            embeds = []
            for t in tqdm(text_list):
                tokenized_text = tokenize(t).to(device).reshape(1, -1)
                with torch.no_grad():
                    prefix_embedding = clip_tokenizer.token_embedding(tokenized_text)
                    embedding = clip_tokenizer(prefix_embedding, tokenized_text)
                embeds.append(embedding.cpu().numpy().reshape(-1))
    
            # compute the distance using cosine similarity
            normed_embeds = embeds/np.linalg.norm(embeds, axis=1, keepdims=True)
            similarity = normed_embeds @ normed_embeds.T
            
            # visualize the similarity as image
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(similarity, interpolation='nearest',cmap='gray', 
                        origin='lower', vmin=similarity.min(), vmax=1)
            plt.title(f"{k}: Cosine Sim {' '.join(tf.split()[:10])}")
            # plt.savefig('number_similarity_gpt.png')
            plt.show()
            
            # compute the distance using euclidean distance
            ## generate the table of distance
            distances = np.zeros((len(text_list), len(text_list)))
            for ida, a in enumerate(normed_embeds):
                for idb, b in enumerate(normed_embeds):
                    distances[ida,idb] = np.linalg.norm(a-b)
            # visualize the distance as image
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(distances, interpolation='nearest',cmap='gray',
                        origin='lower', vmin=distances.min(), vmax=distances.max())
            plt.title(f"{k}: Euclidean distance {' '.join(tf.split()[:10])}")
            # plt.savefig('number_distance_gpt.png')
            plt.show()
    
    return


def visualize_pe_distance(N=150, d_model=512):
    "visualize the cosine similarity & euclidean distance between positional encoding"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # generate the positional encoding for n=1,...,N
    pe = torch.zeros(N, d_model)
    position = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    l2_norm = 0.5
    pe = pe/pe.norm(dim=-1, keepdim=True)*l2_norm

    # encode the text
    clip_tokenizer = CLIPTextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    # load pretrained weights
    ckpt = torch.load('pretrained/clip_pretrained.pth')
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer.to(device)
    clip_tokenizer.eval()
    del ckpt, new_ckpt
    
    sentence_base = 'the walking speed is'
    tokens = tokenize(sentence_base+' X').to(device).reshape(1, -1) # 1x77
    base_embeds = clip_tokenizer.token_embedding(tokens) # 1x77x512
    # compute the integral embedding with the `sentence_base` as text prefix
    # embeds = []
    # for n in range(N):
    #     prefix_embedding = base_embeds.clone()
    #     prefix_embedding[5,:] = pe[n,:].to(device)
    with torch.no_grad():
        embedding = clip_tokenizer(base_embeds, tokens)
    #     embeds.append(embedding.cpu().reshape(1,-1))
    
    # embeds = torch.cat(embeds, dim=0)
    
    # compute the distance using cosine similarity
    pe += embedding.cpu().numpy().reshape(1,-1)
    normalized_embed = pe/pe.norm(dim=1, keepdim=True)
    similarity = normalized_embed[:N,:] @ normalized_embed[:N,:].T
    similarity = similarity.numpy()
    
    # visualize the similarity as image
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(similarity, interpolation='nearest',cmap='gray', 
                   origin='lower', vmin=similarity.min(), vmax=1)
    plt.title('PE: Cosine similarity between number words')
    plt.savefig('number_similarity_pe.png')
    plt.show()

    # compute the distance using euclidean distance
    ## generate the table of distance
    distances = np.zeros((N, N))
    for ida, a in enumerate(normalized_embed):
        for idb, b in enumerate(normalized_embed):
            distances[ida,idb] = np.linalg.norm(a-b)
    # visualize the distance as image
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(distances, interpolation='nearest',cmap='gray',
                   origin='lower', vmin=distances.min(), vmax=distances.max())
    plt.title('PE: Euclidean distance between number words')
    plt.savefig('number_distance_pe.png')
    plt.show()
    
    return

def main(args):
    "Generate the embedding for text or visual data.\nNot computing the similarities (LC coeff.). \n Single GPU."
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    # =====> construct CLIP text encoder
    clip_tokenizer = CLIPTextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    # load pretrained weights
    ckpt = torch.load('pretrained/clip_pretrained.pth')
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer.to(device)
    clip_tokenizer.eval()
    del ckpt, new_ckpt
    # --------> process single text files (class discription in natrual language)
    if len(args.text_file) > 0:
        for text_file in args.text_file:
            text_list, embeddings = [], []
            assert osp.isfile(text_file), f'{text_file} not found!'
            with open(text_file, 'r') as f:
                for _, line in enumerate(f):
                    text_list.append(line.strip())
            for t in tqdm(text_list):
                tokenized_text = tokenize(t)
                tokenized_text = tokenized_text.to(device).reshape(1, -1)
                # encode the 512x1 text embedding
                with torch.no_grad():
                    prefix_embedding = clip_tokenizer.token_embedding(tokenized_text) # 1x77x512
                    text_embedding = clip_tokenizer(prefix_embedding, tokenized_text) # 1x512
                    
                embeddings.append(text_embedding.cpu().numpy().reshape(-1))
            
            if len(embeddings) > 1:
                with open(osp.join(args.output_dir, f"{osp.dirname(text_file).split('/')[-1]}.pkl"), 'wb') as f:
                    pickle.dump(embeddings, f)
    
    # ----> process list-of-text files
    if len(args.text_list_files)>0:
        for text_list_file in args.text_list_files:
            embeddings = []
            assert osp.isfile(text_list_file), f'{text_list_file} not found!'
            assert text_list_file.endswith('.json'), f'{text_list_file} is not a json file!'
            try:
                with open(text_list_file, 'r') as f:
                    text_list = json.load(f)
            except UnicodeDecodeError:
                with open(text_list_file, 'rb') as f:
                    text_list = pickle.load(f)
            assert isinstance(text_list, list), 'text_list should be a list of texts!'
            for text in tqdm(text_list):
                tokenized_text = tokenize(text)
                tokenized_text = tokenized_text.to(device).reshape(1, -1)
                with torch.no_grad():
                    prefix_embedding = clip_tokenizer.token_embedding(tokenized_text)
                    text_embedding = clip_tokenizer(prefix_embedding, tokenized_text)
                embeddings.append(text_embedding.cpu().numpy().reshape(-1))
                
            if len(embeddings) > 1:
                with open(osp.join(args.output_dir, f"{osp.basename(text_list_file).split('.')[0]}.pkl"), 'wb') as f:
                    pickle.dump(embeddings, f)
    
    if osp.isdir(args.vid_dir):
        # =====> construct Vita-CLIP visual encoder
        assert osp.isfile(args.checkpoint), 'checkpoint of pretrained Vita-CLIP not found!'
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        with open(osp.join('/'.join(osp.dirname(args.checkpoint).split('/')[:-1]), 'config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        text_prompt_init = config['text_prompt_init']
        # -----> initialize Vita-CLIP prompt learner
        # load pretraiend weights &
        # find the learned per-class embeddings (8 tokens per class)
        state_dict = ckpt['model']
        ctx_learnable = state_dict['module.prompt_learner.ctx'] # torch
        # get the eventual pre-class tokens using KAPT
        n_cls, n_token, out_dim = ctx_learnable.shape
        kapt_head = ContextualPromptLearner(
            use_cntn='cntn' in text_prompt_init,
            cntn_split='split' in text_prompt_init,
            uni_mlp='uni' in text_prompt_init,
            use_disc='disc' in text_prompt_init,
            out_dim=out_dim,
            n_cls=n_cls,
            n_tokens=n_token,
            cls_type='diag' if 'diag' in args.checkpoint else 'updrs', 
            knowledge_version=config['knowledge_version'],
            use_descriptor=config['use_descriptor'],
            token_wise_mlp=config['token_wise_mlp'],
            class_wise_mlp=config['class_wise_mlp'],
            )
        # get the module state dict
        kapt_state_dict = {}
        for n, params in state_dict.items():
            if 'prompt_learner.context_prompt_learner' in n:
                kapt_state_dict[n.replace('module.prompt_learner.context_prompt_learner.', '')] = params
        ################## !!! check and modify the dimension order !!!!! ##################
        try: del kapt_state_dict['cntn_embeds']
        except: pass
        kapt_head.load_state_dict(kapt_state_dict, strict=True)
        ## forward the kapt head to generate eventual tokens
        kapt_head.eval()
        kapt_head.to(device)
        
        ctx_learnable = ctx_learnable.to(device)
        with torch.no_grad():
            kepler_tokens = kapt_head.cntn_embeds
            learned_tokens = kapt_head(ctx_learnable,)
        
        # encode prefix embeddings to 1x512 text embedding
        tokenized_xs = tokenize(" ".join(["X"] * n_token))
        tokenized_xs = tokenized_xs.to(device).reshape(1, -1).long()
        # get discrete prompts
        discrete_text = kapt_head.cls_disc # [n_cls, n_kn] 
        prefix_text = [" ".join(["X"] * n_token)]*n_cls
        discrete_text = [p + " " + discrete_text[i] for i, p in enumerate(prefix_text)]
        discrete_tokens = torch.cat([tokenize(t).to(device).reshape(1, -1).long() for t in discrete_text], dim=0)
        with torch.no_grad():
            # get suffix embedding
            embeds_xs = clip_tokenizer.token_embedding(tokenized_xs).expand(n_cls, -1, out_dim) # 1x77x512
            prefix= embeds_xs[:,:1,:]
            suffix_xs = embeds_xs[:,1+n_token:,:]
            learned_embed = clip_tokenizer(torch.cat([prefix, learned_tokens, suffix_xs], dim=1), tokenized_xs) # 1x512
            kepler_embed = clip_tokenizer(torch.cat([prefix, kepler_tokens, suffix_xs], dim=1), tokenized_xs) # 1x512
            # get overall suffix
            discrete_embed_pre = clip_tokenizer.token_embedding(discrete_tokens) # n_clsx77x512
            discrete_embed_pre[:, :n_token, :] = learned_tokens
            discrete_embed = clip_tokenizer(discrete_embed_pre, discrete_tokens) # n_clsx512
            # normalize features
            learned_embed /= learned_embed.norm(dim=-1, keepdim=True)
            kepler_embed /= kepler_embed.norm(dim=-1, keepdim=True)
            discrete_embed /= discrete_embed.norm(dim=-1, keepdim=True)
            
        # -----> initialize Vita-CLIP visual encoder
        visual_encoder = CLIPVisionEncoder(
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
        # load pretrained weights for visual encoder
        new_ckpt = OrderedDict()
        for n, param in ckpt['model'].items():
            if 'visual' in n:
                new_ckpt[n.split('visual.')[-1]] = param
        visual_encoder.load_state_dict(new_ckpt, strict=True)
        visual_encoder.to(device)
        visual_encoder.eval()
        # get the ground truth labels
        csv_name = 'train_diag_3cls' if 'diag' in args.checkpoint else 'train_updrs'
        csv_name += '.csv'
        label_path = osp.join(args.vid_dir, csv_name)
        dataset = VideoDataset(
            list_path=label_path,
            data_root=args.vid_dir,
            num_spatial_views=1, num_temporal_views=1, random_sample=False, # originally, True
            auto_augment=None,
            interpolation=False,
            mirror=False,
            num_frames=args.num_frames,
            sampling_rate=1,
            spatial_size=224,
            mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            is_train=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            num_workers=1, pin_memory=False,
        )
        
        # =====> process the videos
        # initialize the dictionary to save the embeddings
        output_dict = {}
        output_dict['learned_embed'] = learned_embed.cpu().numpy().reshape(n_cls, -1)
        output_dict['kepler_embed'] = kepler_embed.cpu().numpy().reshape(n_cls, -1)
        output_dict['final_embed'] = discrete_embed.cpu().numpy().reshape(n_cls, -1)
        output_dict['image_embeddings'] = defaultdict(list)
        output_dict['vidname'] = defaultdict(list)
        for idx, (frames, label, vidname) in tqdm(enumerate(loader)):
            if idx>MIN_VID_NUM: break
            frames = frames.cuda()
            with torch.no_grad():
                video_features = visual_encoder(frames)
                # normalize features
                video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            
            output_dict['image_embeddings'][label.cpu().item()].append(video_features.cpu().numpy())
            output_dict['vidname'][label.cpu().item()].append(vidname[0])
            
        # make summary
        print('Summary of the embedding:')
        print('Number of videos:', len(output_dict['image_embeddings']))
        print('Number of videos per class:')
        for k, v in output_dict['image_embeddings'].items():
            print(f'{k}: {len(v)}')
        with open(osp.join(args.output_dir, osp.basename(args.checkpoint).replace('.pth', '.pkl')), 'wb') as f:
            pickle.dump(output_dict, f)

    return

if __name__ == '__main__':
    
    # visualize_number_distance(N=201)
    # visualize_pe_distance(N=100)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', 
                        help='pretrained Vita-CLIP model path')
    parser.add_argument('--num_frames', type=int, default=60,
                        help='number of frames of the clips to encode.') 
    parser.add_argument('--text_file', type=str, nargs='+', default='',
                        help='text file to encode (discrete embedding).')
    parser.add_argument('--text_list_files', type=str, nargs='+', default='',
                        help='list of texts to encode.')
    parser.add_argument('--vid_dir', type=str, default='', 
                        help='directory with videos to encode.')
    parser.add_argument('--output_dir', type=str, default='./data/embedding/', 
                        help='directory to save the encoded embeddings (textual / visual).')
    args = parser.parse_args()
    
    main(args)