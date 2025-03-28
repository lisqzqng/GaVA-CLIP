# visualization the span of text & visual embeddings

import os
import os.path as osp
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, osp.join(os.getcwd(), 'training/'))
import glob
import yaml

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import random
import pandas as pd
import pickle
from tqdm import tqdm
import copy

from collections import defaultdict, OrderedDict

from training.kapt_head import ContextualPromptLearner
from training.kapt_head_uniproj import ContextualPromptLearnerUP

# =========> Hyperparameters <========= #
CLASS_LABEL_DICT = {
    'updrs': ['Normal', 'Slight', 'Mild', 'Moderate',],
    'diag': ['Healthy', 'Early DLB', 'Early AD', 'Severe DLB', 'Severe AD',],
}
BASELINE_DICT = {
    'updrs': 'logs/updrs_0527-11_Floss/',
    'diag': 'logs/diag_0521-baseline/',
}
# colors to connect the dots, #RGB
Colors = [
        [0,1,0],  # green
        [0,0,1], # blue
        [1,0,0], # red
        [0,1,1],  # cyan
        [1,0,1],  # magenta
]

# =========> divers <========= #
def return_norm(x: np.ndarray):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def svd(X, n_components=3):
    # using SVD to compute eigenvectors and eigenvalues
    # TODO remove the mean of each modality ???
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]


def plot_scattered_cones(data_dict_list, is_text=False, type='pca'):
    """
    Args:\n
    \t data_dict_list: list of dict, each dict contains the following keys:\n
    \t\t `task_name`: str, name of the task\n
    \t\t `embed`: dict, each key is the name of the modality, each value is the embedding of the modality\n
    """
    print('classification task: ', ' '.join([d['task_name'] for d in data_dict_list]))
    type = type.lower()
    assert type in ['pca', 'umap', 'tsne'], 'type should be pca, umap, or tsne'
    total_feature_list = list()
    label_list = list()
    for expriment_idx in range(len(data_dict_list)):
        if is_text:
            data_dict_list[expriment_idx]['embed'].update(other_text_dict)
        for ind, (name, embed) in enumerate(data_dict_list[expriment_idx]['embed'].items()):
            if 'vidname' in name: continue
            # if 'proj' not in name and ind<1: continue
            total_feature_list.append(embed)
            labelname = data_dict_list[expriment_idx]['task_name'] + " " + name
            label_list.extend([labelname] * embed.shape[0])
        
    total_feature_np = np.concatenate(total_feature_list, axis=0) 
    total_feature_np = return_norm(total_feature_np) # L2-normalize

    if type == 'pca':
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(total_feature_np)
    elif type == 'umap':
        reducer = umap.UMAP(n_components=3)
        pca_result = reducer.fit_transform(total_feature_np)
    elif type == 'tsne':
        reducer = TSNE(n_components=2)
        pca_result = reducer.fit_transform(total_feature_np)

    df = pd.DataFrame()
    df['pca_one'] = pca_result[:,0]
    df['pca_two'] = pca_result[:,1]
    df['pca_three'] = pca_result[:,2]
    df['modality cones'] = label_list
    save = False
    if save:
        labls = [int(x.split('_')[-1]) for x in label_list]
        output = np.concatenate([pca_result, labls[:,None]], axis=-1)
        outfile = 'test.npy'
        with open(outfile, 'wb') as f:
            np.save(f, output)
    return df, pca_result
 
def pair_wise_similarity(base_embed, sub_embed_dict, base_name='metadata', save_format='{:s}'):
    "calculate the pair-wise similarity of the embeddings"
    data_dict = {} 
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    for embed_name, sub_embed in sub_embed_dict.items(): 
        sub_dim = sub_embed.shape[0]
        embed = np.concatenate([base_embed, sub_embed], axis=0)
        normed_feature = return_norm(embed)
        similarity = normed_feature @ normed_feature.T
        # similarity = similarity.ravel().squeeze().tolist()
        similarity = np.triu(similarity, k=1)
        valid_ids = np.triu(np.ones(similarity.shape[0], dtype=bool), k=1)
        print(embed_name, 'min similarity:', similarity[valid_ids].min())
        # split the matrix of similarity into base<->base & sub<->base
        base_ids, sub_ids = valid_ids.copy(), valid_ids.copy()
        base_ids[:, -sub_dim:] = False
        sub_ids[:, :-sub_dim] = False

        df1 = pd.DataFrame({base_name: similarity[base_ids].ravel().tolist()})
        df2 = pd.DataFrame({embed_name: similarity[sub_ids].ravel().tolist()})
        tmp_df = pd.concat([df1, df2], ignore_index=True, axis=1,)
        tmp_df.columns = [base_name, embed_name]
        print('mean base', df1[base_name].to_numpy().mean())
        print('mean base<->sub', df2[embed_name].to_numpy().mean())
        print(tmp_df[base_name].describe())
        print(tmp_df[embed_name].describe())
        
        plt.figure(figsize=(5,4))
        sns.histplot(data=tmp_df, alpha=0.2)
        plt.xlabel('')
        plt.ylabel('')
        
        plt.savefig(save_format.format(base_name, embed_name))
        # plt.show()

    return

# =========> visualize the knowledge embedding from KEPLER
def visualize_modality_gap():
    return

def svd_knowledge_embed(ctype='updrs',):
    "visualize inter-class relationship of KEPLER knowledge embeddings"
    # draw multiple svd visualizations together
    base_dir = f'./data/ke_{ctype}'
    embed_file_list = glob.glob(osp.join(base_dir, 'descriptor*.npy'))
    embed_all = np.empty((0, 768)) # save the embedding features of different classes
    class_indices = {}
    pre = 0
    embed_file_list.sort()
    for ef in embed_file_list:
        with open(ef, 'rb') as f:
            kepler_features = np.load(f)
        embed_id = osp.basename(ef).split('_')[1].split('.')[0]
        class_indices[int(embed_id)] = range(pre, kepler_features.shape[0]+pre)
        pre += kepler_features.shape[0]
        embed_all = np.concatenate([embed_all, kepler_features], axis=0)
        
    # perform SVD on the concatenated embeddings
    # results = svd(embed_all, n_components=3)
    reducer = umap.UMAP(n_components=3)
    results = reducer.fit_transform(embed_all)
    
    # visualize the reduced embeddings \
        # use different colors to represent different classes
    fig = plt.figure(figsize=(25, 30), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    # set fontsize in axis label tick
    ax.set_yticklabels(np.arange(int(results[:,0].min()), round(results[:,0].max()), 0.5), fontsize=6)
    ax.set_xticklabels(np.arange(int(results[:,1].min()), round(results[:,1].max()), 0.5), fontsize=6)
    ax.set_zticklabels(np.arange(int(results[:,2].min()), round(results[:,2].max()), 0.5), fontsize=6)
    colors = ['b', 'm', 'y','r','g',]
    for k, v in class_indices.items():
        ax.scatter(results[v, 0], results[v, 1], results[v, 2], c=colors[k], \
            label=CLASS_LABEL_DICT[ctype][k], s=20, linewidth=0, alpha=0.7)
        # for idv, vv in enumerate([*v]):
        #     ax.text(results[vv, 0], results[vv, 1], results[vv, 2], f'{k}:{idv}', size=8, zorder=1, color='b')
    plt.legend(fontsize=6)
    plt.show()
        
    return

def project_dscKapt(model_dir, with_baseline=False):
    """
    Visualize the projected the KEPLER descriptor embeddings,
     added with per-cls learnable params.\n
    Only the `context_prompt_learner` module is needed.\n
    
    Args:
    - model_dir: the directory of the trained model\n
    - with_baseline: whether to add the baseline embeddings
    """
    # check the model directory and get the configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = osp.join(model_dir, 'config.yaml')
    assert osp.isfile(model_config)
    with open(model_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # parse the configuration
    ctx_dim = 512
    ctx_init = config['text_prompt_init']
    ctx_init = ctx_init.lower().split('_')
    use_descriptor = config['use_descriptor']
    knowledge_version = config['knowledge_version']
    n_ctx = config['text_num_prompts']
    ctype = config['type']
    n_cls = 5 if ctype == 'diag' else 4
    nfold = config['nfold']
    # initialize the context prompt learner module
    context_prompt_learner = ContextualPromptLearner(
        use_cntn=True if 'cntn' in ctx_init else False,
        cntn_split=True if 'split' in ctx_init else False,
        uni_mlp=True if 'uni' in ctx_init else False,
        use_disc=True if 'disc' in ctx_init else False,
        emb_dim=ctx_dim//4,
        out_dim=ctx_dim,
        n_cls=n_cls,
        n_tokens=n_ctx,
        cls_type=ctype,
        knowledge_version=knowledge_version,
        use_descriptor=use_descriptor,
    )
    # feature dimension reduction
    reducer = umap.UMAP(n_components=3)
    # process by fold
    for nf in range(nfold):
        # load the module checkpoints
        checkpoint = osp.join(model_dir, f'fold_{nf}', f'fold-{nf}-best.pth')
        if not osp.isfile(checkpoint):
            continue
        cls_text_features = torch.load(checkpoint, map_location='cpu')['text_features'].numpy()
        checkpoint = torch.load(checkpoint, map_location='cpu')['model']
        renamed_ckpt = OrderedDict((k.replace('module.prompt_learner.context_prompt_learner.', ''), v) \
            for k, v in checkpoint.items() if 'context_prompt_learner' in k)
        # load renamed checkpoint to the context prompt learner
        try:
            context_prompt_learner.load_state_dict(renamed_ckpt, strict=True)
        except RuntimeError:
            del context_prompt_learner
            renamed_ckpt['cntn_embeds'] = renamed_ckpt['cntn_embeds'].permute(1, 0, 2)
            context_prompt_learner = ContextualPromptLearnerUP(
                use_cntn=True if 'cntn' in ctx_init else False,
                cntn_split=True if 'split' in ctx_init else False,
                uni_mlp=True if 'uni' in ctx_init else False,
                use_disc=True if 'disc' in ctx_init else False,
                emb_dim=ctx_dim//4,
                out_dim=ctx_dim,
                n_cls=n_cls,
                n_tokens=n_ctx,
                cls_type=ctype,
                knowledge_version=knowledge_version,
                use_descriptor=use_descriptor,                
            )
            context_prompt_learner.load_state_dict(renamed_ckpt, strict=True)
        context_prompt_learner.eval()
        context_prompt_learner.to(device)
        ctx = checkpoint['module.prompt_learner.ctx']
        # load pretrained baseline model
        if with_baseline:
            bsl_ckpt = osp.join(BASELINE_DICT[ctype], f'fold_{nf}', f'fold-{nf}-best.pth')
            bsl_tf= torch.load(bsl_ckpt, map_location='cpu')['text_features'].numpy()
        # save the embeddings and keep the class label information
        embed_all = np.empty((0, ctx_dim))
        class_indices = {}
        pre = 0
        with torch.no_grad():
            cntn_embed = context_prompt_learner(ctx.to(device))
            for idc in range(n_cls):
                # normalize the embeddings
                cntn_embed[idc] = cntn_embed[idc] / cntn_embed[idc].norm(dim=-1, keepdim=True)
                embed_all = np.concatenate([embed_all, cntn_embed[idc].mean(-2).cpu().numpy()], axis=0)
                class_indices[idc] = range(pre, embed_all.shape[0])
                embed_all = np.concatenate([embed_all, cls_text_features[idc][None,...]], axis=0)
                if with_baseline:
                    embed_all = np.concatenate([embed_all, bsl_tf[idc][None,...]], axis=0)
                pre = embed_all.shape[0]   
                        
        # visualize the embeddings
        results = reducer.fit_transform(embed_all)
    
        # visualize the reduced embeddings \
            # use different colors to represent different classes
        fig = plt.figure(figsize=(25, 30), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        # set fontsize in axis label tick
        ax.set_yticklabels(np.arange(int(results[:,0].min()), round(results[:,0].max()), 0.5), fontsize=6)
        ax.set_xticklabels(np.arange(int(results[:,1].min()), round(results[:,1].max()), 0.5), fontsize=6)
        ax.set_zticklabels(np.arange(int(results[:,2].min()), round(results[:,2].max()), 0.5), fontsize=6)
        colors = ['b', 'm', 'y','r','g',]
        for k, v in class_indices.items():
            ax.scatter(results[v, 0], results[v, 1], results[v, 2], c=colors[k], \
                label=CLASS_LABEL_DICT[ctype][k], s=5, linewidth=0, alpha=0.6)
            # add the class-wise text feature
            ax.scatter(results[v[-1]+1, 0], results[v[-1]+1, 1], results[v[-1]+1, 2], c=colors[k], \
                label=f'{CLASS_LABEL_DICT[ctype][k]}_tf', s=5, linewidth=1, alpha=0.6)
            if with_baseline:
                ax.scatter(results[v[-1]+2, 0], results[v[-1]+2, 1], results[v[-1]+2, 2], c=colors[k], \
                    label=f'{CLASS_LABEL_DICT[ctype][k]}_bsl', marker='*',linewidth=0.5, alpha=1.0)
            # for idv, vv in enumerate([*v]):
            #     ax.text(results[vv, 0], results[vv, 1], results[vv, 2], f'{k}:{idv}', size=8, zorder=1, color='b')
        plt.legend(fontsize=4, loc='best')
        plt.title(f'{ctype} embeddings from fold-{nf}')
        plt.show()
        
    return 

def vis_text_feature(model_dir, baseline_dir):
    """
    visualize the learned per-class text features,\n
    together with the baseline text features\n
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = osp.join(model_dir, 'config.yaml')
    baseline_config = osp.join(baseline_dir, 'config.yaml')
    assert osp.isfile(model_config) and osp.isfile(baseline_config)
    with open(model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(baseline_config, 'r') as f:
        baseline_config = yaml.load(f, Loader=yaml.FullLoader)
    # parser the common configuration
    ctype = model_config['type']
    nfold = model_config['nfold']
    n_cls = 5 if ctype == 'diag' else 4
    # feature dimension reduction
    # reducer = umap.UMAP(n_components=2)
    reducer = PCA(n_components=2)
    # process by fold
    for nf in range(nfold):
        # updrs fold-1, diag fold-5
        if ctype == 'updrs' and not (nf in [1,4]): continue
        elif ctype == 'diag' and not (nf == 5): continue
        # load the `text_features` inside the model checkpoint
        checkpoint = osp.join(model_dir, f'fold_{nf}', f'fold-{nf}-best.pth')
        if not osp.isfile(checkpoint):
            continue
        # load the baseline text features
        bsl_ckpt = osp.join(baseline_dir, f'fold_{nf}', f'fold-{nf}-best.pth')
        if not osp.isfile(bsl_ckpt):
            continue
        cls_text_features = torch.load(checkpoint, map_location='cpu')['text_features'].numpy()
        bsl_tf = torch.load(bsl_ckpt, map_location='cpu')['text_features'].numpy()
        # concatenate the text features and apply the feature dimension reduction
        embed_all = np.empty((0, 512))
        class_indices = {}
        with torch.no_grad():
            for idc in range(n_cls):
                embed_all = np.concatenate([embed_all, cls_text_features[idc][None,...]], axis=0)
                embed_all = np.concatenate([embed_all, bsl_tf[idc][None,...]], axis=0)
        results = reducer.fit_transform(embed_all)
        # visualize the embeddings, color the text features by class with different colors
        global CLASS_LABEL_DICT
        fig = plt.figure(figsize=(16, 9), dpi=300)
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        # Define y-tick positions
        yticks = np.arange(int(results[:, 1].min()-0.5), round(results[:, 1].max()+0.5), 0.5)
        ax.set_yticks(yticks)  # Set y-tick positions
        ax.set_yticklabels(yticks, fontsize=20) 
        # Define x-tick positions
        xticks = np.arange(int(results[:, 0].min()-0.5), round(results[:, 0].max()+0.5), 0.5)
        ax.set_xticks(xticks)  # Set x-tick positions
        ax.set_xticklabels(xticks, fontsize=20)
        # ax.set_zticklabels(np.arange(int(results[:,2].min()), round(results[:,2].max()), 0.5), fontsize=6)
        colors = ['b', 'y', 'm','r','g',]
        for k in range(n_cls):
            ax.scatter(results[2*k, 0], results[2*k, 1], c=colors[k], \
                label=f'CPT: {CLASS_LABEL_DICT[ctype][k]}', s=200, alpha=1.0)
            ax.scatter(results[2*k+1, 0], results[2*k+1, 1], c=colors[k], \
                label=f'Baseline: {CLASS_LABEL_DICT[ctype][k]}', marker='*',s=500, alpha=1.0)
        # plt.legend(fontsize=6, loc='best')
        plt.title(f'{ctype} text features from fold-{nf}')
        # plt.draw()
        plt.savefig(f'./images/{ctype}_text_features_fold-{nf}.png')

    return

def generate_html():
    """
    Generate the html file to visualize the embeddings
    """
    return

if __name__=='__main__':
    # svd_knowledge_embed('updrs')
    # svd_knowledge_embed('diag')
    # project_dscKapt('./logs/updrs_DscKAPT_Floss_cntn-split/', with_baseline=True)
    # project_dscKapt('./logs/diag_DscKAPT_cntn-uni-disc/', with_baseline=True)
    vis_text_feature('./logs/test/updrs_CPT_Floss_cntn-uni_v0/', './logs/test/updrs_baseline_Floss/')
    # vis_text_feature('./logs/test/diag_CPT_cntn-uni_v0/', './logs/test/diag_baseline/')
    # generate_html()
    

# embeddings from VLMs
pickle_paths = ['./data/embedding/updrs_part5_4orth_fold-0-best.pkl']
data_dict_list_all = defaultdict(list) # for cone effect visualization
UPDRS_DICT = {'0': 'normal', '1': 'slight', '2': 'mild', '3': 'moderate',}
DIAG_DICT = {'0': 'healthy', '1': 'DLB', '2':'AD'}
for pickle_path in pickle_paths:
    # =====> load the embeddings
    assert osp.isfile(pickle_path)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    if 'updrs' in pickle_path:
        ctype = 'updrs'
    elif 'diag' in pickle_path:
        ctype = 'diag'
    else:
        raise ValueError('Unknown data type.')
    with open(f'./data/embedding/discrete_{ctype}.pkl', 'rb') as f:
    # with open('./data/embedding/discrete_updrs.pkl', 'rb') as f:
        discrete_embed = pickle.load(f)
    # ctype = 'diag'
    print(data.keys())
    print(data['learned_embed'].shape)
    print(data['kepler_embed'].shape)
    print(data['final_embed'].shape)
    mmdata = data['image_embeddings']
    
    # ======> visualize the modality gap
    # generate figures
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    # sns.set_context("talk", font_scale=0.2)
    
    # Use PCA to reduce dimensionality
    # realize the PCA via SVD
    # construct list of video features
    video_features = []
    kepler_features = []
    learned_features = []
    discrete_features = []
    final_features = []
    color = []
    for k, v in mmdata.items():
        video_features.extend(list(v))
        kepler_features.extend([data['kepler_embed'][k]]*len(v))
        learned_features.extend([data['learned_embed'][k]]*len(v))
        final_features.extend([data['final_embed'][k]]*len(v))
        discrete_features.extend([discrete_embed[k]]*len(v))
        color.extend([Colors[k]]*len(v))
    
    # concetenation & normalization
    all_vid_features = return_norm(np.concatenate(video_features, axis=0))
    all_kepler_features = return_norm(np.array(kepler_features))
    all_learned_features = return_norm(np.array(learned_features))
    all_dicrete_features = return_norm(np.array(discrete_features))
    all_final_features = return_norm(np.array(final_features))

    
    if MG_VIS:
        # project the features onto the 2D plan orthogonal to the span of features (embeddings)
        modality_gap = {}
        modality_gap['kepler <-> video'] = svd(np.concatenate([all_vid_features, all_kepler_features], axis=0))
        modality_gap['learned <-> video'] = svd(np.concatenate([all_vid_features, all_learned_features], axis=0)) # kepler + learnable prompts
        modality_gap['discrete <-> video'] = svd(np.concatenate([all_vid_features, all_dicrete_features], axis=0)) # only the discrete prompts
        modality_gap['final <-> video'] = svd(np.concatenate([all_vid_features, all_final_features], axis=0)) # kepler + learnable prompts + discrete prompts
        # TODO: add the cat[learned prompts, discrete prompts] for ablation study !!
        
        # figures text embed v.s. video embed 
        plt.figure(figsize=(5, 4))
        for i, (k, v) in enumerate(modality_gap.items()):
            plt.subplot(2, 2, i+1)
            plt.scatter(v[:-all_vid_features.shape[0], 0], v[:-all_vid_features.shape[0], 1], s=1, c='red', label='video')
            plt.scatter(v[-all_vid_features.shape[0]:, 0], v[-all_vid_features.shape[0]:, 1], s=1, c='blue', label=k.split(' ')[0])
            plt.legend()
            plt.title(k)
            
            # connect the dots
            for j in range(len(all_vid_features)):
                plt.plot([v[j, 0], v[len(all_vid_features)+j, 0]], [v[j, 1], v[len(all_vid_features)+j, 1]], c=color[j], alpha=0.1)
                # fix the graduated scale  of the x axis
                
                
        plt.tight_layout()
        plt.savefig(f'./data/embedding/{ctype}_modality_gap.png')
        plt.show()

        if PROJECT:
            gp_modality_gap = OrderedDict()
            # process gait param - based text features
            kepler_features = OrderedDict()
            learned_features = OrderedDict()
            discrete_features = OrderedDict()
            final_features = OrderedDict()
            # initialize the pca
            pca = umap.UMAP(n_components=2)
            with torch.no_grad():
                for k, v in other_text_dict.items():
                    kname = int(k.split('_')[-1])
                    final_features[kname] = return_norm(tf_project(torch.from_numpy(data['final_embed'][kname]).float()).numpy())
                    discrete_features[kname] = return_norm(tf_project(torch.from_numpy(discrete_embed[kname]).float()).numpy())

            # compute pca analysis
            gp_modality_gap['final <-> gp'] = pca.fit_transform(np.concatenate([np.vstack(list(other_text_dict.values())), \
                np.array(list(final_features.values()))], axis=0))
            # gp_modality_gap['discrete <-> gp'] = pca.fit_transform(np.concatenate([np.vstack(list(other_text_dict.values())), \
            #     np.array(list(discrete_features.values()))], axis=0)) # only the discrete prompts

            # figures text embed v.s. gp-vased text embed 
            plt.figure(figsize=(5, 4))
            value_len = len(np.vstack(list(other_text_dict.values())))
            for i, (k, v) in enumerate(gp_modality_gap.items()):
                plt.subplot(1, 1, i+1)
                plt.scatter(v[:value_len, 0], v[:value_len, 1], s=1, c='black', label='gp')
                plt.scatter(v[value_len:, 0], v[value_len:, 1], s=1, c='yellow', label=k.split(' ')[0])
                plt.legend()
                plt.title(k)
                
                # connect the dots
                pre_len = 0
                for ii, (kk,vv) in enumerate(other_text_dict.items()):
                    assert ii==int(kk.split('_')[-1])
                    _len = len(vv)
                    for j in range(_len):
                        plt.plot([v[pre_len+j, 0], v[value_len+ii, 0]], [v[pre_len+j, 1], v[value_len+ii, 1]], c=Colors[ii], alpha=0.1)
                    pre_len += _len
                    
                    
            plt.tight_layout()
            plt.savefig(f'./data/embedding/{ctype}_GP_modality_gap.png')
            plt.show()
        # compare text embeds
        text_feature_gap = {}
        text_feature_gap['kepler <-> learned'] = svd(np.concatenate([data['kepler_embed'], data['learned_embed']], axis=0))
        text_feature_gap['kepler <-> discrete'] = svd(np.concatenate([data['kepler_embed'], discrete_embed], axis=0))
        text_feature_gap['learned <-> discrete'] = svd(np.concatenate([data['learned_embed'], discrete_embed], axis=0))
        text_feature_gap['final <-> discrete'] = svd(np.concatenate([data['final_embed'], discrete_embed], axis=0))
        
        # figures text embed
        plt.figure(figsize=(5, 4))
        for i, (k, v) in enumerate(text_feature_gap.items()):
            plt.subplot(2, 2, i+1)
            plt.scatter(v[:-data['kepler_embed'].shape[0], 0], v[:-data['kepler_embed'].shape[0], 1], s=1, c='red', label=k.split(' ')[0])
            plt.scatter(v[-data['kepler_embed'].shape[0]:, 0], v[-data['kepler_embed'].shape[0]:, 1], s=1, c='blue', label=k.split(' ')[-1])
            plt.legend()
            plt.title(k)
            
            # connect the dots
            for j in range(len(data['kepler_embed'])):
                plt.plot([v[j, 0], v[len(data['kepler_embed'])+j, 0]], [v[j, 1], v[len(data['kepler_embed'])+j, 1]], c=Colors[j], alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(f'./data/embedding/{ctype}_text_embed_gap.png') 
        plt.show()
    
    if CE_VIS:
        # add data of all modalites into the data list
        # for later visualization of cone effect
        # cone effect among text embeds
        textdata = copy.deepcopy(data)
        del textdata['image_embeddings']
        textdata.update({
            'discrete_embed': np.vstack(discrete_embed),
        })
        # textdata.update(other_text_dict)
        data_dict_list_all[f'{ctype}_text'] = [{'task_name': ctype,
                                        'embed': {},}] # textdata
        
        data['video_embed'] = all_vid_features
        del data['image_embeddings']
        del data['kepler_embed']
        del data['learned_embed']
        data_dict_list_all[f'{ctype}_text<->video'] = [{'task_name': ctype, 
                            'embed': {},}] # data
        
        
# =============================================> Cone Effect <=================================================== #
# visualize the span of text embeddings
# import plotly.graph_objects as go
# import plotly.express as px
if OTHER:
    # visualize the raw data using plotly
    file_names = ['data_dict_part5_4f_orth2',]
    for file_name in file_names:
        file_path = file_path_format.format(file_name)
        assert osp.isfile(file_path)
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        # create dataframe from data_dict
        # ----- > generate valid indices
        data = data_dict['embeds']
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        updrs_labels = data_dict['updrs'][:,0]
        diag_labels = data_dict['diag'][:,0]
        valid_updrs = [np.where(data_dict['updrs']==i)[0] for i in range(4)]
        valid_diag = [np.where(data_dict['diag']==i)[0] for i in range(3)]
        try:
            texts = data_dict['text']
        except:
            texts = ['N/A',] * updrs_labels.shape[0]
        # reduce the dimension from 512 to 3 using UMAP algorithm
        reducer = umap.UMAP(n_components=3,)
        data_updrs = reducer.fit_transform(data[np.concatenate(valid_updrs, axis=0)])
        # data_diag = reducer.fit_transform(data[np.concatenate(valid_diag, axis=0)])
        # create dataframe
        dict_updrs = {
            'component 1': data_updrs[:, 0],
            'component 2': data_updrs[:, 1],
            'component 3': data_updrs[:, 2],
            'updrs': updrs_labels[np.concatenate(valid_updrs, axis=0)].astype(int),
            'text': np.array(texts)[np.concatenate(valid_updrs, axis=0)].tolist(),
        }
        # dict_diag = {
        #     'component 1': data_diag[:, 0],
        #     'component 2': data_diag[:, 1],
        #     'component 3': data_diag[:, 2],
        #     'diag': diag_labels[np.concatenate(valid_diag, axis=0)].astype(int),
        #     'text': np.array(texts)[np.concatenate(valid_diag, axis=0)].tolist(),
        # }
        df0 = pd.DataFrame(dict_updrs)
        # df1 = pd.DataFrame(dict_diag)
        # create 3D plot from dataframe
        if USE_PS:
            import polyscope as ps
            for ctype in ['updrs',]: #'diag']:
                points = {}
                pt_colors = {}
                pre_ind = 0
                if ctype == 'updrs':
                    cls_names = ['Normal', 'Slight', 'Mild', 'Moderate']
                    labels = updrs_labels[np.concatenate(valid_updrs, axis=0)].astype(int)
                    valid_inds = valid_updrs
                    _data = data_updrs
                else:
                    cls_names = ['Healthy', 'DLB', 'AD']
                    labels = diag_labels[np.concatenate(valid_diag, axis=0)].astype(int)
                    valid_inds = valid_diag
                    _data = data_diag
                    
                for vi, vind in enumerate(valid_inds):
                    points[cls_names[labels[pre_ind]]] = _data[pre_ind:pre_ind+len(vind)]
                    pt_colors[cls_names[labels[pre_ind]]] = tuple(Colors[labels[pre_ind]])
                    pre_ind += len(vind)
                ps.init()
                ps.remove_all_structures()
                for k, v in points.items():
                    ps.register_point_cloud(k, v, color=pt_colors[k], \
                        transparency=0.7, radius=0.008)
                ps.show()
        else:
            # ----- > UPDRS
            fig = go.Figure(
                data=go.Scatter3d(x=df0['component 1'], y=df0['component 2'], z=df0['component 3'], 
                                hovertext=df0['text'], 
                                mode='markers',
                                marker=dict(
                                    sizemode='diameter',
                                    size=8,
                                    color=df0['updrs'],
                                    colorscale=[(0.00, "green"),   (0.25, "green"),
                                                    (0.25, "blue"), (0.5, "blue"),
                                                    (0.5, "red"),  (0.75, "red"),
                                                    (0.75, "cyan"), (1.00, "cyan")],
                                    colorbar=dict(
                                                    title="MDS-UPDRS Gait",
                                                    thicknessmode="pixels", thickness=50,
                                                    tickvals=[0.2, 1.0, 1.8, 2.6],
                                                    ticktext=["Normal", "Slight", "Mild", "Moderate"],
                                                    tickfont=dict(size=25, family='Sans Serif'),
                                                    lenmode='pixels', len=180,
                                                    xref='paper',
                                                    x=0.1,
                                                )),
                                ))
            fig.update_layout(
                scene1=dict(
                    xaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=30,)),
                        showticklabels=False,
                        showgrid=False,),
                    yaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=30,)),
                        showticklabels=False,
                        showgrid=False,),
                    zaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=30,)),
                        showticklabels=False,
                        showgrid=False,),
                ),
            )
            # save the figure as html
            fig.write_html(f'./data/embedding/{file_name}_updrs.html')
            # ----- > DIAG
            fig = go.Figure(
                data=go.Scatter3d(x=df1['component 1'], y=df1['component 2'], z=df1['component 3'], 
                                hovertext=df1['text'], 
                                mode='markers',
                                marker=dict(
                                    sizemode='diameter',
                                    color=df1['diag'],
                                    size=5,
                                    colorscale=[(0.00, "green"),   (0.33, "green"),
                                                    (0.33, "blue"), (0.66, "blue"),
                                                    (0.66, "red"),  (1.00, "red")],
                                    colorbar=dict(
                                                    title="Diagnostic group",
                                                    tickvals=[0.2, 1.0, 1.8],
                                                    ticktext=["Healthy", "DLB", "AD",],
                                                    tickfont=dict(size=14, family='Sans Serif'),
                                                    lenmode='pixels', len=120,
                                                    xref='paper',
                                                    x=0.1,
                                                )),
                                ))
            fig.update_layout(
                scene1=dict(
                    xaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=20,)),
                        showticklabels=False,
                        showgrid=False,),
                    yaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=20,)),
                        showticklabels=False,
                        showgrid=False,),
                    zaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=20,)),
                        showticklabels=False,
                        showgrid=False,),
                ),
            )
            fig.write_html(f'./data/embedding/{file_name}_diag.html')
            
        
if CE_VIS:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    file_path = file_path_format.format('data_dict_part5_4f_orth2')
    assert osp.isfile(file_path)
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    # create dataframe from data_dict
    # ----- > generate valid indices
    updrs_labels = copy.deepcopy(data_dict['updrs'][:,0])
    diag_labels = copy.deepcopy(data_dict['diag'][:,0])
    valid_updrs = [np.where(updrs_labels==i)[0] for i in range(4)]
    valid_diag = [np.where(diag_labels==i)[0] for i in range(3)]
    updrs_labels = updrs_labels.reshape(-1)[np.concatenate(valid_updrs, axis=0)]
    diag_labels = diag_labels.reshape(-1)[np.concatenate(valid_diag, axis=0)].html
    # visualize the projected text efatures in the latent space
    embed_path = {
        'updrs': './data/embedding/updrs_0524_kapt_NTE_fold-5-best.pkl',
        # 'diag':  './data/embedding/diag_BEST_0221_fold-9-best_subset.pkl',
    }
    model_path = {
        'updrs': './logs/updrs_0524-08_Flosskapt_NTE/fold_5/fold-5-best.pth',
        #'diag':  './train_output/hospital_diag/BEST_0221_fold-9-best.pth',
    }
    reducer = umap.UMAP(n_components=3,)
    tf_project = nn.Sequential(
        nn.Linear(embed_dim, embed_dim//4),
        nn.Tanh(),
        nn.Linear(embed_dim//4, embed_dim//8),
    )
    
    for k,v in embed_path.items():
        with open(v, 'rb') as f:
            edata = pickle.load(f)
        final_embed = edata['final_embed']
        # load projection layers
        memo_dict, tf_dict = OrderedDict(), OrderedDict()
        for n,p in torch.load(model_path[k], map_location=device)['model'].items():
            if 'memory_project' in n:
                memo_dict[n.split('memory_project.')[-1]] = p
            elif 'tf_project' in n:
                tf_dict[n.split('tf_project.')[-1]] = p
        # create 3D plot from dataframe
        if k=='updrs':
            _Colors = [
                'rgb(0,255,0)',
                'rgb(0,0,255)',
                'rgb(255,0,0)',
                'rgb(0,255,255)',
                'rgb(255,255,0)',
            ]
            cls_names = ['Normal', 'Slight', 'Mild', 'Moderate'] 
            X_shift = [-50, 50,-100,50]
            valid_inds = valid_updrs
            labels = updrs_labels
            proj_text = ['Projection: Normal', 'Projection: Slight', 'Projection: Mild', 'Projection: Moderate']
            colorscale = [(0.00, "green"),   (0.2, "green"),
                            (0.2, "blue"), (0.4, "blue"),
                            (0.4, "red"),  (0.6, "red"),
                            (0.6, "cyan"), (0.8, "cyan"),
                            (0.8, "yellow"), (1.00, "yellow")]
            colorbar = dict(
                            title="Labels",
                            thicknessmode="pixels", thickness=50,
                            tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
                            ticktext=["Normal", "Slight", "Mild", "Moderate", "Projection"],
                            tickfont=dict(size=25, family='Sans Serif'),
                            lenmode='pixels', len=180,
                            xref='paper',
                            x=0.1,
                        )
            memory_project = nn.ModuleList()
            for _ in range(4):
                memory_project.append(nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//4),
                    nn.Tanh(),
                    nn.Linear(embed_dim//4, embed_dim//8),))
        elif k=='diag':
            _Colors = [
                'rgb(0,255,0)',
                'rgb(0,0,255)',
                'rgb(255,0,0)',
                'rgb(255,255,0)',
            ]
            cls_names = ['Healthy', 'DLB', 'AD']
            X_shift = [100,-50,50]
            valid_inds = valid_diag
            labels = diag_labels
            proj_text = ['Projection: Healthy', 'Projection: DLB', 'Projection: AD',]
            colorscale = [(0.00, "green"),   (0.25, "green"),
                            (0.25, "blue"), (0.5, "blue"),
                            (0.5, "red"),  (0.75, "red"),
                            (0.75, "yellow"), (1.00, "yellow")]
            colorbar = dict(
                            title="Labels",
                            thicknessmode="pixels", thickness=50,
                            tickvals=[0.2, 1.0, 1.8, 2.6,],
                            ticktext=["Healthy", "DLB", "AD", "Projection"],
                            tickfont=dict(size=25, family='Sans Serif'),
                            lenmode='pixels', len=180,
                            xref='paper',
                            x=0.1,
                        )
            memory_project = nn.ModuleList()
            for _ in range(3):
                memory_project.append(nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//4),
                    nn.Tanh(),
                    nn.Linear(embed_dim//4, embed_dim//8),))
        else:
            raise ValueError(f'Unknown key: {k}')

        # project features and create dataframe
        tf_project.load_state_dict(tf_dict, strict=True)
        tf_project.to(device)
        tf_project.eval()
        memory_project.load_state_dict(memo_dict, strict=True)
        memory_project.to(device)
        memory_project.eval()
        with torch.no_grad():
            _final_embed = torch.from_numpy(final_embed).float().to(device)
            _final_embed = tf_project(_final_embed)
            _final_embed = _final_embed / _final_embed.norm(dim=-1, keepdim=True)
            # project support memory
            support_memory = [torch.from_numpy(copy.deepcopy(data_dict['embeds'][i])).float().to(device) for i in valid_inds]
            assert len(support_memory)==len(memory_project)
            proj_memory = []
            proj_features = []
            for im in range(len(support_memory)):
                _memory = memory_project[im](support_memory[im].mean(dim=-2))
                _memory = _memory / _memory.norm(dim=-1, keepdim=True)
                proj_memory.append(_memory)
                # calculate the cosine similarity and get the projections
                sim = _final_embed[im] @ _memory.t()
                proj_embed = sim@_memory
                proj_embed = proj_embed / proj_embed.norm(dim=-1, keepdim=True)
                proj_features.append(proj_embed)
            proj_memory = torch.vstack(proj_memory)
            proj_features = torch.vstack(proj_features)
            len_memo = proj_memory.shape[0]
        
        all_features = torch.cat([proj_memory, proj_features], dim=0).cpu().numpy()
        # use reducer to reduce the dimension
        all_feat = reducer.fit_transform(all_features)
        
        femb_size = final_embed.shape[0]
        df_dict = {
            'component 1': all_feat[:,0],
            'component 2': all_feat[:,1],
            'component 3': all_feat[:,2],
            'label': np.concatenate([labels, np.ones(femb_size)*(femb_size)], axis=0),
            'opacity': np.concatenate([np.ones(len_memo)*0.4, np.ones(femb_size)*1.0], axis=0),
            'size': np.concatenate([np.ones(len_memo)*15, np.ones(femb_size)*30], axis=0),
            'text': np.array(texts)[np.concatenate(valid_inds, axis=0)].tolist()+proj_text,
        }
        df = pd.DataFrame(df_dict)
        
        # visualize the projections together with the learned latent space
        if USE_PS:
            import polyscope as ps
            points = {}
            pt_colors = {}
            pre_ind = 0
            for vi, vind in enumerate(valid_inds):
                points[cls_names[labels[pre_ind]]] = all_feat[pre_ind:pre_ind+len(vind)]
                points['Projection '+cls_names[labels[pre_ind]]] = all_feat[len_memo+vi,:].reshape(1,-1)
                pt_colors[cls_names[labels[pre_ind]]] = tuple(Colors[labels[pre_ind]])
                pt_colors['Projection '+cls_names[labels[pre_ind]]] = (1,1,0)
                pre_ind += len(vind)
            ps.init()
            ps.remove_all_structures()
            for k, v in points.items():
                pt_size = 0.02 if 'Projection' in k else 0.008
                ps.register_point_cloud(k, v, color=pt_colors[k], \
                    transparency=1.0 if 'Projection' in k else 0.25, radius=pt_size)
            
            ps.show()
        else:
            fig = go.Figure(
                data=go.Scatter3d(x=df['component 1'], y=df['component 2'], z=df['component 3'], 
                                hovertext=df['text'], 
                                mode='markers',
                                marker=dict(
                                    sizemode='diameter',
                                    line_color=[_Colors[int(i)] for i in df['label']],
                                    opacity=0.8,
                                    size=df['size'],
                                    color=df['label'],
                                    colorscale=colorscale,
                                    colorbar=colorbar,),
                                ))
            # add text annotations for projected features
            annotations = []
            for i, pt in enumerate(proj_text):
                annotations.append(dict(
                    showarrow=True,
                    x=all_feat[i+len_memo,0], y=all_feat[i+len_memo,1], z=all_feat[i+len_memo,2],
                    text=pt, textangle=0, font=dict(color='black', size=25, family='Sans Serif'),
                    xanchor='right', xshift=0,
                    arrowcolor='black', arrowsize=200, arrowwidth=2, arrowhead=0,
                ))
            fig.update_layout(
                scene=dict(
                    xaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=40,)),
                        showticklabels=False,
                        showgrid=False,),
                    yaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=40,)),
                        showticklabels=False,
                        showgrid=False,),
                    zaxis = dict(
                        title=dict(text='', font=dict(family='Sans Serif',size=40,)),
                        showticklabels=False,
                        showgrid=False,),
                    annotations=annotations,
                ),
            )
            # save the figure as html
            fig.write_html(f"./data/embedding/{osp.basename(embed_path[k]).split('.')[0]}_w_projection.html")