import sys, os
sys.path.insert(0, os.getcwd())
import os.path as osp

import torch
import torch.nn as nn
import numpy as np

import pickle
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from tqdm import tqdm
import copy
from collections import OrderedDict

import polyscope as ps

Nfold = 10

CLASS_LABEL_DICT = {
    'updrs': ['Normal', 'Slight', 'Mild', 'Moderate',],
    'diag': ['Healthy', 'Early DLB', 'Early AD', 'Severe DLB', 'Severe AD',],
}

Colors = [
    # blue
    (0, 0, 1),
    # magenta
    (1, 0, 1),
    # yellow
    (1, 1, 0),
    # red
    (1, 0, 0),
    # green
    (0, 1, 0),
]

def visualize_original_NTE(nte_file='data/gait/data_dict_basic_4f.pkl',):
    "Visualize the numerical text embeddings (NTE) using polyscope"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert osp.isfile(nte_file)
    with open(nte_file, 'rb') as f:
        data_dict = pickle.load(f)
    # create dataframe from data_dict
    # ----- > generate valid indices
    updrs_labels = copy.deepcopy(data_dict['updrs'][:,0])
    diag_labels = copy.deepcopy(data_dict['diag'][:,0])
    valid_updrs = [np.where(updrs_labels==i)[0] for i in range(4)]
    valid_diag = [np.where(diag_labels==i)[0] for i in range(5)]
    updrs_labels = updrs_labels.reshape(-1)[np.concatenate(valid_updrs, axis=0)]
    diag_labels = diag_labels.reshape(-1)[np.concatenate(valid_diag, axis=0)]

    reducer = umap.UMAP(n_components=3)
    all_embeds = copy.deepcopy(data_dict['embeds'].mean(-2))
    reduced_embeds = reducer.fit_transform(all_embeds)
    # UPDRS
    ps.init()
    ps.remove_all_structures()
    for cind in range(4): 
        valid_inds = valid_updrs[cind]
        ps.register_point_cloud(CLASS_LABEL_DICT['updrs'][cind], reduced_embeds[valid_inds], color=Colors[cind], \
                                transparency=0.7, radius=0.008)
    ps.show()

    # DIAG
    ps.init()
    ps.remove_all_structures()
    for cind in range(5):
        valid_inds = valid_diag[cind]
        ps.register_point_cloud(CLASS_LABEL_DICT['diag'][cind], reduced_embeds[valid_inds], color=Colors[cind], \
                                transparency=0.7, radius=0.008)
    ps.show()

    return

def visualize_projected_NTE(nte_file='data/gait/data_dict_basic_4f.pkl', 
                            updrs_model_dir='logs/updrs_0706_Floss_cntn-split-uni-disc_v1-v2-v3-v4-v5_conventionalNTE', 
                            diag_model_dir='logs/diag_0707-2212_cntn-split-uni-disc_v1-v2-v3-v4-v5_NTE',
                            save_linear_combinations=False,
                            save_base_dir='./decap/image_features/',
                            ):    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert osp.isfile(nte_file)
    with open(nte_file, 'rb') as f:
        data_dict = pickle.load(f)
    # create dataframe from data_dict
    # ----- > generate valid indices
    updrs_labels = copy.deepcopy(data_dict['updrs'][:,0])
    diag_labels = copy.deepcopy(data_dict['diag'][:,0])
    valid_updrs = [np.where(updrs_labels==i)[0] for i in range(4)]
    valid_diag = [np.where(diag_labels==i)[0] for i in range(5)]
    updrs_labels = updrs_labels.reshape(-1)[np.concatenate(valid_updrs, axis=0)]
    diag_labels = diag_labels.reshape(-1)[np.concatenate(valid_diag, axis=0)]
    try:
        texts = data_dict['text']
    except:
        texts = ['N/A',] * updrs_labels.shape[0]
    
    # initialize projection model
    embed_dim = 512

    # build projection MLPs
    tf_project = nn.Sequential(
        nn.Linear(embed_dim, embed_dim//4),
        nn.Tanh(),
        nn.Linear(embed_dim//4, embed_dim//8),
    )
    
    updrs_memory_project = nn.ModuleList([])
    for _ in range(4):
        updrs_memory_project.append(
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim//4),
                nn.Tanh(),
                nn.Linear(embed_dim//4, embed_dim//8),
            )
        )
    
    diag_memory_project = nn.ModuleList([])
    for _ in range(5):
        diag_memory_project.append(
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim//4),
                nn.Tanh(),
                nn.Linear(embed_dim//4, embed_dim//8),
            )
        )
    

    # visualize the projected text features in the latent space
    reducer = umap.UMAP(n_components=3,)
    # load the final embeddings
    nte_embed = copy.deepcopy(data_dict['embeds'].mean(-2))
    # process models of updrs and diag in each fold
    for ctype in ['diag', 'updrs']:
        if ctype=='updrs':
            labels = updrs_labels
            valid_inds = valid_updrs
            memory_project = updrs_memory_project
            cls_names = CLASS_LABEL_DICT['updrs']
            model_dir = updrs_model_dir
            proj_text = [ 'Projection '+cls for cls in cls_names]
        elif ctype=='diag':
            labels = diag_labels
            valid_inds = valid_diag
            memory_project = diag_memory_project
            cls_names = CLASS_LABEL_DICT['diag']
            model_dir = diag_model_dir
            proj_text = [ 'Projection '+cls for cls in cls_names]
        else:
            raise ValueError(f'Invalid class type: {ctype}!')

        for nf in range(Nfold):
            # load the models
            model_path = osp.join(model_dir, f'fold_{nf}', f'fold-{nf}-best.pth')
            if not osp.isfile(model_path):
                continue
            # checkpoints
            model_state_dict = torch.load(model_path, map_location=device)['model']
            # ==========> UPDRS <========== #
            tf_dict = OrderedDict((k.replace('module.tf_project.', ''), v) for k, v in model_state_dict.items() if 'tf_project' in k)
            memo_dict = OrderedDict((k.replace('module.memory_project.', ''), v) for k, v in model_state_dict.items() if 'memory_project' in k)
            # project features and create dataframe
            tf_project.load_state_dict(tf_dict, strict=True)
            tf_project.to(device)
            tf_project.eval()
            memory_project.load_state_dict(memo_dict, strict=True)
            memory_project.to(device)
            memory_project.eval()
            # project all features \
            # # and calculate the cosine similarity for weights of linear combination
            with torch.no_grad():
                cls_text_features = torch.load(model_path, map_location=device)['text_features']
                assert len(cls_text_features)==len(valid_inds)
                proj_text_features = tf_project(cls_text_features)
                proj_text_features = proj_text_features / proj_text_features.norm(dim=-1, keepdim=True)
                # project support memory
                support_memory = [torch.from_numpy(nte_embed)[inds].float().to(device) for inds in valid_inds]
                assert len(support_memory)==len(memory_project) # assert the class numbers
                proj_memory = []
                final_proj_tf = []
                interprete_proj = []
                for im in range(len(support_memory)):
                    _memory = memory_project[im](support_memory[im])
                    _memory = _memory / _memory.norm(dim=-1, keepdim=True)
                    proj_memory.append(_memory)
                    sim = proj_text_features[im] @ _memory.T.float()
                    sim = 100*sim.softmax(dim=-1)
                    final_proj_tf.append(sim @ _memory)
                    interprete_proj.append(sim @ support_memory[im])
                proj_memory = torch.vstack(proj_memory)
                # interpret the text features as linear combination of memory features
                # compute cosine similarity
                # sim = proj_text_features @ proj_memory.T.float()
                # sim = sim.softmax(dim=-1)
                final_proj_text_features = torch.vstack(final_proj_tf)
                interprete_proj = torch.vstack(interprete_proj)
                # linear_comb_tfeats = sim @ torch.concatenate(support_memory, dim=0)
                # linear_comb_tfeats = torch.vstack(linear_comb_tfeats)
                len_memo = proj_memory.shape[0]
            
            if save_linear_combinations: 
                print(f'Saving linear combinations of {ctype} for fold {nf}...')
                os.makedirs(osp.join(save_base_dir, f'SegKPT_NTE_{ctype}'), exist_ok=True)
                save_fp = osp.join(save_base_dir, f'SegKPT_NTE_{ctype}', f'{ctype}_fold{nf}.pkl')
                np.save(save_fp, interprete_proj.cpu().numpy())
            else:
                all_features = torch.cat([proj_memory, final_proj_text_features], dim=0).cpu().numpy()
                # use reducer to reduce the dimension
                all_reduced_features = reducer.fit_transform(all_features)
                # tfEmb_size = linear_comb_tfeats.shape[0]
                # df_dict = {
                #     'component 1': all_reduced_features[:,0],
                #     'component 2': all_reduced_features[:,1],
                #     'component 3': all_reduced_features[:,2],
                #     'label': np.concatenate([labels, np.ones(tfEmb_size)*(tfEmb_size)], axis=0),
                #     'opacity': np.concatenate([np.ones(len_memo)*0.4, np.ones(tfEmb_size)*1.0], axis=0),
                #     'size': np.concatenate([np.ones(len_memo)*15, np.ones(tfEmb_size)*30], axis=0),
                #     'text': np.array(texts)[np.concatenate(valid_inds, axis=0)].tolist()+proj_text,
                # }
              # df = pd.DataFrame(df_dict)
                # visualize the projections together with the learned latent space
                points = {}
                pt_colors = {}
                pre_ind = 0
                for vi, vind in enumerate(valid_inds):
                    points[cls_names[labels[pre_ind]]] = all_reduced_features[pre_ind:pre_ind+len(vind)]
                    points['Projection '+cls_names[labels[pre_ind]]] = all_reduced_features[len_memo+vi,:].reshape(1,-1)
                    pt_colors[cls_names[labels[pre_ind]]] = tuple(Colors[labels[pre_ind]])
                    # grey for projection points
                    pt_colors['Projection '+cls_names[labels[pre_ind]]] = (0.5, 0.5, 0.5)
                    pre_ind += len(vind)

                ps.init()
                ps.remove_all_structures()
                for k, v in points.items():
                    pt_size = 0.02 if 'Projection' in k else 0.008
                    ps.register_point_cloud(k, v, color=pt_colors[k], \
                        transparency=1.0 if 'Projection' in k else 0.25, radius=pt_size)
                
                ps.show()
    
    return

if __name__=='__main__':
    # visualize_original_NTE()
    visualize_projected_NTE(save_linear_combinations=True)