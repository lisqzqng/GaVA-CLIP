# metadata preprocessing with outlier selection

import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './training')
import os.path as osp

import pandas as pd
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import torch
torch.manual_seed(0) 
import numpy as np

from scipy.optimize import minimize
from sklearn.cluster import KMeans
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from training.VitaCLIP_text_encoder import CLIPTextEncoder, tokenize

N = 200 # 0 ~ 199
SUBSET_LEN = 4
MAX_TEXT_LEN = 77
VOCAB_SIZE = 49408
MAX_COMB = 5000
SEPERATE = True # whether to keep the embedding within one sentence seperated, not making average
FILTER = False # whether to filter out the correleated gait parameters

def data_preprocess(metadata_file:str, save_dir='./data/gait/', video_dir='datasets/miccai_10_fold0fold', l2_norm=None, 
                    no_pe=False, new_pe=False, ke_path=''):
    """
    Search for the correct value to scale the PE values &
    the correct radius to select outliners
    so that\n
    \t the maximum inter-class embedding distance (i.e. both normal) is lower /cosine similarity
    is lower/larger that the minimum inter-class embedding distance/cosine similarity.
    returns\n
    a dictionary containing the the filtered metadata (per rows) 
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ==========> construct the position embedding <========== #
    global N
    d_model = 512
    LOAD_KE = False
    if osp.isfile(ke_path):
        ke = np.load(ke_path)
        LOAD_KE = True
        # d_model = ke.shape[-1]

    PE = torch.zeros(1000, d_model)
    position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)

    # keep all the numbers at the same L2-Norm
    PE = PE/PE.norm(dim=-1, keepdim=True)
    

    # -----> initialize the text encoder, always needed for tokenization <----- #
    clip_tokenizer = CLIPTextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    # load pretrained weights
    ckpt = torch.load(osp.join(os.getcwd(), './pretrained/clip_pretrained.pth'))
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer.to(device)
    clip_tokenizer.eval()
    del ckpt, new_ckpt

    # load the metadata
    metadata = pd.read_excel(metadata_file, sheet_name='part1')
    data_dict = pd.DataFrame(metadata,).to_dict()
    unit_info = pd.read_excel(metadata_file, sheet_name='unit')
    if LOAD_KE:
        id_ke = pd.read_excel(metadata_file, sheet_name='ID')
    unit_dict = pd.DataFrame(unit_info).to_dict()
    unit_dict = {k: v[0] for (k,v) in unit_dict.items() if v[0] is not np.nan}
    unit_dict.update({k: '' for (k,v) in unit_dict.items() if type(v) is float})
    
    other_names = ['vidname', 'updrs', 'diag', 'leglength']
    video_names = data_dict['vidname']
    value_names = set(list(data_dict.keys())).difference(other_names)
    value_names = list(value_names)
    # assert len(value_names) == 29, 'The number of gait parameters is not 29'
    diag_annos = pd.DataFrame(metadata, columns=['diag']).to_numpy().astype(int)
    normal_idx = np.where(diag_annos==0)[0]
    if normal_idx.size == 0:
        normal_idx = np.where(pd.DataFrame(metadata, columns=['updrs']).to_numpy().astype(int)==1)[0]
    # ==========> process the gait parameters <========== #
    new_dict, raw_dict = {}, {}
    info_dict = pd.DataFrame(metadata, columns=other_names[1:3]).to_dict()
    info_dict = {k: np.array(list(v.values())) for k, v in info_dict.items()}
    leg_lengths = pd.DataFrame(metadata, columns=other_names[3:4]).to_numpy().astype(float) # left, right
    # -----> data / embeding preparation
    # save the original mean, std and the final shift of the gait parameters
    scale_dict = defaultdict(dict)
    max_value = 0
    base_embeds = []
    embeds = []
    tokens = []

    for name in value_names:
        assert len(name.split())<70, f'The name {name} is too long'
        with torch.no_grad():
            token = tokenize(name).reshape(1,77).to(device) # 'is' contains in the name
            if LOAD_KE:
                embed = torch.tensor(ke[id_ke[name]]).reshape(1,768).to(device)
            else:
                pre_embed = clip_tokenizer.token_embedding(token)
                embed = clip_tokenizer(pre_embed, token)
        tokens.append(token.cpu())
        base_embeds.append(embed.reshape(1,-1).cpu())
        embeds.append(embed.reshape(1,-1).cpu())
        assert isinstance(data_dict[name], dict)
        new_value = np.array([x for x in data_dict[name].values()])
        # normalize the distance values by leg length if necessary, including the difference of distance
        if ('distance' in name) or ('speed' in name) or ('margin of stability' in name.lower()):
            # if 'right' and 'left' in name:
            #     new_value /= leg_lengths.mean(axis=-1)
            # elif 'right' in name:
            #     new_value /= leg_lengths[:,1]
            # elif 'left' in name:
            #     new_value /= leg_lengths[:,0]
            # else:
            #     # assert 'speed' in name, f'The name {name} is not recognized'
            #     new_value /= leg_lengths.mean(axis=-1)
            new_value /= leg_lengths.mean(axis=-1)
        # normalize the values by the mean and std
        raw_dict[name] = new_value
        # ============>> Calculate the mean value based on the average of the healthy people
        #mean = new_value.mean()
        mean = new_value[normal_idx].mean()
        std = new_value.std()
        new_value = (new_value - mean) / std
        if no_pe:
            # shift = -(new_value.max()+new_value.min())/2 # force the data to be zero-centered
            shift = 0
            # weight = 4.99/(new_value.max()-new_value.min()) # scale the data in to ange [-2.5, 2.5]
            weight = 2.5/np.abs(new_value).max()
        else:
            shift = -new_value.min()
            weight = 1.0
        new_value += shift
        new_value *= weight
        if no_pe:
            assert np.abs(new_value).max() < 5.0
        max_value = max(max_value, new_value.max())
        scale_dict[name].update({'mean': mean, 'std': std, 'shift': shift, "weight": weight,})
        new_dict[name] = new_value
                    
    del data_dict
    tokens = torch.cat(tokens, dim=0).cpu()
    base_embeds = torch.cat(base_embeds, dim=0).cpu() # (28, 512)
    base_embeds /= np.linalg.norm(base_embeds, axis=-1, keepdims=True)
    embeds = torch.cat(embeds, dim=0).cpu() # (28, 512)
    embeds /= np.linalg.norm(embeds, axis=-1, keepdims=True)
    name_sims = embeds @ embeds.T # (28, 28)

    # prepare start, placeholder, end tokens
    if LOAD_KE:
        IS = clip_tokenizer.token_embedding(tokenize('is').to(device))
    else:
        with torch.no_grad():
            ne_tok = tokenize('X is X').to(device) # (1,77)
            pre_tok = clip_tokenizer.token_embedding(ne_tok) # (1,77,512)
        
        pre_tok[:,[1,3],:] = 1.
        
    if no_pe: # [NUM] embedding
        # =====> use fixed [NUM] embedding @https://arxiv.org/abs/2310.02989
        # get the embedding [NUM] orthogonal to the position embedding
        from numpy.linalg import svd
        A = np.vstack((PE[:d_model-1].cpu().numpy(), np.zeros((1, d_model))))
        u, S, Vt = svd(A)
        ss = np.zeros((d_model, d_model))
        ss[-1,-1] = 1
        NE = (u @ ss @ Vt)[-1]
        NE /= np.linalg.norm(NE, axis=-1, keepdims=True)
    elif not new_pe:
        # calculate the suitable L2-Norm for PE
        if l2_norm is None:
            test_sentence = 'the walking speed is'
            text_token = tokenize(test_sentence)
            
            with torch.no_grad():
                prefix_embedding = clip_tokenizer.token_embedding(text_token.to(device))
                text_embedding = clip_tokenizer(prefix_embedding, text_token.to(device))
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            
            pe0 = PE[0].cpu().numpy()
            pe1 = PE[250].cpu().numpy()
            text_embedding = text_embedding.cpu().numpy().reshape(-1)
            function = lambda l: ((text_embedding + pe0*l)/np.linalg.norm(text_embedding + pe0*l, axis=-1, keepdims=True) \
                @ (text_embedding + pe1*l)/np.linalg.norm(text_embedding + pe1*l, axis=-1, keepdims=True) - name_sims.mean().cpu().numpy())**2
            res = minimize(function, x0=1., tol=1e-9) #options={'maxiter':100}, 
            l2_norm = res.x[0]
            
        PE *= l2_norm
    
    # calculate the graduated scale for the gait parameters
    if no_pe:
        graduated = 5/N
        scale_dict['extra_info'] = {'graduated': graduated, 'l2_norm': 'n/a', 'global_shift':N/2,}
    else:
        graduated = max_value / (N-1)
        scale_dict['extra_info'] = {'graduated': graduated, 'l2_norm': l2_norm,}
    
    output = {
        'embeds': [],
        'updrs': [],
        'diag': [],
        'tokens': [],
        'text': [],
    }
    if LOAD_KE:
        output['values'] = []
    
    # -----> combination pairs preparation
    assert SUBSET_LEN == 4
    index = np.arange(len(value_names)).astype(int)
    grid = np.array(np.meshgrid(index, index, index, index)).T.reshape(-1, SUBSET_LEN)
    if FILTER:
        ## compute the Pearson correlation coefficient between each pair of gait parameters
        all_values = np.vstack([raw_dict[value_names[i]] for i in range(len(value_names))])
        pearson = np.corrcoef(all_values)
        threshold = 0.4
    num_rows = list(new_dict.values())[0].shape[0]
    try:
        IS = IS.unsqueeze(0).expand(num_rows, SUBSET_LEN, -1, -1)
    except: pass
    all_comb = []
    token_point = tokenize('.')[0,1:3]
    ## visualize the progress
    progress = tqdm(total=math.comb(len(value_names), SUBSET_LEN), desc="Generating text embeddings",)

    ## initialize the dictionary to save the npy files
    init_npy = lambda: np.empty((0,512))
    npy_dict = defaultdict(init_npy)

    for comb in list(grid):
        if set(comb) in all_comb:
            continue
        elif len(np.unique(comb)) < SUBSET_LEN:
            continue
        else:
            progress.update(1)
        if FILTER:
            # check whether the correlation coefficient is larger than the threshold
            to_continue = False
            for i in range(SUBSET_LEN):
                for j in range(i+1, SUBSET_LEN):
                    if abs(pearson[comb[i], comb[j]])>threshold:
                        to_continue = True
                        break
                if to_continue:
                    break
            if to_continue:
                continue
        all_comb.append(set(comb))
        values = np.vstack([new_dict[value_names[i]] for i in comb])
        
        scaled_values = (values / graduated)
        if no_pe:
            scaled_values += N/2
            
        assert scaled_values.min() >= 0
        scaled_values = scaled_values.astype(int)
        
        # get joint embeddings
        embs = base_embeds[comb]
        embs = embs.unsqueeze(1).expand(-1, num_rows, -1).to(device)
        if no_pe:
            nes = scaled_values[...,None] * NE
            nes = torch.from_numpy(nes).to(device)
        else:
            nes = PE[scaled_values.reshape(-1)].reshape(SUBSET_LEN, num_rows, 512).to(device)
        if no_pe or new_pe:
            # construct new prefix tokens & tokenized text
            if LOAD_KE:
                with torch.no_grad():
                    isValues =nes.permute(1,0,2)
            else:
                curr_pretok = pre_tok.clone().expand(num_rows*SUBSET_LEN, -1, -1)
                curr_tok = ne_tok.clone().expand(num_rows*SUBSET_LEN, -1)
                mulmak = torch.ones_like(curr_pretok)
                mulmak[:,1,:] = embs.clone().reshape(-1, d_model)
                mulmak[:,3,:] = nes.clone().reshape(-1, d_model)
                curr_pretok = curr_pretok.clone()*mulmak
                with torch.no_grad():
                    # embs = clip_tokenizer.transformer(curr_pretok.permute(1,0,2)).permute(1,0,2)
                    # embs = clip_tokenizer.ln_final(embs)
                    # embs = embs[torch.arange(embs.shape[0]), torch.where(ne_tok==49407,)[-1]] @ clip_tokenizer.text_projection
                    embs = clip_tokenizer(curr_pretok, curr_tok)
                embs = embs.reshape(SUBSET_LEN, num_rows, d_model)
                embs /= embs.norm(dim=-1, keepdim=True)
        else:
            embs = embs.clone() + nes
            
            embs /= embs.norm(dim=-1, keepdim=True)

        if not SEPERATE:
            embs = embs.mean(dim=0)
            embs /= embs.norm(dim=-1, keepdim=True)
        else:
            embs = embs.permute(1,0,2).contiguous()
            
        
        # get natural language texts
        base_text = ' _ , '.join([value_names[i] for i in comb])
        base_text += ' _'
        num_pos = np.where(np.array(base_text.split())=='_')[0]
        # if num_pos.max() > 75: continue
        texts = np.repeat(np.expand_dims(np.array(base_text.split()), axis=0), num_rows, axis=0)
        for ip, pos in enumerate(num_pos):
            numbers = np.array(np.round(raw_dict[value_names[comb[ip]]], 3)).astype(str)
            numbers = np.array([ str(n) + ' ' + unit_dict[value_names[comb[ip]]] for n in numbers])
            texts[:,pos] = numbers
        texts = texts.tolist()
        texts = [' '.join(t) for t in texts]
        # get joint tokens
        tks = tokens[comb]
        tok = torch.zeros((num_rows, MAX_TEXT_LEN))
        tok = tok.clone()
        end_ids = torch.argmax(tks, dim=-1)
        prev_ind = 0
        scaled_values = torch.from_numpy(scaled_values).float()
        # add the number after 'is', for visualization
        for j, tk in enumerate(tks):
            start_id = 0 if j==0 else 1
            tok[:, prev_ind:prev_ind+end_ids[j]-start_id] = tk[start_id:end_ids[j]]
            tok[:, prev_ind+end_ids[j]-start_id] = scaled_values[j] + VOCAB_SIZE
            prev_ind += end_ids[j]-start_id+1
        tok[:, prev_ind:prev_ind+2] = token_point.unsqueeze(0) 
        vis = False
        if vis:
            fig = plt.figure()
            updrs_label = info_dict['updrs']
            diag_label = info_dict['diag']
            ax_updrs = fig.add_subplot(121, projection='3d')
            ax_updrs.set_title('updrs')
            ax_diag = fig.add_subplot(122, projection='3d')
            ax_diag.set_title('diag')
            markers = ['o', 'x', '^', 'v', 's']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i in range(scaled_values.shape[1]):
                m_updrs = markers[updrs_label[i]]
                m_diag = markers[diag_label[i]]
                c_updrs = colors[updrs_label[i]]
                c_diag = colors[diag_label[i]]
                if updrs_label[i]>=0:
                    ax_updrs.scatter(scaled_values[0,i], scaled_values[1, i], scaled_values[2, i], marker=m_updrs, c=c_updrs)
                ax_diag.scatter(scaled_values[0, i], scaled_values[1, i], scaled_values[2, i], marker=m_diag, c=c_diag)

            ax_updrs.set_xlabel(value_names[comb[0]])
            ax_updrs.set_ylabel(value_names[comb[0]])
            ax_updrs.set_zlabel(value_names[comb[1]])
            ax_diag.set_xlabel(value_names[comb[2]])
            ax_diag.set_ylabel(value_names[comb[1]])
            ax_diag.set_zlabel(value_names[comb[2]])
            
            plt.show()
            plt.close()
        
        # save to the output dictionary
        embs = embs.cpu().numpy()
        output['embeds'].append(embs)
        ### =====>> prepare per-video embeddings
        for vid, vn in video_names.items():
            npy_dict[vn] = np.vstack([npy_dict[vn], embs[vid].mean(0, keepdims=True)])
            
        if LOAD_KE:
            output['values'].append(isValues.cpu().numpy())
        output['updrs'].append(info_dict['updrs'].reshape(num_rows, 1))
        output['diag'].append(info_dict['diag'].reshape(num_rows, 1))
        output['tokens'].append(tok.reshape(num_rows, MAX_TEXT_LEN).numpy())
        output['text'].extend(texts)
        
    
    if not FILTER:
        assert len(all_comb) == math.comb(len(value_names), SUBSET_LEN)
    else:
        print(f'Number of combinations after filtering: {len(all_comb)}')
    
    # save the embeddings in a per-video manner
    os.makedirs(osp.join(video_dir, 'nte'), exist_ok=True)
    for vid, vn in video_names.items():
        npy_save_path = osp.join(video_dir, 'nte', f'{vn}.npy')
        np.save(npy_save_path, npy_dict[vn])

    # concatenate the data in the output dictionary
    for k, v in output.items():
        if isinstance(v[0], str):
            output[k] = v
            print(f'{k}: {len(output[k])}')
            continue
        if v[0].ndim==1:
            output[k] = np.concatenate(v, axis=0)
        elif v[0].ndim==2:
            output[k] = np.vstack(v,)
        elif v[0].ndim>=3:
            output[k] = np.concatenate(v, axis=0)
        else:
            raise ValueError('The dimension of the output is not supported.')
        print(f'{k}: {output[k].shape}')
        
    # save the output as pkl file
    data_save_path = osp.join(save_dir, f"{osp.basename(metadata_file).split('_')[0].replace('.','')}_dict_basic_{SUBSET_LEN}f{'_ke' if LOAD_KE else''}.pkl")
    with open(data_save_path, 'wb') as f:
        pickle.dump(output, f)
    
    # save the dictionary containing the scale information
    scale_dict['extra_info'].update(unit_dict)
    with open(data_save_path.replace('dict', 'scale_dict'), 'wb') as f:
        pickle.dump(scale_dict, f)

    # split the data into 'updrs' & 'diag' and save them as pkl files
    # updrs_dict = defaultdict(list)
    # diag_dict = defaultdict(list)

    # for idx, v in tqdm(enumerate(output['embeds'])):
    #     if output['updrs'][idx] >= 0:
    #         updrs_dict[output['updrs'][idx].item()].append(np.expand_dims(v, axis=0))
    #     diag_dict[output['diag'][idx].item()].append(np.expand_dims(v, axis=0))

    # for k, v in updrs_dict.items():
    #     updrs_dict[k] = np.concatenate(v, axis=0)

    # for k, v in diag_dict.items():
    #     diag_dict[k] = np.concatenate(v, axis=0)

    # # save the data
    # with open(data_save_path.replace('data_dict', 'updrs_dict'), 'wb') as f:
    #     pickle.dump(updrs_dict, f)
        
    # with open(data_save_path.replace('data_dict', 'diag_dict'), 'wb') as f:
    #     pickle.dump(diag_dict, f)
    return

if __name__=='__main__':
    data_preprocess(metadata_file='./data/tulip_basic_gparams.xlsx',
                    no_pe=True, new_pe=False, video_dir='./datasets/tulip') #ke_path='./data/ke_gait_params.npy') # 0.2ff, 2.0 f