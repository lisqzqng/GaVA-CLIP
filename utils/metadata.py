# d.wang@unistra.fr
# processing related to the Robertsau metadata of gaits

import sys
import os
import os.path as osp
sys.path.insert(0, os.getcwd())
sys.path.insert(0, osp.join(os.getcwd(), 'training/'))
import argparse

import pandas as pd
import pickle

import torch
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, defaultdict

from training.VitaCLIP_text_encoder import tokenize, CLIPTextEncoder

# -----> define the unit conversion dictionary
unit_replace_dict = {
    'cm': 'mm',
    'second': 'ms',
}

# -----> define the adjectif dictionary
adj_dict = {
    'short': 'long',
    'slow': 'fast',
    'minimal': 'maximal',
    'close': 'far',
    'minor': 'major',
}
GRAD_SCALE = 1/99

# compute positional encoding (PE) for numbers
N = 5000
d_model = 512
l2_norm = 2.0

PE = torch.zeros(N, d_model)
position = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

PE[:, 0::2] = torch.sin(position * div_term)
PE[:, 1::2] = torch.cos(position * div_term)

# keep all the numbers at the same L2-Norm
PE = PE/PE.norm(dim=-1, keepdim=True) * l2_norm


def slerp(v0, v1, weight=0.5, dot_thresh=0.9995):
    """
    Does a spherical linear interpolation (slerp) between two vectors.
    return:\n
    \t - the interpolated vector
    """
    convert = False
    device = 'cpu'
    if not isinstance(v0, np.ndarray):
        convert = True
        device = v0.device
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        convert = True
        device = v1.device
        v1 = v1.detach().cpu().numpy()
    
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    
    v0 = v0/np.linalg.norm(v0, keepdims=True)
    v1 = v1/np.linalg.norm(v1, keepdims=True)
    
    dot = np.sum(v0*v1, axis=-1)
    
    assert np.abs(dot).max() < dot_thresh, f'v0 and v1 are not almost colinear: {dot}'   
    
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_thetha_0 = np.sin(theta_0)
    # angle with the defined weight
    theta_w = theta_0 * weight
    sin_thetha_w = np.sin(theta_w)
    # apply the slerp formula
    w0 = np.sin(theta_0 - theta_w) / sin_thetha_0
    w1 = sin_thetha_w / sin_thetha_0
    
    interp = w0[:,None] * v0_copy + w1[:,None] * v1_copy
    
    if convert:
        return torch.from_numpy(interp).to(device)
    else:
        return interp

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    global PE
    PE = PE.to(device)
    # -----> load the metadata
    metadata = pd.read_excel(args.metadata_file, sheet_name='part3')
    
    data_dict = pd.DataFrame(metadata,).to_dict()
    
    all_keys = list(data_dict.keys())
    # -----> split the label names
    cls_label_names = all_keys[:2]
    assert set(cls_label_names) == set(['diag', 'updrs'])
    
    value_names = all_keys[2:]
    lleg_name = 'left leg length'
    rleg_name = 'right leg length'
    value_names.remove(lleg_name)
    value_names.remove(rleg_name)
    
    toNorm_names = [x for x in value_names if 'distance' in x and 'difference' not in x]
    labels = pd.DataFrame(metadata, columns=cls_label_names).to_dict()
    leg_lengths = pd.DataFrame(metadata, columns=[lleg_name, rleg_name]).to_numpy().astype(float) # left, right
    gait_params = pd.DataFrame(metadata, columns=value_names).to_dict()
    num_sample = leg_lengths.shape[0]
    # unique_threshold = num_sample//10 # should have at least `unique_threshold` unique values for each parameter
    
    if not args.keep_length:
        # -----> normalize the distance values by leg lengths
        for name in toNorm_names:
            if 'right' and 'left' in name:
                gait_params[name] = np.array(list(gait_params[name].values())) / leg_lengths.mean(axis=1)
            elif 'left' in name:
                gait_params[name] = np.array(list(gait_params[name].values())) / leg_lengths[:,0]
            elif 'right' in name:
                gait_params[name] = np.array(list(gait_params[name].values())) / leg_lengths[:,1]
            else:
                gait_params[name] = np.array(list(gait_params[name].values())) / leg_lengths.mean(axis=1)
    
    # -----> process the values
    new_dict = {}
    for k, v in gait_params.items():
        if isinstance(v, dict):
            v = np.array(list(v.values()))
        new_dict[k] = v
    #     if v.mean() < 2:
    #         if 'speed' in k:
    #             kname = k.replace('cm per second', 'cm per minute')
    #             nv = (v*60).astype(int)
    #         else:
    #             for n in unit_replace_dict.keys():
    #                 if n in k:
    #                     kname = k.replace(n, unit_replace_dict[n])
    #                     nv = (v*1000).astype(int)
    #                     break
    #     else:
    #         kname = k
    #         nv = v.astype(int)
        
    #     kname = ' '.join(kname.split())
        # new_dict[kname] = nv
            
    #     assert len(set(new_dict[kname])) > unique_threshold, f'{kname} has less than {unique_threshold} unique values !!'
    
    del gait_params

    # -----> create the data dictionary
    output = {
        'embeds': [],
        'updrs': [],
        'diag': [],
        'tokens': [], # with extra 5000 number words: 49408 + 5000 = 54408
    }
    
    # -----> save the values as embedding dictionary
    embed_dict = {} # save the graduated values
     
    # -----> generate sentences and emcode using clip and gpt2 tokenizer
    # initialize the text encoder
    clip_tokenizer = CLIPTextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    # load pretrained weights
    ckpt = torch.load(osp.join(os.getcwd(), 'pretrained/clip_pretrained.pth'))
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer.to(device)
    clip_tokenizer.eval()
    del ckpt, new_ckpt
    
    # define the updated vocab with 0~4999 extra number words
    vocab_size = int(49407 + 100)
    token_start = 49408 #  0-->49408+0=49408
    # visualize the progress
    value_names = list(new_dict.keys())
    progress = tqdm(total=(len(value_names)-1)**2//2+len(value_names), desc="Generating text embeddings",)
    # process the rows in the dataframe
    embed_dict['key_embed'] = {}
    
    tokenized_point = tokenize(['.']).reshape(1,-1)[0,1:3]
    
    for a, k1 in enumerate(value_names):
        # save the graduated values into the embedding dictionary
        values = new_dict[k1]
        grad_values = (values - values.min())/values.ptp()
        grad_values = (grad_values/GRAD_SCALE).astype(int)
        v1 = grad_values
        if k1 in embed_dict['key_embed'].keys():
            pass
        else:
            _values = np.sort(grad_values)
            _values = _values*GRAD_SCALE*values.ptp() + values.min()
            embed_dict[k1] = _values
        
        for b, k2 in enumerate(value_names):
            if b < a:
                continue
            progress.update(1)
            
            values = new_dict[k2]
            grad_values = (values - values.min())/values.ptp()
            grad_values = (grad_values/GRAD_SCALE).astype(int)
            v2 = grad_values
            if k1 in embed_dict['key_embed'].keys():
                pass
            else:
                _values = np.sort(grad_values)
                _values = _values*GRAD_SCALE*values.ptp() + values.min()
                embed_dict[k1] = _values
                    
            sentence = [k1]
            end_word = [k1.split()[-1]]
            end_sentence = [k1.replace(end_word[0], adj_dict[end_word[0]])]
            if k1 == k2:
                pass
            else:
                sentence.append(' and ' + k2)
                end_word.append(k2.split()[-1])
                end_sentence.append(' and ' + k2.replace(end_word[1], adj_dict[end_word[1]]))
                
            # tokenize the sentence
            try:
                assert len(sentence[0].split())+ len(sentence[-1].split()) < 60, f'{sentence} has more than 60 words !!'
            except AssertionError:
                continue
            # keep tracking the maximum value
            # max_value = max(max_value, s1, s2)
            
            tokenized_start = tokenize(sentence).reshape(len(sentence),77)
            tokenized_end = tokenize(end_sentence).reshape(len(sentence),77)
            
            with torch.no_grad():
                # encode the sentence to prefix embedding
                prefix_embedding_start = clip_tokenizer.token_embedding(tokenized_start.to(device))
                prefix_embedding_end = clip_tokenizer.token_embedding(tokenized_end.to(device))
                
                # encode prefix embedding to text embedding
                text_embedding_start = clip_tokenizer(prefix_embedding_start, tokenized_start)
                text_embedding_start /= text_embedding_start.norm(dim=-1, keepdim=True)
                text_embedding_end = clip_tokenizer(prefix_embedding_end, tokenized_end)
                text_embedding_end /= text_embedding_end.norm(dim=-1, keepdim=True)

            if k1 in embed_dict['key_embed'].keys(): 
                pass
            else:
                embed_dict['key_embed'][k1] = text_embedding_start.cpu().numpy()
            if k2 in embed_dict['key_embed'].keys(): 
                pass
            else:
                embed_dict['key_embed'][k2] = text_embedding_start.cpu().numpy()
            # process the rows
            for label_id, (num1, num2) in enumerate(zip(v1, v2)):
                # get actual tokenized text
                if k1==k2:
                    weight = np.array([num1])
                else:
                    weight = np.array([num1, num2])
                text_embedding = slerp(text_embedding_start, text_embedding_end, weight=weight) # dimension
                text_embedding = text_embedding.mean(dim=0) # dimension
                tokenized_text = None
                base_tensor = torch.ones_like(tokenized_start[0:1,0])
                for idx, w in enumerate(weight+token_start):
                    if tokenized_text is None:
                        tokenized_text = torch.cat([tokenized_start[idx, :tokenized_start.argmax(dim=-1)[idx]-1], base_tensor*w], dim=-1)     
                    else:
                        tokenized_text = torch.cat([tokenized_text, tokenized_start[idx, 1:tokenized_start.argmax(dim=-1)[idx]-1], base_tensor*w], dim=-1)
                
                tokenized_text = torch.cat([tokenized_text, tokenized_point], dim=-1)
                # extend the size of the last dimemsion to 77
                tokenized_text = torch.cat([tokenized_text, torch.zeros_like(base_tensor).expand(77-tokenized_text.shape[-1])], dim=-1)
                tokenized_text = tokenized_text.cpu().numpy()
                
                # get the labels
                updrs = labels['updrs'][label_id]
                diag = labels['diag'][label_id]
                output['tokens'].append(tokenized_text.reshape(1, 77).astype(np.int64))
                output['embeds'].append(text_embedding.cpu().numpy().reshape(1, 512))
                output['updrs'].append(int(updrs))
                output['diag'].append(int(diag))
        
    
    for k,v in output.items():
        if isinstance(v[0], np.ndarray):
            output[k] = np.concatenate(v, axis=0)
        else:
            output[k] = np.array(v)
        print(f'{k}: {output[k].shape}')
                
    os.makedirs(args.save_dir, exist_ok=True)
    
    # save the dictionary using pickle
    save_file_path = osp.join(args.save_dir, f"data_dict{'_raw' if args.keep_length else ''}.pkl")
    with open(save_file_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"Saved the data dictionary to {save_file_path}.")
    
    # print(f"Maximum value: {max_value}.")
    
    # also save the embedding dictionary
    save_embdict_path = osp.join(args.save_dir, f"embed_dict{'_raw' if args.keep_length else ''}.pkl")
    with open(save_embdict_path, 'wb') as f:
        pickle.dump(embed_dict, f)
    
    print(f"Saved the embedding dictionary to {save_embdict_path}.")

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--metadata_file', type=str, default='./decap/metadata.xlsx', 
                        help='path to the Excel file containing the gait parameters.')
    parser.add_argument('--save_dir', type=str, default='./data/gait/',
                        help='path to save the processed data.')
    parser.add_argument('--keep_length', help='Will not normalize the walking distance / speed with the leg length.', 
                        action='store_true')
    parser.add_argument('--no_dict_from_data', action='store_true', \
        help='Disable the generation of data dictionary for `UPDRS` and `dementia subtype` from the overall dictionary.')
    
    args = parser.parse_args()
    
    main(args)
    
    if not args.no_dict_from_data: 

        data_file_path = osp.join(args.save_dir, f"data_dict{'_raw' if args.keep_length else ''}.pkl")
        with open(data_file_path, 'rb') as f:
            output = pickle.load(f)

        # convert the data into dictionary
        updrs_dict = defaultdict(list)
        diag_dict = defaultdict(list)
        import copy
        data = copy.deepcopy(output)

        for idx, v in tqdm(enumerate(output['embeds'])):
            if output['updrs'][idx] >= 0:
                updrs_dict[output['updrs'][idx]].append(v.reshape(1, -1))
            diag_dict[output['diag'][idx]].append(v.reshape(1, -1))

        for k, v in updrs_dict.items():
            updrs_dict[k] = np.concatenate(v, axis=0)

        for k, v in diag_dict.items():
            diag_dict[k] = np.concatenate(v, axis=0)

        # save the data
        with open(data_file_path.replace('data_', 'updrs_'), 'wb') as f:
            pickle.dump(updrs_dict, f)
            
        with open(data_file_path.replace('data_', 'diag_'), 'wb') as f:
            pickle.dump(diag_dict, f)
    