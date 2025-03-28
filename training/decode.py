# decode the learned features (text side) using the trained DeCap model
import os
import os.path as osp
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from torch import nn

import argparse
from tqdm import tqdm
import pickle
import time
import copy
import shutil

from collections import OrderedDict, defaultdict

from training.VitaCLIP_text_encoder import _Tokenizer, CLIPTextEncoder, tokenize
from training.decoder_train import DeCap, MLP

def generate_npy_from_dict(dict_path, save_dir='./decap/image_features/'):
    exp_name = osp.basename(dict_path).split('.')[0].split('train_')[-1]
    if osp.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=False)
    save_format = osp.join(save_dir, '{:s}_{:s}.npy')
    
    with open(dict_path, 'rb') as f:
        tokens = pickle.load(f)
    
    if isinstance(tokens, dict):
        for key in tqdm(tokens.keys()):
            try:
                if key.split('_')[0] == 'image':
                    for label, vidname in tokens['vidname'].items():
                        array_all = tokens[key][label]
                        for idx, array in enumerate(array_all):
                            np.save(save_format.format(exp_name+f' {label}', vidname[idx]), array)
                elif 'embed' in key:
                    array = tokens[key]
                    np.save(save_format.format(exp_name, key), array)
            except:
                continue # for other key names
    elif isinstance(tokens, list):
        array = np.array(tokens)
        np.save(save_format.format(exp_name, 'embed'), array)
            
    return

def test(model, tokenizer, vocab_size=49408):
    "decode the text with different values"
    from utils.metadata import PE
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    text_format = 'the person walks with X steps per minute .'
    
    clip_encoder = CLIPTextEncoder(
                embed_dim=512,
                context_length=77,
                vocab_size=49408,
                transformer_width=512,
                transformer_heads=8,
                transformer_layers=12,
            )
    ckpt = torch.load('pretrained/clip_pretrained.pth')
    new_ckpt = OrderedDict()
    for n, param in ckpt.items():
        if 'textual' in n:
            new_ckpt[n.replace('textual.','')] = param
    clip_tokenizer.load_state_dict(new_ckpt, strict=True)
    clip_tokenizer = clip_tokenizer.to(device)
    clip_tokenizer.eval()
    for percent in range(30,130,14):
        text = text_format.replace('X', str(round(percent)))
        tokenized_text = tokenize(' '.join(text_format.split())).reshape(1,77).to(device)
        prefix_embedding = clip_encoder.token_embedding(tokenized_text)
        text_embedding = clip_encoder(prefix_embedding, tokenized_text)
        text_embedding += PE[round(percent), :]
            
        generated_text = Decoding(model, text_embedding, clip_tokenizer, vocab_size=vocab_size)

        print('Percent: {:d}, Generated text: {:s}'.format(percent, generated_text))
        
    return

def Decoding(model, text_embedding, tokenizer, embed_dict=None, vocab_size=49408,):
    "process one sentence at once."
    model.eval()
    temperature = 1/150. # temp=1/150 for video captionning
    # project the embedding dim 512-->768
    text_embedding_cat = model.clip_project(text_embedding).reshape(1,1,-1) # 1x1x768
    tokens = None
    input_length = 77 # customized w.r.t the sentence length in the training data
    num_ids = []
    numbers = []
    prefix = -1
    for _ in range(input_length):
        outputs = model.decoder(inputs_embeds=text_embedding_cat)
        logits = outputs.logits
        logits = logits[:, -1, :] / temperature
        
        logits = nn.functional.softmax(logits, dim=-1)
        # replace the number-word tokens with 'x' (token_id = 343)
        next_token = torch.argmax(logits, dim=-1).detach()
        # already inside `with torch.no_grad():`
        if next_token >=vocab_size:
            # get numbers from number_id
            num = next_token.item() - vocab_size
            numbers.append(num)
            next_token[:] = 286 # token_id for '?'
            
        # get the embedding for the next word
        next_token_embed = model.decoder.transformer.wte(next_token.unsqueeze(0))

        if tokens is None:
            tokens = next_token.unsqueeze(0)
        else:
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        
        if next_token.item() == vocab_size-1: # EOS
            break
        text_embedding_cat = torch.cat([text_embedding_cat, next_token_embed], dim=1)
        prefix += 1

    ids = torch.where(tokens==49406)[1].cpu().numpy()
    if len(ids) > 1:
        for i in range(len(ids)-1):
            tokens[0, ids[i]] = 267
    output_list = list(tokens.squeeze().cpu().numpy())
    output = tokenizer.decode(output_list)
    output = output.replace('<|startoftext|>', '')
    # find the position s of the '?'s
    output = output.split()
    num_ids = [i for i, x in enumerate(output) if x == '?']
    # preprocess the embed_dict & get the graduated scale
    use_pe = True
    if embed_dict is not None:
        graduated = embed_dict['extra_info']['graduated']
        if embed_dict['extra_info']['l2_norm'] == 'n/a':
            global_shift = embed_dict['extra_info']['global_shift']
            use_pe = False
        new_dict = {}
        for kk, vv in embed_dict.items():
            if 'extra_info' not in kk:
                new_dict[' '.join(kk.split()[:-1])] = vv
        key_names = [' '.join(x.split()[:-1]) for x in embed_dict.keys() if 'extra_info' not in x]
        str_length = max([len(x.replace(' ','')) for x in key_names])
        key_list = [x.replace(' ', '')+'_'*(str_length-len(x.replace(' ', ''))) for x in key_names]
        key_array = np.vstack([bytearray(x.encode())[:str_length] for x in key_list])
        del embed_dict
    else: new_dict = None
    prev_id = 0
    for nid, n in zip(num_ids, numbers):
        if new_dict is not None:
            # find the key with the smallest distance
            paraname = copy.deepcopy(output[prev_id:nid])
            # padding the paraname
            if len(''.join(paraname)) < str_length:
                paraname = ''.join(paraname)+'_'*(str_length-len(''.join(paraname)))
            else:
                paraname = ''.join(paraname)[:str_length]
            eqs = np.array(bytearray(paraname.encode())[:str_length]).reshape(1,-1) == key_array
            kid = np.argmax(np.sum(eqs, axis=1))
            mean = new_dict[key_names[kid]]['mean']
            std = new_dict[key_names[kid]]['std']
            shift = new_dict[key_names[kid]]['shift']
            weight = new_dict[key_names[kid]]['weight']
            if use_pe:
                rn = round((float(n)*graduated/weight-shift)*std + mean, 3)
            else:
                rn = round(((float(n)-global_shift)*graduated/weight-shift)*std + mean, 3)
        else:
            rn = n
        output[nid] = str(rn)
        prev_id = nid + 1
        
    output = ' '.join(output)
    output = output.replace('<|startoftext|>','').replace('<|endoftext|>','')
    
    return output

def main(args):
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    model = DeCap(cfg_path=args.config_path,)
    
    assert osp.isfile(args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')), strict=False)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = _Tokenizer()

    # -----------> load / generate support memory
    PER_CLS = False
    if osp.isfile(args.memory_tokens):
        if args.memory_tokens.endswith('.json'): # generate support memory from natural language
            support_features_path = args.memory_tokens.replace('.json', '.npy')
            print('Constructing support memory...')
            
            with open(args.memory_tokens, 'rb') as f:
                tokens = pickle.load(f)
            
            tokens = torch.from_numpy(tokens).long()
            
            clip_tokenizer = CLIPTextEncoder(
                    embed_dim=512,
                    context_length=77,
                    vocab_size=49408,
                    transformer_width=512,
                    transformer_heads=8,
                    transformer_layers=12,
                )
            ckpt = torch.load('pretrained/clip_pretrained.pth')
            new_ckpt = OrderedDict()
            for n, param in ckpt.items():
                if 'textual' in n:
                    new_ckpt[n.replace('textual.','')] = param
            clip_tokenizer.load_state_dict(new_ckpt, strict=True)
            clip_tokenizer = clip_tokenizer.to(device)
            clip_tokenizer.eval()
            
            support_features = []
            for i in tqdm(range(0, len(tokens)//args.batch_size+1)):
                batch_tokens = tokens[i*args.batch_size:(i+1)*args.batch_size]
                batch_tokens = batch_tokens.to(device)
                
                with torch.no_grad():
                    prefix_embedding = clip_tokenizer.token_embedding(batch_tokens)
                    text_embedding = clip_tokenizer(prefix_embedding, batch_tokens) # Nx512

                support_features.append(text_embedding.float())
            
            support_features = torch.cat(support_features, dim=0)
            support_features /= support_features.norm(dim=-1, keepdim=True).float()
            
            print('Saving support memory with {} features...'.format(support_features.shape[0]))
            
            np.save(support_features_path, support_features.cpu().numpy())
            
        elif args.memory_tokens.endswith('.npy'):
            print(f'Loading support memory from {args.memory_tokens}...')
            support_features = np.load(args.memory_tokens, allow_pickle=True)
            if isinstance(support_features, list):
                support_features = np.vstack(support_features)
            support_features = torch.from_numpy(support_features).to(device)
            
        elif args.memory_tokens.endswith('.pkl'):
            # dictionary
            print(f'Loading support memory from {args.memory_tokens}...')
            with open(args.memory_tokens, 'rb') as f:
                data = pickle.load(f)
            assert isinstance(data, dict)
            if 'embeds' in data.keys():
                support_features, diag_suppt_feat = defaultdict(list), defaultdict(list)
                for (embd, lupdrs, ldiag) in zip(data['embeds'], data['updrs'].flatten().tolist(), data['diag'].flatten().tolist()):
                    support_features['updrs ' + str(lupdrs)].append(embd)
                    diag_suppt_feat['diag ' + str(ldiag)].append(embd)
                # concatenate the embeddings
                for k, v in support_features.items():
                    support_features[k] = torch.from_numpy(np.vstack(v)).float().to(device)
                try: 
                    del support_features['updrs -1']
                except: pass
                for k, v in diag_suppt_feat.items():
                    diag_suppt_feat[k] = torch.from_numpy(np.vstack(v)).float().to(device)
                support_features.update(diag_suppt_feat)
                del data, diag_suppt_feat
            else:
                print("Use per-class support memory")
                PER_CLS = True
                support_features = data
                for k, v in support_features.items():
                    support_features[k] = torch.from_numpy(v).float().to(device)
            
        else:
            raise ValueError('Unsupported support memory file format.')
    
    # <---------- finish constructing support memory
    # -----------> project the support memory
    # if args.use_mlp_projection:
        # retrieve the projection layers from the VLM checkpoint
    assert osp.isfile(args.mlp_checkpoint_path)
    ckpt_dict = torch.load(args.mlp_checkpoint_path, map_location=torch.device('cpu'))['model']
    embed_dim = 512
    # instantiate the projection layers
    memory_project = nn.Sequential(
        nn.Linear(embed_dim, embed_dim//4),
        nn.Tanh(),
        nn.Linear(embed_dim//4, embed_dim//8),
    )
    if args.use_centroid:
        tf_project = None
    else:
        tf_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.Tanh(),
            nn.Linear(embed_dim//4, embed_dim//8),
        )
        tf_dict = OrderedDict()
    # preprocess the ckpt_dict
    memo_dict = OrderedDict()
    for n, p in ckpt_dict.items():
        if 'memory_project' in n:
            memo_dict[n.split('memory_project.')[-1]] = p
        elif not args.use_centroid and 'tf_project' in n:
            tf_dict[n.split('tf_project.')[-1]] = p
    # load the projection layers
    # load the projection layers
    is_updrs = 'updrs' in args.mlp_checkpoint_path    
    try:
        memory_project.load_state_dict(memo_dict, strict=True)
    except RuntimeError:
        memory_project = nn.ModuleList()
        if not is_updrs:
            for _ in range(5):
                memory_project.append(nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//4),
                    nn.Tanh(),
                    nn.Linear(embed_dim//4, embed_dim//8),))
        else:
            try:
                for _ in range(4):
                    memory_project.append(nn.Sequential(
                        nn.Linear(embed_dim, embed_dim//4),
                        nn.Tanh(),
                        nn.Linear(embed_dim//4, embed_dim//8),))
                memory_project.load_state_dict(memo_dict, strict=True)
            except RuntimeError:
                memory_project = nn.ModuleList()
                for _ in range(3):
                    memory_project.append(nn.Sequential(
                        nn.Linear(embed_dim, embed_dim//4),
                        nn.Tanh(),
                        nn.Linear(embed_dim//4, embed_dim//8),))
                memory_project.load_state_dict(memo_dict, strict=True)
    if not args.use_centroid:
        tf_project.load_state_dict(tf_dict, strict=True)
        tf_project.eval()
        tf_project.to(device)
        with torch.no_grad():
            text_features = torch.load(args.mlp_checkpoint_path, map_location=torch.device('cpu'))['text_features']
            text_features = text_features.to(device)
            tf_proj = tf_project(text_features)
            tf_proj = tf_proj/tf_proj.norm(dim=-1, keepdim=True).float()
        del tf_dict
    
    del ckpt_dict, memo_dict
    # project the support memory
    memory_project = memory_project.to(device)
    sim_support_features = dict.fromkeys(support_features.keys())
    with torch.no_grad():
        for k, v in support_features.items():
            if is_updrs and 'diag' in k: continue
            elif not is_updrs and 'updrs' in k: continue
            cls_id = int(k.split(' ')[-1])
            if cls_id == -1: continue
            assert v.dim() in [2,3]
            try:
                if v.dim()==3:
                    sim_support_features[k] = memory_project[cls_id](v.mean(dim=-2))
                    support_features[k] = v.mean(dim=-2)
                elif v.dim()==2:
                    sim_support_features[k] = memory_project[cls_id](v)
            except:
                if v.dim()==3:
                    sim_support_features[k] = memory_project(v.mean(dim=-2))
                    support_features[k] = v.mean(dim=-2)
                elif v.dim()==2:
                    sim_support_features[k] = memory_project(v)
            sim_support_features[k] /= sim_support_features[k].norm(dim=-1, keepdim=True).float()
            support_features[k] /= support_features[k].norm(dim=-1, keepdim=True).float()
    
    if args.test_only:
        from utils.metadata import PE
        PE = PE.to(device)
        # text_format = 'the duration when the right foot is off the ground takes X percent of the complete cycle of the right foot movement .'
        text_format = 'the person walks at a speed of X cm per second .'
        clip_tokenizer = CLIPTextEncoder(
                    embed_dim=512,
                    context_length=77,
                    vocab_size=49408,
                    transformer_width=512,
                    transformer_heads=8,
                    transformer_layers=12,
                )
        ckpt = torch.load('pretrained/clip_pretrained.pth')
        new_ckpt = OrderedDict()
        for n, param in ckpt.items():
            if 'textual' in n:
                new_ckpt[n.replace('textual.','')] = param
        clip_tokenizer.load_state_dict(new_ckpt, strict=True)
        clip_tokenizer = clip_tokenizer.to(device)
        clip_tokenizer.eval()
        value_list = [*range(30,140,7)]
        for value in value_list:
            print('*'*100+'\n')
            text = text_format.replace('X', str(round(value)))
            # directly encode using CLIPTextEncoder
            tokenized_text = tokenize(' '.join(text_format.split())).reshape(1,77).to(device)
            prefix_embedding = clip_tokenizer.token_embedding(tokenized_text)
            text_embedding = clip_tokenizer(prefix_embedding, tokenized_text)
            text_embedding += PE[round(value), :]
                
            decoded_text = Decoding(model, text_embedding, tokenizer, vocab_size=args.vocab_size)

            print('Original: {:s}, \nDirectly decoded text: {:s}'.format(text, decoded_text))
            
            # decode using support memory
            #### Caculate Coefficients for Linear Combination ####
            if tf_project is not None:
                text_embedding = tf_project(text_embedding)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True).float()
            sim = text_embedding @ sim_support_features.T.float()
            sim = (sim*100).softmax(dim=-1)
            _text_embedding = sim@support_features.float()
            _text_embedding /= _text_embedding.norm(dim=-1,keepdim=True)
            
            generated_text = Decoding(model, _text_embedding, tokenizer, vocab_size=args.vocab_size)
            
            print('Original: {:s}, \nGenerated text: {:s}'.format(text, generated_text))
            
    else:
        # -----------> decode  

        os.system('clear')
        # also write the results to a txt file
        os.makedirs(args.output_dir, exist_ok=True)
        if args.out_fn is not None:
            output_txt = osp.join(args.output_dir, args.out_fn)
        else:
            output_txt = osp.join(args.output_dir, f"{args.checkpoint_path.split('/')[-2]}{time.strftime('%Y%m%d-%H%M')}_output.txt")

        # load the scaling dictionary
        if osp.isfile(args.scale_dict_path):
            with open(args.scale_dict_path, 'rb') as f:
                scaled_dict = pickle.load(f)
        else:
            scaled_dict = None  

        if args.use_centroid:
            out_str = 'CENTROID\n'
            # ==========> Calculate the centroid of the Per-CLASS support memory <========== #
            for emb_key, emb_val in sim_support_features.items():
                if emb_val is None: continue
                centroid_embed = emb_val.mean(dim=0)
                centroid_embed /= centroid_embed.norm(dim=-1, keepdim=True).float()
                # Use cosine similarity as weigths in the linear combination
                lc_weights = emb_val@centroid_embed.T.float()
                text_embedding = lc_weights@support_features[emb_key].float()
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True).float()
                generated_text = Decoding(model, text_embedding, tokenizer, embed_dict=scaled_dict, vocab_size=args.vocab_size)
                print(generated_text)

                out_str += emb_key +' : '+ generated_text + '\n'
                        
            with open(output_txt,'a') as f:
                f.write(out_str)
        else:
            # -----------> load Per-CLASS text features
            if osp.isdir(args.feature_dir):
                feat_file_list = [x for x in os.listdir(args.feature_dir) if x.endswith('.npy')]
                feat_file_list.sort()
            
                for feat_file in feat_file_list:
                    if 'vid' in feat_file or 'OAW' in feat_file:
                        continue
                    image_features_all = np.load(osp.join(args.feature_dir, feat_file))
                    out_str = f"\n{'*'*10}"+f"{osp.basename(feat_file).split('.')[0]}: \n"
                    print(out_str)
                    for cls_id, image_features in enumerate(image_features_all):
                        image_features = torch.from_numpy(image_features).to(device)
                        with torch.no_grad():
                            text_embedding = image_features / image_features.norm(dim=-1, keepdim=True).float()
                            generated_text = Decoding(model, text_embedding, tokenizer, embed_dict=scaled_dict, vocab_size=args.vocab_size)
                            print(generated_text)
                            out_str += generated_text+'\n'
            else:
                # represent the projected text features with support memory
                out_str = f"\n{'*'*10}"+f"{args.mlp_checkpoint_path.replace('logs/','').replace('./','').replace('/','_').split('.pth')[0]}: \n"
                with torch.no_grad():
                    for cls_id, tf in enumerate(tf_proj):
                        emb_key = f'updrs {cls_id}' if is_updrs else f'diag {cls_id}'
                        emb_val = sim_support_features[emb_key]
                        emb_val = emb_val/emb_val.norm(dim=-1, keepdim=True).float()
                        sim = tf@emb_val.T.float()
                        sim = (sim*100).softmax(dim=-1)
                        text_embedding = sim@support_features[emb_key].float()
                        text  = text_embedding/text_embedding.norm(dim=-1, keepdim=True).float()
                        generated_text = Decoding(model, text, tokenizer, embed_dict=scaled_dict, vocab_size=args.vocab_size)
                        print(generated_text)
                        out_str += generated_text+'\n'
                
                        
            with open(output_txt,'a') as f:
                f.write(out_str+'\n')
    
    return
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', type=str, default='./decap/config.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='train_output/decap/20250319-0007/train_best.pt')
    parser.add_argument('--memory_tokens', type=str, default='./data/gait/tulip_dict_basic_4f.pkl')
    parser.add_argument('--feature_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./decap/results/')
    parser.add_argument('--out_fn', type=str, default=None, help='Name of the output file containing the decoded results.')
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_only', action='store_true', help='test the DeCap model with encoded text.')
    parser.add_argument('--use_mlp_projection', action='store_true', 
                        help='load the weights of the MLP from the checkpoint to project the support memory.')
    parser.add_argument('--mlp_checkpoint_path', type=str, default='',
                        help='path to the checkpoint of the Vita-CLIP model containing MLP projection layers.')
    parser.add_argument('--use_centroid', action='store_true', help='use the centroid of the support memory \
                        to decode per-class description.')

    parser.add_argument('--scale_dict_path', type=str, default='./data/gait/tulip_scale_dict_basic_4f.pkl',
                        help='dictionary path to generate npy file')
    
    parser.add_argument('--dict_path', type=str, default='', 
                        help='dictionary path to convert dict of embeddings into per-embedding npy file')
    
    args = parser.parse_args()
    
    if osp.isfile(args.dict_path):
        generate_npy_from_dict(args.dict_path, args.feature_dir,)
        
    main(args)
    
