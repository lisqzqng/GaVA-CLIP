# prepare csv file containing list of videos and the mapped action class
# for training, validation and evaluation (test)
import sys
import os
sys.path.insert(0, os.getcwd())
import os.path as osp
import joblib
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

from collections import defaultdict

BASE_DIR = 'datasets/kinetics-dataset/k400_resized' # i.e. `data_root` in VideoDataset
ANNOS_DIR = 'datasets/kinetics-dataset/k400/annotations'
OUT_DIR = 'datasets/kinetics-dataset/k400_resized'

def k400_to_csv(viddir='test', mapfile='data/k400_class_mappings.json',):
    csv_dict = {} # pair class label for each video path
    _viddir = osp.join(BASE_DIR, viddir)
    assert osp.isdir(_viddir), f'Video directory {_viddir} does not exist'
    vid_list = [x for x in os.listdir(_viddir) if x.endswith('.mp4')]
    # load annotations
    anno_file = osp.join(ANNOS_DIR, f'{viddir}.csv')
    data = pd.read_csv(anno_file)
    annos = pd.DataFrame(data, columns=['youtube_id', 'label']).to_numpy()
    # convert annos to dict marked by video names
    # load mapping
    with open(mapfile, 'r') as f:
        action_map = json.load(f)
    action_map = {k:v for v,k in enumerate(action_map)}
    for vid in tqdm(vid_list):
        vidname = vid.split('_')[0]
        index = np.where(annos[:,0]==vidname)[0]
        if index.size ==0: continue
        csv_dict[vid] = action_map[annos[index][0,1]]
    # handle output
    os.makedirs(OUT_DIR, exist_ok=True)
    outfile = osp.join(OUT_DIR, f'{viddir}_for_model.csv')
    if osp.isfile(outfile):
        os.remove(outfile)
    with open(outfile, 'w') as f:
        for k, v in csv_dict.items():
            f.write(f'{k},{v}\n')
    return

def gait_to_csv(diag_file='data/robertsau/robertsau_annotations.json', # without TOAW
                score_file='data/robertsau/robertsau_annotation_UPDRS_2024.xlsx', # includes TOAW
                output_format='./data/{:s}.csv'):
    # ========= load annotations ========= #
    # load xlsx into dictionary
    annos_score = pd.read_excel(score_file, sheet_name='dataset_synthesis',)
    df = pd.DataFrame(annos_score, columns=['vid_name', 'QUESTION']).to_numpy()
    # load json for diagnosis
    annos_diag = joblib.load(diag_file)
    # arrange list of dictionary to dictionary anchored with video name
    diag_dict = {}
    print('preprocessing diagnosis annotations...')
    for anno in tqdm(annos_diag):
        diag_dict[anno['vid_name']] = anno['Diag']
    # save results to a dictionary
    csv_dict = defaultdict(list)
    diag_mapping = {
        'TEMOIN': 0,
        'MCL LEGERE': 1,
        'MA LEGERE': 2,
        'DEMENCE MCL': 3,
        'DEMENCE MA': 4,
    }
    for idx in tqdm(range(len(df))):
        vidname = df[idx,0]
        score = df[idx,1]
        try:
            csv_dict['diag'].append(diag_mapping[diag_dict[vidname]])
        except KeyError:
            assert 'OAW' in vidname, f'Video {vidname} does not have diagnosis'
            csv_dict['diag'].append(diag_mapping['TEMOIN'])
        csv_dict['score'].append(score)
        csv_dict['vidname'].append(vidname)
    
    # write dictionary to csv
    with open(output_format.format('label'), 'w') as f:
        f.write('vidname,score,diag\n')
        for i in range(len(csv_dict['vidname'])):
            f.write(f"{csv_dict['vidname'][i]},{csv_dict['score'][i]},{csv_dict['diag'][i]}\n")
            
    return

def parkinson_to_csv(excel_path='./datasets/parkinson_label_orig.xlsx', vid_dir='./datasets/parkinson/'):
    "produce label csv file for parkinson dataset"
    # load excel file
    df = pd.read_excel(excel_path, sheet_name='PDGinfo')
    # get the label using vid_name
    ON_col = 'ON-UPDRS-III-walking'
    OFF_col = 'OFF-UPDRS-III-walking'
    df = pd.DataFrame(df, columns=['ID', ON_col, OFF_col]).to_numpy()
    label_dict = {}
    for ele in df:
        label_dict[ele[0]] = [ele[1], ele[2]] # on, off
    # create split json
    vid_names = [x for x in os.listdir(vid_dir) if x.endswith('.mp4')]
    csv_dict = {}
    for vn in tqdm(vid_names):
        if vn.split('_')[0] not in label_dict.keys():
            print(f'Video {vn} does not have label')
            continue
        else:
            if vn.split('_')[1] == 'on':
                csv_dict[vn.split('.')[0]] = label_dict[vn.split('_')[0]][0]
            elif vn.split('_')[1] == 'off':
                csv_dict[vn.split('.')[0]] = label_dict[vn.split('_')[0]][1]
            else:
                print(f'Video {vn} does not have label')
                continue
    # write dictionary to csv
    with open('./datasets/parkinson_label.csv', 'w') as f:
        f.write('vidname,score\n')
        for k, v in csv_dict.items():
            f.write(f"{k},{v}\n")
    return

if __name__=='__main__':
    # k400_to_csv()
    # gait_to_csv()
    parkinson_to_csv()
