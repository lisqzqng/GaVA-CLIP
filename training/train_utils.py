# split videos into chunks and rewrite csv files
import sys, os
sys.path.insert(0, os.getcwd())
import os.path as osp
import subprocess
import shutil
import time
import re

import numpy as np

import joblib
import json
import csv
import pandas as pd

from tqdm import tqdm

from collections import defaultdict


CLASS_LABEL_DICT = {
    'updrs': ['Normal', 'Slight', 'Mild',], # 'Moderate',],
    'diag': ['Healthy', 'Early DLB', 'Early AD', 'Severe DLB', 'Severe AD',],
}

No_Healthy = True # only make 2-class diagnosis classfiication
MIN_REST = 20
STRIDE = 25
'''split_names_fp =  './datasets/hospital/split_names.json'
# load predefined splited names for each fold
assert osp.isfile(split_names_fp), f"split names file {split_names_fp} not found !!"
split_names = joblib.load(split_names_fp)
# preprocess video names
for fold in split_names:
    for key in split_names[fold]:
        split_names[fold][key] = list(set([x.split('*')[0] for x in split_names[fold][key]]))
        '''

def update_updrs_annotations(full_anno_file='./datasets/hospital/robertsau_annotation_UPDRS_2024.xlsx', csvfile='data/updrs.csv',):
    " update the UPDRS annotations using the file of updated full annotations "
    # load original annotations
    assert osp.isfile(csvfile), f"csv file {csvfile} not found !!"
    score = pd.read_csv(csvfile, header=None).to_numpy()
    score_dict = {}
    # load the column 'new_score' from the csv file 'full annotations'
    data = pd.read_excel(full_anno_file, sheet_name="dataset_synthesis")
    updated_annos = pd.DataFrame(data, columns=["vid_name","new_score"]).to_dict()
    # convert dataframe object into dictionary
    updated_dict = {}
    for k, vn in enumerate(updated_annos['vid_name'].items()):
        updated_dict['f'+vn[1]+'.mp4'] = updated_annos['new_score'][k]
    # update the UPDRS score a.c.d to the full annotations
    for ind, s in enumerate(score):
        score_dict[s[0]] = updated_dict[s[0]]
        if s[1]!=updated_dict[s[0]]:
            print(f"Updated {s[0][1:].split('.')[0]} from {s[1]} to {updated_dict[s[0]]}\n")

    # save new dict to csv format
    with open(csvfile.replace('.csv', '_new.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in score_dict.items():
            writer.writerow([key, value])    
    return

def split_videos_into_chunks(viddir, tablefile, outdir, seqlen=70, 
                             val_subs=['Subject_1'], fps=30, dataset='hospital'):
    """
    giving the video folder and the intergal csv file,
    split the videos into chunks for train/val,
    rewrite the annotations into new csv files
    """
    assert dataset in ['hospital', 'tulip'], f"Unknown dataset: {dataset} !!"
    # if osp.isdir(outdir):
    #     shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    vidnames = [x.split('.')[0] for x in os.listdir(viddir) if x.endswith('.mp4')]
    # load annotations
    df = pd.read_excel(tablefile, sheet_name="label_info")
    annos_label = pd.DataFrame(df, columns=["vidname","diag","score"]).to_numpy()
    train_dict = {}
    val_dict = {}
    outvid_format = osp.join(outdir, '{:s}')
    
    # split the videos into train/val
    train_names, val_names = [], []
    for vn in vidnames:
        subname = '_'.join(vn.split('_')[:2]) if dataset == 'tulip' else vn.split('_')[0]
        if subname in val_subs:
            val_names.append(vn)
        else:
            train_names.append(vn)
        
    for _vn in tqdm(vidnames):
        vid_path = osp.join(viddir, _vn+'.mp4')
        is_train = _vn in train_names
        vn = _vn.split('_CC')[0] if 'CC' in _vn else _vn
        diag = annos_label[np.where(annos_label[:, 0] == vn)[0]][0, 1]
        score = annos_label[np.where(annos_label[:, 0] == vn)[0]][0, 2]
        # extract frames from video
        img_folder = osp.join('/tmp', _vn)
        os.makedirs(img_folder, exist_ok=True)
        command = ['ffmpeg',
                '-i', vid_path,
                '-f', 'image2',
                '-v', 'error',
                '-vf', f'fps={fps}',
                f'{img_folder}/%06d.png']
        subprocess.call(command)
        img_list = [x for x in os.listdir(img_folder) if x.endswith('.png')]
        img_list = sorted(img_list, key=lambda x: int(x.split('.')[0])) # list of images starting from 000001.png
        # calculate frame index according to seqlen
        last_frame = len(img_list)-1
        if last_frame < seqlen - 6:
            print(f"Video {_vn} has only {last_frame+1} frames !!")
            shutil.rmtree(img_folder)
            continue
        elif last_frame < seqlen-1:
            print(f"Video {_vn} has only {last_frame+1} frames.")
            # copy the last frame to fill the rest
            for i in range(seqlen-1-last_frame):
                shutil.copy(osp.join(img_folder, img_list[-1]), osp.join(img_folder, f"{last_frame+i+2:06d}.png"))
            last_frame = seqlen-1
        if is_train:
            # generate frame indices using STRIDE and cover the whole videos if the rest frames > MIN_REST
            index = np.arange(0, last_frame, STRIDE)
            # remove the last indices if index[-i]+seqlen>last_frame
            while last_frame - index[-1] < seqlen-1:
                index = index[:-1]
                
            if last_frame - index[-1] - seqlen >= MIN_REST-1:
                index = np.append(index, last_frame-seqlen)
        else:
            index = np.arange(0, last_frame, seqlen)
            if last_frame - index[-1] < seqlen-1:
                index = index[:-1]
        assert len(index) > 0, f"Video {vn} has only {last_frame+1} frames."
        # get chunks from extracted frames 
        for i in range(len(index)):
            start, end = index[i], index[i]+seqlen
            img_list_tmp = img_list[start:end]
            img_folder_tmp = img_folder + f'_{i}'
            os.makedirs(img_folder_tmp, exist_ok=True)
            new_vidname = _vn + f'*{i}.mp4'
            output_vid_file = outvid_format.format(new_vidname)
            for img in img_list_tmp:
                shutil.copy(osp.join(img_folder, img), osp.join(img_folder_tmp, img))
            command = ['ffmpeg', '-y', '-threads', '16', '-start_number', f"{int(img_list_tmp[0].split('.')[0])}", '-i', f'{img_folder_tmp}/%06d.png', '-profile:v', 'baseline',
                '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', '-vf', "fps=30", output_vid_file,]
            subprocess.call(command)
            if is_train:
                train_dict[new_vidname] = f"{diag},{score}"
            else:
                val_dict[new_vidname] = f"{diag},{score}"
            shutil.rmtree(img_folder_tmp)
        
        shutil.rmtree(img_folder)
    
    # write csv files
    # shuffle train_dict
    new_index = np.random.permutation(len(train_dict))
    keys = list(train_dict.keys())
    _train_dict = {keys[i]: train_dict[keys[i]] for i in new_index}
    with open(osp.join(outdir, 'train_diag.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in _train_dict.items():
            writer.writerow([key, value.split(',')[0]])
    with open(osp.join(outdir, 'train_updrs.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in _train_dict.items():
            writer.writerow([key, value.split(',')[1]])
    # shuffle val_dict
    new_index = np.random.permutation(len(val_dict))
    keys = list(val_dict.keys())
    _val_dict = {keys[i]: val_dict[keys[i]] for i in new_index}
    with open(osp.join(outdir, 'val_diag.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in _val_dict.items():
            writer.writerow([key, value.split(',')[0]])
    with open(osp.join(outdir, 'val_updrs.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in _val_dict.items():
            writer.writerow([key, value.split(',')[1]])  

    # save the `train_names` and `valid_names`  
    return {'train': train_names, 'val': val_names}

def save_split_names(chunk_dir, outdir='./', nfold=5):
    "Labeling focus on 3-class diag & 4-cls UPDRS."
    chunk_diagcsv_format = osp.join(chunk_dir, 'chunks_{:d}/{:s}_diag_3cls.csv')    
    chunk_gscsv_format = osp.join(chunk_dir, 'chunks_{:d}/{:s}_updrs.csv')
    name_dict = {n: {} for n in range(nfold)}
    train_diag, val_diag = np.zeros(3), np.zeros(3)
    train_updrs, val_updrs = np.zeros(4), np.zeros(4)
    for n in range(nfold):
        chunk_diagcsv_train = chunk_diagcsv_format.format(n, 'train')
        chunk_diagcsv_val = chunk_diagcsv_format.format(n, 'val')
        trains = pd.read_csv(chunk_diagcsv_train, header=None)
        train_names = trains.values[:, 0].tolist()
        train_labels = np.array(trains.values[:, 1].tolist())
        vals = pd.read_csv(chunk_diagcsv_val, header=None)
        val_names = vals.values[:, 0].tolist()
        val_labels = np.array(vals.values[:, 1].tolist())
        name_dict[n]['train'] = train_names
        name_dict[n]['val'] = val_names
        # add to label distributions
        train_diag += np.bincount(train_labels, minlength=3)
        val_diag += np.bincount(val_labels, minlength=3)
        train_labels = pd.read_csv(chunk_gscsv_format.format(n, 'train'), header=None).values[:, 1]
        train_updrs += np.bincount(np.array(train_labels.tolist()), minlength=4)
        val_labels = pd.read_csv(chunk_gscsv_format.format(n, 'val'), header=None).values[:, 1]
        val_updrs += np.bincount(np.array(val_labels.tolist()), minlength=4)
    
    fp = osp.join(outdir, 'split_names.json')
    joblib.dump(name_dict, fp)
    
    # save the class distributions to csv
    cls_fp = osp.join(outdir, 'cls_distributions.csv')
    with open(cls_fp, 'w') as f:
        f.write('cls_type,0,1,2,3\n')
        f.write('train_diag,' + ','.join(train_diag.astype(str)) + '\n')
        f.write('val_diag,' + ','.join(val_diag.astype(str)) + '\n')
        f.write('train_updrs,' + ','.join(train_updrs.astype(str)) + '\n')
        f.write('val_updrs,' + ','.join(val_updrs.astype(str)) + '\n')        

    return
    
def get_3cls_csv(diag_csv, score_csv):
    "convert multi-class labels to 3-class labels"
    diag_df = pd.read_csv(diag_csv, header=None)
    diag_dict = {x[0]: x[1] for x in diag_df.values}
    for k,v in diag_dict.items():
        if v == 0:
            if No_Healthy: raise ValueError("Should not contain healthy samples in the dataset!!")
            diag_dict[k] = 0
        elif v == 1 or v==3:
            diag_dict[k] = 0 if No_Healthy else 1
        else:
            diag_dict[k] = 1 if No_Healthy else 2
    score_df = pd.read_csv(score_csv, header=None)
    score_dict = {x[0]: x[1] for x in score_df.values}
    for k,v in score_dict.items():
        if v>2:
            score_dict[k] = 2
    with open(diag_csv.replace('.csv', '_3cls.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in diag_dict.items():
            writer.writerow([key, value])
    with open(score_csv.replace('.csv', '_3cls.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in score_dict.items():
            writer.writerow([key, value])   
    
    return

def get_average_class_distribution(label_file='data/tulip_label_60.xlsx', video_dir='datasets/tulip/', split_names_fp=None,): #'./datasets/tulip/split_names.json'):
    "Calculate per-class samples in each fold"
    df = pd.read_excel(label_file, sheet_name='label_info')
    max_updrs = np.max(df['score'].values)
    max_diag = np.max(df['diag'].values)
    score_dict = {int(n):0 for n in range(max_updrs+1)}
    diag_dict = {int(n):0 for n in range(max_diag+1)}
    if split_names_fp is not None:
        try:
            split_names = joblib.load(split_names_fp)
        except KeyError: # the json has been saved with browser-compatible format
            with open(split_names_fp, 'r') as f:
                split_names = json.load(f)
            split_names = {int(k):v for k,v in split_names.items()}

        for fold in split_names.keys():
            for split in ['train', 'val']:
                for vidname in tqdm(split_names[fold][split]):
                    vname = vidname.split('*')[0].replace('f', '')
                    vname = re.sub(r'_CC\d+', '', vname)
                    score = int(df.loc[df['vidname'] == vname, 'score'].values[0])
                    diag = int(df.loc[df['vidname'] == vname, 'diag'].values[0])
                    score_dict[score] += 1
                    diag_dict[diag] += 1
    else:
        assert osp.isdir(video_dir)
        folders = [x for x in os.listdir(video_dir) if 'chunk' in x and osp.isdir(osp.join(video_dir, x))]
        for fold in range(len(folders)):
            for split in ['train', 'val']:
                diag_csv = osp.join(video_dir, folders[fold], f'{split}_diag.csv')
                score_csv = osp.join(video_dir, folders[fold], f'{split}_updrs.csv')
                diag_df = pd.read_csv(diag_csv, header=None)
                score_df = pd.read_csv(score_csv, header=None)
                for x in diag_df.values:
                    diag_dict[x[1]] += 1
                for x in score_df.values:
                    score_dict[x[1]] += 1
    # print the average class distribution
    nfold = len(split_names) if split_names_fp is not None else len(folders)
    print('Average class distribution in each fold:')
    print('UPDRS:')
    for k,v in score_dict.items():
        print('UPDRS {:d}: {:f}'.format(k, v/nfold))
    print('Diagnosis:')
    for k,v in diag_dict.items():
        print('Diagnosis {:d}: {:f}'.format(k, v/nfold))
    return

def draw_confusion_matrix(name, cm_dir=None, type='updrs'):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sample 4x4 confusion matrix
    # confusion_matrix = np.array([
    #     [68, 20,1,0],
    #     [6,140,3,3],
    #     [0,39,23,3],
    #     [0,21,6,41]
    # ])
    if cm_dir is not None:
        cm_lists = [x for x in os.listdir(cm_dir) if x.endswith('.txt') and 'metrics' not in x]
        # sort the cm_lists
        cm_lists = sorted(cm_lists, key=lambda x: int(x.split('fold-')[-1].split('.')[0]))
        # load per-fold confusion matrix
        if type == 'updrs':
            confusion_matrix = np.zeros((3, 3)) 
        else:
            raise ValueError("Only support UPDRS now !!")
        fold_acc = {}
        for cm in cm_lists:
            with open(osp.join(cm_dir, cm), 'r') as f:
                lines = f.readlines()
                # load the fold conf_mat and calculate per-fold accuracy
                cm_fold = np.zeros_like(confusion_matrix) 
                for i in range(len(lines)):
                    cm_fold[i] += np.array([int(x) for x in lines[i].strip().split()])
                acc_fold = np.sum(np.diag(cm_fold))/np.sum(cm_fold)
                fold_acc[cm.split('_')[-1].split('.')[0]] = acc_fold
                confusion_matrix += cm_fold

    # calculate metrics based on conf_mat
    accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
    precision = np.nan_to_num(np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=0),0)
    recall = np.nan_to_num(np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=1),0)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    weighted_f1 = np.sum(f1*np.sum(confusion_matrix, axis=1)/np.sum(confusion_matrix))

    # save to txt file
    with open(osp.join(cm_dir, f'{name}_metrics.txt'), 'w') as f:
        for k, v in fold_acc.items():
            f.write(f'{k}: {v:.4f}\t')
        f.write(f'\nAccuracy: {accuracy}\n')
        f.write(f'F1: {f1.mean()}\n')
        f.write(f'Precision: {precision.mean()}\n')
        f.write(f'Recall: {recall.mean()}\n')
        f.write(f'Weighted F1: {weighted_f1}\n')
    # confusion_matrix = np.array([
    #     [59,24,2,4],
    #     [5,130,8,10],
    #     [3,28,26,8],
    #     [0,17,13,38]
    # ])

    # Sample 5x5confusion matrix
    # confusion_matrix = np.array([
    #     [33,3,3,0,5],
    #     [3,51,9,3,1],
    #     [0,19,100,0,2],
    #     [0,1,3,59,0],
    #     [0,7,10,0,58],
    # ])
    # confusion_matrix = np.array([
    #     [32,3,4,0,5],
    #     [0,53,7,1,6],
    #     [0,4,100,0,17],
    #     [0,0,1,61,1],
    #     [0,0,4,0,71],
    # ])

    # Create a heatmap with larger label and tick sizes
    plt.figure(figsize=confusion_matrix.shape)
    ax = sns.heatmap(confusion_matrix, annot=False, cmap='Blues', cbar=True, square=True, linewidths=0.5, linecolor='white')

    # Customize label and tick sizes
    ax.set_xlabel('Predicted labels', fontsize=20)
    ax.set_ylabel('True labels', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Customize tick labels
    class_labels = CLASS_LABEL_DICT['updrs'] if 'updrs' in name else CLASS_LABEL_DICT['diag']
    ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=15)
    ax.set_yticklabels(class_labels, rotation=0, fontsize=15)
    
    # # Add color bar with label
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_label('Samples', size=15)
    # colorbar.ax.tick_params(labelsize=12)

    plt.savefig(osp.join(cm_dir if cm_dir else './images', name+'.png'), dpi=300, bbox_inches='tight')
    # plt.show()

def split_videos_per_subjects(csv_label='./datasets/parkinson_label.csv', 
                              vid_dir='./datasets/parkinson/',
                              out_dir='./datasets/parkinson_cv/',
                              frame_num=70, 
                              fps=30,):
    "take the first `frame_num` frames of each video and split them into N_subject folds"
    assert osp.isdir(vid_dir), f"Video directory {vid_dir} not found !!"
    vid_names = [osp.join(vid_dir, x) for x in os.listdir(vid_dir) if x.endswith('.mp4')]
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(csv_label, header=0).to_numpy()
    subnames = defaultdict(int)

    for vn in tqdm(vid_names):
        vid_name = osp.basename(vn)
        subname = vid_name.split('_')[0]
        vid_name = vid_name.split('.')[0]
        # extract the first 70 frames from video
        tmp_dir = osp.join(out_dir, vid_name)
        os.makedirs(tmp_dir, exist_ok=True)
        command = ['ffmpeg',
                '-i', vn,
                '-f', 'image2',
                '-v', 'error',
                '-vf', f'fps=30',
                f'{tmp_dir}/%06d.png']
        subprocess.call(command)
        # make video from frames
        command = ['ffmpeg', '-y', '-threads', '16', '-start_number', '3', '-i', f'{tmp_dir}/%06d.png', '-profile:v', 'baseline',
                   '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', '-vframes', f'{frame_num}', '-vf', f"fps={fps}", f'{out_dir}/{vid_name}.mp4',]
        subprocess.call(command)
        # remove the tmp folder
        shutil.rmtree(tmp_dir)
        subnames[subname] += 1

    # split video into train/valid by names
    split_dict = defaultdict(dict)
    for nid, subid in enumerate(subnames.keys()):
        train_csv_name = osp.join(out_dir, f'train_updrs_{nid:02d}.csv')
        val_csv_name = osp.join(out_dir, f'val_updrs_{nid:02d}.csv')
        split_dict[nid]['train'] = []
        split_dict[nid]['val'] = []
        for vname, label in labels:
            if subid in vname:
                split_dict[nid]['val'].append(vname)  
                with open(val_csv_name, 'a') as f:
                    f.write(f'{vname}.mp4,{label}\n')
            else:
                split_dict[nid]['train'].append(vname)
                with open(train_csv_name, 'a') as f:
                    f.write(f'{vname}.mp4,{label}\n')

    joblib.dump(split_dict, osp.join(out_dir, 'split_names.json'))

    return

def get_eval_dataset(base_folder='./datasets/miccai_10_fold/', fold=10):
    out_dir = osp.join(base_folder, 'mix')
    os.makedirs(out_dir, exist_ok=True)
    eval_label = {} 
    for nf in range(fold):
        vid_dir = osp.join(base_folder, f'chunks_{nf}')
        valid_label = pd.read_csv(osp.join(vid_dir, 'val_updrs.csv'), header=None).to_numpy()
        # valid_names = joblib.load(osp.join(base_folder, 'split_names.json'))[nf]['val']
        for line in tqdm(valid_label):
            vname = line[0]
            vlabel = line[1]
            if vlabel < 3:
                shutil.copy(osp.join(vid_dir, vname), osp.join(out_dir, vname))
                eval_label[vname] = vlabel
    # write the label to csv
    with open(osp.join(out_dir, 'eval_updrs.csv'), 'w') as f:
        for k, v in eval_label.items():
            f.write(f'{k},{v}\n')
    return

def split_real_dataset(base_folder='./datasets/miccai_10_fold/', fold=10):
    out_dir = osp.join(base_folder, 'real_3cls')
    train_dir = osp.join(out_dir, 'train')
    test_dir = osp.join(out_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    split_names = joblib.load(osp.join(base_folder, 'split_names.json'))
    eval_dict, train_dict = {}, {} 
    for nf in range(fold):
        vid_dir = osp.join(base_folder, f'chunks_{nf}')
        valid_label = pd.read_csv(osp.join(vid_dir, 'val_updrs.csv'), header=None).to_numpy()
        valid_names = split_names[nf]['val']
        train_label = pd.read_csv(osp.join(vid_dir, 'train_updrs.csv'), header=None).to_numpy()
        train_names = split_names[nf]['train']
        for vname in tqdm(valid_names):
            vlabel = valid_label[np.where(valid_label[:, 0] == vname)[0]][0, 1]
            if vlabel < 3:
                # shutil.copy(osp.join(vid_dir, vname), osp.join(test_dir, vname))
                eval_dict[vname] = vlabel
        for vname in tqdm(train_names):
            tlabel = train_label[np.where(train_label[:, 0] == vname)[0]][0, 1]
            if tlabel < 3:
                # shutil.copy(osp.join(vid_dir, vname), osp.join(train_dir, vname))
                train_dict[vname] = tlabel
    # write the label to csv
    # to split the train names into train/valid
    eval_subjects = [x.split('_')[0] if 'vid' in x else x.split('-')[0] for x in list(eval_dict.keys())]
    eval_subjects = list(set(eval_subjects))
    subjects = [x.split('_')[0] if 'vid' in x else x.split('-')[0] for x in list(eval_dict.keys())]
    subjects = list(set(subjects))
    assert len(subjects) == len(eval_subjects), "The subjects in train and eval are not the same !!"
    # split the subjects into train/valid/test: 0.4 0.3 0.3
    train_subjects = np.random.choice(subjects, int(len(subjects)*0.4), replace=False)
    valid_subjects = np.random.choice(list(set(subjects) - set(train_subjects)), int(len(subjects)*0.3), replace=False)
    test_subjects = list(set(subjects) - set(train_subjects) - set(valid_subjects))
    # remove videos containing test subjects from train_dir
    for vname in os.listdir(train_dir):
        if vname.split('_')[0] in test_subjects:
            os.remove(osp.join(train_dir, vname))
            train_dict.pop(vname)
    # write the train/val csv files
    trainf =  open(osp.join(train_dir, 'train_updrs.csv'), 'w')
    valf =  open(osp.join(train_dir, 'val_updrs.csv'), 'w')
    for k, v in train_dict.items():
        if k.split('_')[0] in train_subjects:
            trainf.write(f'{k},{v}\n')
        else:
            valf.write(f'{k},{v}\n')

    with open(osp.join(test_dir, 'test_updrs.csv'), 'w') as f:
        for k, v in eval_dict.items():
            if k.split('_')[0] in test_subjects:
                f.write(f'{k},{v}\n')
    return

def count_sequence_num(base_dir):
    "count respectively the number of videos used in train/valid and test sets"
    train_dir = osp.join(base_dir, 'train')
    test_dir = osp.join(base_dir, 'test')
    train_csv = pd.read_csv(osp.join(train_dir, 'train_updrs.csv'), header=None).to_numpy()
    valid_csv = pd.read_csv(osp.join(train_dir, 'val_updrs.csv'), header=None).to_numpy()
    test_csv = pd.read_csv(osp.join(test_dir, 'test_updrs.csv'), header=None).to_numpy()
    train_vids = [x.split('*')[0] for x in train_csv[:, 0]]
    valid_vids = [x.split('*')[0] for x in valid_csv[:, 0]]
    test_vids = [x.split('*')[0] for x in test_csv[:, 0]]
    train_vids = list(set(train_vids))
    valid_vids = list(set(valid_vids))
    test_vids = list(set(test_vids))
    print(f"Train videos: {len(train_vids)}")
    print(f"Valid videos: {len(valid_vids)}")
    print(f"Test videos: {len(test_vids)}")

    return

def merge_dataset(real_dir='datasets/parkinson_cv/', syn_dir='datasets/'):
    real_train_csv = pd.read_csv(osp.join(real_dir, 'train_updrs.csv'), header=None).to_numpy()
    real_val_csv = pd.read_csv(osp.join(real_dir, 'val_updrs.csv'), header=None).to_numpy()
    syn_train_csv = pd.read_csv(osp.join(syn_dir, 'train_updrs.csv'), header=None).to_numpy()
    syn_val_csv = pd.read_csv(osp.join(syn_dir, 'val_updrs.csv'), header=None).to_numpy()
    # merge the csv files
    train_csv = np.vstack((real_train_csv, syn_train_csv))
    val_csv = np.vstack((real_val_csv, syn_val_csv))
    # permute the rows
    train_csv = train_csv[np.random.permutation(len(train_csv))]
    val_csv = val_csv[np.random.permutation(len(val_csv))]
    # save the csv files
    with open(osp.join(real_dir, 'train_updrs_merge.csv'), 'w') as f:
        for k, v in train_csv:
            f.write(f'{k},{v}\n')
    with open(osp.join(real_dir, 'val_updrs_merge.csv'), 'w') as f:
        for k, v in val_csv:
            f.write(f'{k},{v}\n')
    return

def visualize_example():
    fp = 'datasets/tulip/subject1_gait_poses.npy'
    data = np.load(fp)/1e3
    # show 3D points
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for d in tqdm(data):
        ax.clear()
        # remove the root position
        d -= d[19:20]
        for idx, joint in enumerate(d):
            ax.scatter(joint[0], joint[1], joint[2],)
            ax.text(joint[0], joint[1], joint[2], f'{idx}', fontsize=6)
        # fix the grid
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.set_zlim(-1., 1.)
        plt.draw()
        if plt.waitforbuttonpress(0.5):
            break
    plt.close()
    return

def crop_video_with_bbox(vid_dir='datasets/orig_tulip/video_30fps', 
                         tracking_bbox='datasets/orig_tulip/tulip_465_bbox.json',
                         out_dir='datasets/subseq_tulip',
                         fps = 30):
    "crop the video based on bounding boxes and resize into 256x256"
    import cv2
    vid_names = [x for x in os.listdir(vid_dir) if x.endswith('.mp4')]
    os.makedirs(out_dir, exist_ok=True)
    # load the tracking bbox
    bbox_dict = joblib.load(tracking_bbox)
    for vn in tqdm(vid_names):
        # extract frames from video
        vid_path = osp.join(vid_dir, vn)
        img_folder = osp.join('./tmp', vn.replace('.mp4', f'_{fps}'))
        os.makedirs(img_folder, exist_ok=True)
        command = ['ffmpeg',
                '-i', vid_path,
                '-f', 'image2',
                '-v', 'error',
                '-vf', f'fps={fps}',
                f'{img_folder}/%06d.png']
        subprocess.call(command)
        img_list = [x for x in os.listdir(img_folder) if x.endswith('.png')]
        img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
        # get the bboxes for all the subsequences
        seqnames = [x for x in bbox_dict.keys() if x.startswith(vn.split('.')[0])]
        for sn in seqnames:
            bbox = bbox_dict[sn]['bbox']
            fid = bbox_dict[sn]['frame_ids']
            imgnames = [f'{x+1:06d}.png' for x in fid]
            out_seq_folder = osp.join(out_dir, sn)
            if osp.isdir(out_seq_folder):
                shutil.rmtree(out_seq_folder)
            os.makedirs(out_seq_folder)

            for i, img in enumerate(imgnames):
                img_path = osp.join(img_folder, img)
                frame = cv2.imread(img_path)
                # retrieve the bbox
                c_x, c_y, bsize = bbox[i]
                bsize *= 224
                bsize = int(bsize/2)
                x1, y1 = int(c_x)-bsize, int(c_y)-bsize
                x2, y2 = int(c_x)+bsize, int(c_y)+bsize
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                y2 = min(y2, frame.shape[0])
                x2 = min(x2, frame.shape[1])
                vis = False
                if vis:
                    # draw the bbox
                    copy_frame = frame.copy()
                    cv2.rectangle(copy_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow('frame', copy_frame)
                    k = cv2.waitKey()
                    if k == ord('q'):
                        cv2.destroyAllWindows()
                        vis = False
                # crop the frame
                crop_frame = frame[y1:y2, x1:x2]
                # padd into square
                w = x2-x1
                h = y2-y1
                if w > h:
                    crop_frame = np.concatenate((crop_frame, np.zeros((w-h, w, 3), dtype=np.uint8)), axis=0)
                elif h > w:
                    crop_frame = np.concatenate((crop_frame, np.zeros((h, h-w, 3), dtype=np.uint8)), axis=1)
                crop_frame = cv2.resize(crop_frame, (256, 256))
                cv2.imwrite(osp.join(out_seq_folder, f'{i+1:06d}.png'), crop_frame)
            try: cv2.destroyAllWindows()
            except: pass
            # generate video from crop frames
            command = ['ffmpeg', '-y', '-threads', '16', '-start_number', '3', '-i', f'{out_seq_folder}/%06d.png', '-profile:v', 'baseline',
                    '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', '-vf', f"fps={fps}", f'{out_dir}/{sn}.mp4',]
            subprocess.call(command)
            shutil.rmtree(out_seq_folder)

        shutil.rmtree(img_folder)

    return

def gold_standard2label(csv_file='datasets/orig_tulip/gait_label.csv',
                        vid_dir='datasets/orig_tulip/video_30fps'):
    """
    Convert gold standard  csv to xlsx \n
    Add `label_info` sheet to the xlsx file
    """
    # load csv file
    data = pd.read_csv(csv_file, header=0)
    # select columns
    data = data[['Subject', 'gold_standard', 'diag']].to_numpy()
    label_info = {'vidname':[], 'diag':[], 'score':[]}

    # process the videos in the video directory
    vid_names = [x for x in os.listdir(vid_dir) if x.endswith('.mp4')]
    # sort the items in the name list
    vid_names = sorted(vid_names, key=lambda x: int(x.split('_')[1]+x.split('Camera')[1][0]))
    for vn in vid_names:
        subname = '_'.join(vn.split('_')[:2])
        sub_id = int(subname.split('_')[1])
        diag = data[np.where(data[:,0]==sub_id)[0],2][0]
        diag = 0 if diag == 'HT' else 1
        score = data[np.where(data[:,0]==sub_id)[0], 1][0]
        label_info['vidname'].append(vn.split('.')[0])
        label_info['diag'].append(diag)
        label_info['score'].append(score)
    
    # convert dictioanry to dataframe
    df = pd.DataFrame(label_info)
    # remove the first column
    # save to xlsx file
    with pd.ExcelWriter('data/tulip_label_60.xlsx') as writer:
        df.to_excel(writer, sheet_name='label_info', index=False)

    return

if __name__ == '__main__':
    # ================== public PD data set ================== #
    # crop_video_with_bbox()
    # split_real_dataset()
    # draw_confusion_matrix(name='conf_mat_updrs_gava-clip_tulip', cm_dir='logs/GaVA-CLIP_tulip_results', type='updrs')
    # get_average_class_distribution()
    # ================== Robertsau hospital data set ================== #
    nfold = 10
    vratio = 1/nfold *2
    sratio = 1/nfold
    num_frames = 70
    dataset_name = 'tulip'
    tablefile = 'data/label_118.xlsx' if dataset_name == 'hospital' else 'data/tulip_label_60.xlsx'
    # get the video names from xlsx file
    with open(tablefile, 'rb') as f:
        table = pd.read_excel(f)
        vidnames = table['vidname'].values.tolist()
    chunk_csv_format = 'datasets/{:s}/chunks_{:d}/{:s}_{:s}.csv'
    # TODO write val_subs for Robertsau hospital data
    subnames = ['_'.join(x.split('_')[:2]) for x in vidnames] if dataset_name == 'tulip' else [x.split('_')[0] for x in vidnames]
    subnames = list(set(subnames))
    # sort the subnames
    subnames = sorted(subnames, key=lambda x: int(x.split('_')[1]))
    num_sub = len(subnames)
    sub_per_fold = int(num_sub/nfold)
    assert sub_per_fold > 0, "Number of subjects per fold should be greater than 0 !!"
    subname_set = [subnames[i*sub_per_fold:(i+1)*sub_per_fold] for i in range(nfold-1)]
    subname_set.append(subnames[(nfold-1)*sub_per_fold:])
    split_names = {}
    for n in range(nfold):
        # use different train/val csv files for each fold
        split_names[n] = split_videos_into_chunks(
            viddir='datasets/hospital/mp4_cropped' if dataset_name == 'hospital' else 'datasets/subseq_tulip',
            tablefile=tablefile,
            outdir=f'datasets/{dataset_name}/chunks_{n}',
            val_subs=subname_set[n],
            seqlen=num_frames,
            dataset=dataset_name,     
        )
        # get_3cls_csv(chunk_csv_format.format(dataset_name, n, 'train', 'diag'), chunk_csv_format.format(n, 'train', 'updrs'))
        # get_3cls_csv(chunk_csv_format.format(dataset_name, n, 'val', 'diag'), chunk_csv_format.format(n, 'val', 'updrs'))
    
    # save_split_names(chunk_dir=f'datasets/{dataset_name}', outdir=f'datasets/{dataset_name}', nfold=nfold)
    # with open(f'datasets/{dataset_name}/split_names.json', 'w') as f:
    #     json.dump(split_names, f, indent=4)

    get_average_class_distribution()