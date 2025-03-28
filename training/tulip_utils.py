import os
import os.path as osp
import sys
sys.path.insert(0, os.getcwd())

import joblib
import numpy as np

from tqdm import tqdm
import shutil
import cv2
import subprocess

from collections import defaultdict

import scipy.signal as signal

def get_reduced_dict(wham_data, keep_frames=[]):
    '''remoev unnecessary keys from the wham data'''
    keep_keys = ['pose', 'betas', 'frame_ids', 'trans']
    out_dict = {k: wham_data[k] for k in keep_keys}
    if len(keep_frames)>0:
        for k, v in out_dict.items():
            if isinstance(v, np.ndarray):
                out_dict[k] = v[keep_frames]
            elif isinstance(v, list):
                out_dict[k] = [v[i] for i in keep_frames]
    return out_dict

def post_process_tulip(data_dir='datasets/orig_tulip', fps=30):
    '''Align the wham with tracking results'''
    video_dir = osp.join(data_dir, 'video_30fps')
    wham_dir = osp.join(data_dir, 'wham')
    tracking_dir = osp.join(data_dir, 'tracking')
    assert osp.exists(video_dir), f'Video dir {video_dir} does not exist'
    assert osp.exists(wham_dir), f'Wham dir {wham_dir} does not exist'
    assert osp.exists(tracking_dir), f'Tracking dir {tracking_dir} does not exist'

    video_list = [x for x in os.listdir(video_dir) if x.endswith('.mp4')]

    # save good bounding boxes to `bbox.json`
    bbox_dict = defaultdict(dict)
    # create folder to save reduced wham pickle files
    wham_reduced_dir = osp.join(data_dir, 'wham_reduced')
    os.makedirs(wham_reduced_dir, exist_ok=True)
    for video_name in tqdm(video_list):
        vis = False
        wham_fp = osp.join(wham_dir, video_name.replace('.mp4', '_wham.pkl'))
        tracking_fp = osp.join(tracking_dir, video_name.replace('.mp4', '_tracking.pkl'))
        video_fp = osp.join(video_dir, video_name)
        # check the number of frames
        # # load wham
        wham = joblib.load(wham_fp)
        # # load tracking
        tracking = joblib.load(tracking_fp)
        bbox = tracking['bbox']
        # # check the number of frames
        if len(wham['frame_ids']) != fps*60:
            frame_ids = wham['frame_ids']
            print(f'Number of frames only {len(frame_ids)} for {video_name}')
            assert len(frame_ids) == len(bbox), f"Different wham frames and bbox length: wham {len(frame_ids)} vs bbox {len(bbox)}"

        # cut out the turning frames
        # # find the turning frames by locating the local extremum of bbox center
        c_x = np.array([x[0] for x in bbox])
        c_y = np.array([x[1] for x in bbox])
        # # find the most variable dimension
        c_x_diff = np.abs(np.diff(c_x)).sum()
        c_y_diff = np.abs(np.diff(c_y)).sum()
        # initialize the filter
        order = 2
        cutoff = 0.4 # cut-off frequency in Hz
        b, a = signal.butter(order, cutoff/ (fps/2))
        if c_x_diff > c_y_diff:
            # filter c_x
            pos = signal.filtfilt(b, a, c_x)
        else:
            # filter c_y
            pos = signal.filtfilt(b, a, c_y)

        vis_center = False
        # if vis_center:
        #     import matplotlib.pyplot as plt
        #     plt.plot(pos)
        #     plt.show()
        # # find the turning points
        turn_pt = signal.argrelextrema(pos, np.greater)[0]
        turn_pt_less = signal.argrelextrema(pos, np.less)[0]
        turn_pt = np.concatenate([turn_pt, turn_pt_less])
        # order the turning points
        turn_pt = np.sort(turn_pt)
        # # find the turning frames
        duration = fps*2
        turning_frames = []
        all_length = len(wham['frame_ids'])
        start_ids = []
        for tp in turn_pt:
            start = max(0, tp-duration/2)
            end = min(tp+duration/2, all_length-1)
            start, end = int(start), int(end)
            if len(start_ids)==0 or start>(turning_frames[-1][-1])+fps:
                start_ids.append(start)
                turning_frames.append([*range(start, end+1)])
            elif start<=(turning_frames[-1][-1])+fps:
                # combine current turning frames with the last one
                turning_frames[-1] = [*range(start_ids[-1], end+1)]

        # check for duplicate turning
        all_turning_frames = np.concatenate(turning_frames)
        assert np.unique(all_turning_frames).size == all_turning_frames.size, f'Duplicate turning frames found for {video_name}'

        # visualize the turning frames
        if vis_center:
            import matplotlib.pyplot as plt
            plt.plot(pos)
            for tf in turning_frames:
                plt.plot(tf, pos[tf], 'ro')
            plt.show()

        # save the keep frames as subsequence
        for idt, (last_start, start) in enumerate(zip(start_ids[:-1], start_ids[1:])):
            keep_frames = np.array([*range(last_start+duration, start)])
            wham_dict = get_reduced_dict(wham, keep_frames)
            seq_name = video_name.replace('.mp4', f'_CC{idt}')
            joblib.dump(wham_dict, osp.join(wham_reduced_dir, seq_name + f'_wham.pkl'))
            bbox_dict[seq_name]['bbox'] = bbox[keep_frames]
            bbox_dict[seq_name]['frame_ids'] = wham_dict['frame_ids']
        # calculate the keep frames from the turning frames
        # if vis:
        #     # extract frames from video
        #     img_folder = osp.join('./tmp', video_name.replace('.mp4',''))
        #     os.makedirs(img_folder, exist_ok=True)
        #     command = ['ffmpeg',
        #             '-i', video_fp,
        #             '-f', 'image2',
        #             '-v', 'error',
        #             '-vf', f'fps={fps}',
        #             f'{img_folder}/%06d.png']
        #     subprocess.call(command)
        #     img_list = [x for x in os.listdir(img_folder) if x.endswith('.png') and int(osp.basename(x).split('.')[0]) in frame_ids+1]
        #     img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
        #     for i, img in enumerate(img_list):
        #         img_path = osp.join(img_folder, img)
        #         frame = cv2.imread(img_path)
        #         # retrieve the bbox
        #         c_x, c_y, bsize = bbox[i]
        #         bsize *= 224
        #         bsize = int(bsize/2)
        #         x1, y1 = int(c_x)-bsize, int(c_y)-bsize
        #         x2, y2 = int(c_x)+bsize, int(c_y)+bsize
        #         x1 = max(x1, 0)
        #         y1 = max(y1, 0)
        #         y2 = min(y2, frame.shape[0])
        #         x2 = min(x2, frame.shape[1])
        #         if vis:
        #             # draw the bbox
        #             copy_frame = frame.copy()
        #             cv2.rectangle(copy_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.imshow('frame', copy_frame)
        #             k = cv2.waitKey()
        #             if k == ord('q'):
        #                 cv2.destroyAllWindows()
        #                 vis = False
        #     try: cv2.destroyAllWindows()
        #     except: pass
        #     shutil.rmtree(img_folder)         

        # # save the reduced wham output and `bbox_dict`
        # bbox_dict[video_name] = bbox[keep_frames]
        # joblib.dump(wham_dict, osp.join(wham_reduced_dir, video_name.replace('.mp4', '_wham.pkl')))
    
    # save the bbox_dict
    print(f'Saving {len(bbox_dict)} bounding box sequences..')
    joblib.dump(bbox_dict, osp.join(data_dir, f'tulip_{len(bbox_dict)}_bbox.json'))

    return

if __name__=='__main__':
    post_process_tulip()