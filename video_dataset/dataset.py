#!/usr/bin/env python

import os, sys
import os.path as osp
from typing import Optional
import av
import io
import numpy as np

import pickle

import torch
torch.manual_seed(0) 
from torchvision import transforms

from .transform import create_random_augment, random_resized_crop


NUM_COMB = 70 # number of combinations for NTE

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, list_path: str, data_root: str,
        num_spatial_views: int, num_temporal_views: int, random_sample: bool,
        num_frames: int, sampling_rate: int, spatial_size: int,
        mean: torch.Tensor, std: torch.Tensor,
        auto_augment: Optional[str] = None, interpolation: str = 'bicubic',
        mirror: bool = False, is_train: bool = True,
        cls_type: str = '', num_folds: int = 1,
        add_nte: bool = False,
    ):
        self.data_root = data_root
        self.nte_root = osp.join(data_root, 'nte')
        self.interpolation = interpolation
        self.spatial_size = spatial_size

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate
        self.num_folds = num_folds
        self.cls_type = cls_type
        # whether include additional CL with NTE
        self.add_nte = add_nte
        
        self.is_train = is_train # control the outputs of __getitem__

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

        # load the video names
        if self.num_folds>1:
            assert self.cls_type in ['updrs', 'updrs_3cls', 'diag', 'diag_3cls']
            # load the names by fold
            self.data_list = []
            for nf in range(self.num_folds):
                list_path = osp.join(data_root, f'chunks_{nf}', f'val_{self.cls_type}.csv')
                with open(list_path) as f:
                    data_list = f.read().splitlines()
                # prepend folder name to video names
                data_list = [osp.join(f'chunks_{nf}', line) for line in data_list]
                self.data_list.extend(data_list)
        else:
            with open(list_path) as f:
                self.data_list = f.read().splitlines()


    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        line = self.data_list[idx]
        _path, label = line.split(',')
        path = os.path.join(self.data_root, _path)
        label = int(label)

        container = av.open(path)
        frames = {}
        # Here, frame is the video frame (image), not the frame number
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]

        if self.random_sample:
            frame_idx = self._random_sample_frame_idx(len(frames))
            frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
            frames = torch.as_tensor(np.stack(frames)).float() / 255.

            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(1), frames.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                frames = aug_transform(frames)
                frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                frames = frames.permute(0, 2, 3, 1)

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            frames = random_resized_crop(
                frames, self.spatial_size, self.spatial_size,
            )
            
        else:
            frames = [x.to_rgb().to_ndarray() for x in frames]
            frames = torch.as_tensor(np.stack(frames))
            frames = frames.float() / 255.

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            
            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode='bilinear', align_corners=False,
            )

            frames = self._generate_spatial_crops(frames)
            frames = sum([self._generate_temporal_crops(x) for x in frames], [])
            #if len(frames) > 1:
            #frames = torch.stack(frames)
            frames = frames[0]

        if self.is_train:
            # retrieve NTE and valid id for video data
            if self.add_nte: 
                # load video NTE from the NTE path
                if 'SUB' in _path:
                    npy_fn = "_".join(osp.basename(_path).split('_')[:-1]) + '.npy'
                else:
                    npy_fn = _path.replace('fvid','vid').split('*')[0].split('.')[0] + '.npy'
                if osp.isfile(osp.join(self.nte_root, npy_fn)):
                    vid_nte = torch.from_numpy(np.load(osp.join(self.nte_root, npy_fn))).float()
                else:
                    vid_nte = torch.zeros(NUM_COMB, 512).float()
                return frames, label, vid_nte
            else:
                return frames, label, torch.zeros(NUM_COMB, 512).float()
        else:
            vidname = path.split('/')[-1].split('.')[0]
            return frames, label, vidname

    def _generate_temporal_crops(self, frames):
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0

class DummyMemoDataset(torch.utils.data.Dataset):
    
    def __init__(self, num_cls=2, batch_size=64, embed_size=512,):
        self.num_cls = num_cls
        self.batch_size = batch_size
        self.embed_size = embed_size
        
    def __len__(self):
        return self.batch_size*1000
    
    def __getitem__(self, idx):
        # return torch.zeros(self.num_cls, self.embed_size), torch.zeros(self.num_cls, self.embed_size), 0
        return torch.zeros(self.num_cls, self.embed_size), 0
    
class MemoryDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path='', cls_type='', batch_size=64, for_zero_shot=True):
        self.batch_size = batch_size
        self.cls_type = cls_type.lower()
        assert self.cls_type in ('updrs', 'diag', 'diag_3cls')
        assert osp.isfile(data_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, dict)
        self.data = data['embeds']
        # new content for ke
        # self.values = data['values']
        self.labels = data[cls_type.split('_')[0]].reshape(-1)
        valid_idx = np.where(self.labels >= 0)[0]
        self.labels = self.labels[valid_idx]
        self.data = self.data[valid_idx]
        # select data for zero-shot model training
        if (cls_type in ['diag_3cls', 'updrs_3cls']) and self.labels.max()>2:
            get_3cls = lambda x: 0 if x==0 else 1 if (x==1 or x==3) else 2
            self.labels = np.array([get_3cls(x) for x in self.labels])
        if for_zero_shot and cls_type == 'diag': # remove early-stage and severe AD
            early_AD = np.where(self.labels==2)[0]
            severe_AD = np.where(self.labels==4)[0]
            severe_DLB = np.where(self.labels==3)[0]
            self.labels[severe_DLB] = 2
            self.labels = np.delete(self.labels, np.concatenate([early_AD, severe_AD]))
            self.data = np.delete(self.data, np.concatenate([early_AD, severe_AD]), axis=0)
        elif for_zero_shot and cls_type == 'diag_3cls':
            valid_idx = np.where(self.labels > 0)[0]
            self.labels = self.labels[valid_idx]-1  
            self.data = self.data[valid_idx] 
        
        # shuffle
        new_idx = np.random.permutation(len(self.labels))
        self.labels = self.labels[new_idx]
        self.data = self.data[new_idx]
        # new content for ke
        # self.values = self.values[new_idx]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # return self.data[idx], self.labels[idx], self.labels[idx]
        return self.data[idx], self.labels[idx]
