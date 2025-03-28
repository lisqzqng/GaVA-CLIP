"""
Implementation of Knowledge-Aware Prompts.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import os.path as osp

import torch
torch.manual_seed(0) 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List


class ContextualPromptLearnerUP(nn.Module):
    """
    Intialize N tokens for each class.
    """
    def __init__(self, 
                 use_cntn:bool,
                 cntn_split:bool,
                 uni_mlp:bool,
                 use_disc:bool,
                 emb_dim=128,
                 out_dim=512,
                 inp_dim=768,
                 n_cls=4,
                 n_tokens=16, 
                 cls_type='updrs',
                 knowledge_version:List[str]=['v0'],
                 use_descriptor:bool=False,
                 ):
        super(ContextualPromptLearnerUP, self).__init__()
        #--> parameters
        self.type = cls_type.lower().split('_')[0]
        self.n_cls = n_cls
        self.n_tokens = n_tokens
        self.n_kv = len(knowledge_version)
        #--> general settings
        self.use_descriptor = use_descriptor # use per-class descriptors to replace class descriptions
        self.cntn = use_cntn # integrate continous embeddings
        self.cntn_split = cntn_split # use different knowledge embedding \
        # when constructing `cntn_embeds` for different knowledge versions
        self.uni_mlp = uni_mlp # use a single mlp to project cntn emb onto all learnable token positions
        self.disc = use_disc # integrate class description texts in natural language
        # load the precompted embeddings
        #--> Initialize semantic-rich category descriptions / labels
        self.updrs_ke_dir = f'./data/ke_{self.type}'
        assert osp.isdir(self.updrs_ke_dir)
        
        
        assert len(knowledge_version)>0 or self.use_descriptor, "No knowledge is specified."
        if self.use_descriptor:
            assert self.disc or self.cntn_split, "Descriptor is used but disc is not enabled."
            ENT_base = np.load(osp.join(self.updrs_ke_dir, 'all.npy'))
            cntn_embeds = []
            cls_disc = []
            # load per-class descriptors
            for idc in range(self.n_cls):
                disc_file = f"descriptor_{idc}.txt"
                disc_file = osp.join(self.updrs_ke_dir, disc_file)
                cls_disc.append(self.load_disc_knowledge(disc_file))
                if self.cntn:
                    if self.cntn_split: # when use descriptor, cntn_split False means use overall embedding in cntn
                        # load pre-computed per-class knowledge embeddings
                        ke_path = osp.join(self.updrs_ke_dir, f'descriptor_{idc}.npy')
                        ENT = np.load(ke_path)
                        cntn_embeds.append(torch.from_numpy(ENT).float())
                    else:
                        cntn_embeds.append(torch.from_numpy(ENT_base[idc]).float().unsqueeze(0).expand(len(cls_disc[idc]), -1))
        else:
            cntn_embeds = torch.empty(n_cls, 0, inp_dim)
            cls_disc = [[] for _ in range(self.n_cls)]
            for kv in knowledge_version:
                ke_path = osp.join(self.updrs_ke_dir, f'EntityEmb_{kv}.npy')
                ke_v0_path = osp.join(self.updrs_ke_dir, f'EntityEmb_v0.npy')
                disc_file = f"simQdesc_{kv}.txt"
                disc_file = osp.join(self.updrs_ke_dir, disc_file)
                if self.cntn:
                    if self.cntn_split:
                        # load pre-computed per-class knowledge embeddings
                        ENT = np.load(ke_path)
                        cntn_embeds = torch.cat([cntn_embeds, torch.from_numpy(ENT).float().unsqueeze(1)], dim=1)
                    else:
                        cntn_embeds = torch.cat([cntn_embeds, torch.from_numpy(np.load(ke_v0_path)).float().unsqueeze(1)], dim=1)
                if self.disc:
                    # load the class descriptions
                    description = self.load_disc_knowledge(disc_file)
                    for idc, desc in enumerate(description):
                        cls_disc[idc].append(desc)
                
    
        if self.cntn:
            if self.uni_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(inp_dim, emb_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(emb_dim, out_dim),
                )
            else:
                self.projector = nn.ModuleList()
                for _ in range(n_tokens):
                    self.projector[-1].append(nn.Sequential(            
                    nn.Linear(inp_dim, emb_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(emb_dim, out_dim),
                ))
        else:
            self.projector = None
        # =====> register the buffers for the embeddings
        if len(cntn_embeds)>0:
            # register the embeddings into buffer
            self.register_buffer('cntn_embeds', cntn_embeds)
        if len(cls_disc)>0:
            self.cls_disc = cls_disc


        
    @staticmethod
    def load_disc_knowledge(disc_file):
        cls_disc = []
        with open(disc_file, 'r') as f:
            for _, line in enumerate(f):
                cls_disc.append(line.strip())        
        
        return cls_disc
        
    def forward(self, ctx_prompt,):
        """
        Args:
            ctx_prompt: (n_cls, N, F), the learnable parameters for the prompts
        """
        if self.cntn:
            # project the embeddings per knowledge version
            prompts = []
            embeds = []
            if self.uni_mlp:
                embeds = self.projector(self.cntn_embeds).unsqueeze(2) # [n_cls, n_kv, 1, out_dim]
                embeds = embeds.expand(-1, -1, self.n_tokens, -1) # [n_cls, n_kv, N, out_dim]
            else:
                for i in range(self.n_tokens):
                    embeds.append(self.projector[i](self.cntn_embeds)).unsqueeze(2) # [n_cls, n_kv, 1, out_dim]
                embeds = torch.stack(embeds, dim=2) # [n_cls, n_kv, N, out_dim]

            prompts = ctx_prompt.unsqueeze(1) + embeds # [n_cls, n_kv, N, out_dim]
        else:
            prompts = ctx_prompt

        return prompts # [n_cls, n_kv, N, out_dim]

if __name__ == '__main__':
    pass
