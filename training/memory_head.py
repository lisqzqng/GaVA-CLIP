# preprocess the memory data before feed into CLIP text encoder
import torch
torch.manual_seed(0)
import torch.nn as nn

import numpy as np

from VitaCLIP_text_encoder import tokenize

class MemoryPromptLearner(nn.Module):
    """
    Project the memory embedding from RoBERTa(KEPLER) to CLIP text token embeddings
    """
    def __init__(self, 
                 text_model, 
                 num_class,
                 splitMLP=True, # class-wise MLP
                 sublen=4, 
                 context_length=77, 
                 inp_dim=768, 
                 out_dim=512, 
                 batch_size=32,
                ):
        super(MemoryPromptLearner, self).__init__()

        # -----> parameters
        self.sublen = sublen
        self.context_length = context_length
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.splitMLP = splitMLP

        if self.splitMLP:
            self.mem_projet = nn.ModuleList([])
            for _ in range(num_class):
                self.mem_projet.append(
                    nn.Sequential(
                        nn.Linear(inp_dim, out_dim//2),
                        nn.Tanh(),
                        nn.Linear(out_dim//2, out_dim),            
                ))
        else:
            self.mem_projet = nn.Sequential(
                nn.Linear(inp_dim, out_dim//2),
                nn.Tanh(),
                nn.Linear(out_dim//2, out_dim),            
            )

        self.base_token = tokenize('X is X').expand(batch_size*sublen, -1)
        self.text_model = text_model
        with torch.no_grad(): #B, 3, 77, 512
            base_embedding = text_model.token_embedding(self.base_token)
            self.register_buffer("preEmbed", base_embedding[:,:1,:].clone())
            self.register_buffer("isEmbed", base_embedding[:,2:3,:].clone())
            self.register_buffer("postEmbed", base_embedding[:,4:,:].clone())

    def forward(self, m, v):
        "input the memory featuures and the value features"
        m = m.reshape(-1, self.inp_dim)
        if self.splitMLP:
            mem_features = []
            for mproj in self.mem_projet:
                mem = mproj(m)
                # replace the 'X' in `self.base_embedding` with projected memory feature & value
                mem_tok_features = torch.concat([self.preEmbed, mem.reshape(-1,1,self.out_dim), self.isEmbed, \
                                                v.reshape(-1,1,self.out_dim), self.postEmbed], dim=-2)
                mem_features.append(self.text_model(mem_tok_features.float(), self.base_token).reshape(-1, self.sublen, self.out_dim).mean(1).unsqueeze(0))
            mem_features = torch.concat(mem_features, dim=0)
        else:
            mem = self.mem_projet(m) # B*3, 512
            # B, 3, 77, 512
            # replace the 'X' in `self.base_embedding` with projected memory feature & value
            mem_tok_features = torch.concat([self.preEmbed, mem.reshape(-1,1,self.out_dim), self.isEmbed, \
                                            v.reshape(-1,1,self.out_dim), self.postEmbed], dim=-2)
            mem_features = self.text_model(mem_tok_features.float(), self.base_token).reshape(-1, self.sublen, self.out_dim).mean(1)

        return mem_features # the gait parameter set embedding