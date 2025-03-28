# calculate focal loss + ordinal loss
# to combat class imbalance and intrinsic class order
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

def categorical_ordinal_focal_weight(gamma=2., alpha=.25, beta=0.0, eps=1e-7, scale=1.0):
    """
    Categorical focal loss defined in https://arxiv.org/pdf/2007.08920v1.pdf. 
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
      beta -- weighting factor for ordinal component
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/2007.08920v1.pdf
        Focal loss implementation: https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    Usage:
     model.compile(loss=[categorical_ordinal_focal_loss(gamma=2, alpha=.25, beta=0.2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_ordinal_focal_weight_fixed(y_pred, y_true):
        """
        :param y_pred: A tensor resulting from a softmax
        :param y_true: A tensor of the same shape as y_pred
        :return: Output tensor.
        """
        # y_pred /= torch.sum(y_pred, dim=-1, keepdims=True)
        if y_true.dim() == y_pred.dim() - 1:
            y_true = F.one_hot(y_true, num_classes=y_pred.shape[-1])
        # y_pred = torch.clamp(y_pred, eps, 1. - eps)
        y_pred = y_pred.softmax(dim=-1)
        # cross_entropy = -y_true * torch.log(y_pred)
        ordinal_dist = torch.abs(torch.argmax(y_true, dim=-1) - torch.argmax(y_pred, dim=-1))
        weights = (ordinal_dist/(y_pred.shape[-1] - 1)).float()
        focal_loss = alpha * torch.pow(1 - y_pred, gamma)
        classes = y_pred.shape[-1]
        weights_expanded = (weights.unsqueeze(-1)).expand(-1, classes)
        combined_weights = (beta * weights_expanded + focal_loss ) * y_true # OF Loss
        # combined_weights = weights_expanded * y_true # Ordinal loss only
        return combined_weights.sum(-1)*scale

    return categorical_ordinal_focal_weight_fixed

def cosine_similarity_nce_loss(temperature=10.,weight=1.0, reduction='mean'):
    """
    Cosine similarity NCE loss
    Parameters:
      temperature -- temperature parameter for scaling logits
    Default value:
      temperature -- 10.0
    References:
        Official paper: https://arxiv.org/pdf/2007.08920v1.pdf
        cosimNCE loss implementation:
    """
    def cosim_nce_loss(sim_mat:torch.Tensor):
      """
      :param sim_mat: A tensor resulting from a cosine similarity matrix
      """
      # calculate per-class exponential ratio
      assert sim_mat.shape[-1] == sim_mat.shape[-2]
      nomin = torch.exp(torch.diagonal(sim_mat, dim1=-2, dim2=-1) / temperature)
      denomin = torch.exp(sim_mat.sum(-1) / temperature)
      ratio = - torch.log(nomin / denomin)
      if reduction == 'mean':
        ratio = ratio.mean()
      elif reduction == 'sum':
        ratio = ratio.sum()
      elif reduction == 'none':
        pass
      else:
        raise NotImplementedError
      
      return weight*ratio
    
    return cosim_nce_loss

def InfoNCE_loss(n_cls, temperature=0.1,weight=1.0, reduction='mean', eps=1e-7, focal=False,):
    """
    InfoNCE loss
    Parameters:
      temperature -- temperature parameter for scaling logits
    Default value:
      temperature -- 10.0
    References:
        Official paper: https://arxiv.org/pdf/2007.08920v1.pdf
        cosimNCE loss implementation:
    """
    # for focal weight
    alpha = 0.25
    gamma = 2.0
    def info_nce_loss(y_pred, y_true, y=None):
      """
        :param y_pred: A tensor resulting from a softmax
        :param y_true: A tensor of the same shape as y_pred
        :return: Output tensor.
      """
      # calculate per-class exponential ratio
      # convert y_true to one-hot
      _y_true = torch.zeros(y_pred.shape[0], n_cls, device=y_true.device)
      _y_true.scatter_(1, y_true.unsqueeze(-1), 1)
      if y is not None: 
        # convert y to one-hot
        _y = torch.zeros(y.shape[0], n_cls, device=y.device)
        _y.scatter_(1, y.unsqueeze(-1), 1)
        _y_true = _y_true @ _y.t()
        
      _y_true = _y_true.bool()
      # construct positive/negative pairs
      pair_pos = y_pred[_y_true].clone() # N
      pair_neg = y_pred # N * C
      
      prob_pos = torch.exp(pair_pos / temperature)
      prob_neg = torch.exp(pair_neg / temperature)
  
      if focal:
        # weight the loss with the focal loss
        focal_weight = alpha * torch.pow(1-prob_pos/prob_neg.sum(-1), gamma)
        prob_pos *= focal_weight
      
      loss = - torch.log(prob_pos.sum() / (prob_neg.sum() + eps))
      
      if reduction == 'mean':
        loss = loss.mean()
      elif reduction == 'sum':
        loss = loss.sum()
      elif reduction == 'none':
        pass
      else:
        raise NotImplementedError
      
      return weight*loss
    
    return info_nce_loss

def sigmoid_focal_loss(alpha=0.25, gamma=2.0, use_focal=False, 
                       scale=1.0, eps=1e-7, reduction='none'):
   
   def sigmoid_loss_fixed(y_pred, y_true):
      """
      :param y_pred: A tensor resulting from a softmax
      :param y_true: A tensor of the same shape as y_pred
      :return: Output tensor.
      """
      # construct [-1,1] ground-truth labels
      if y_true.dim() == y_pred.dim() - 1:
         y_true = F.one_hot(y_true, num_classes=y_pred.shape[-1])
      y_true = y_true.float()

      ce_loss = -F.logsigmoid((y_true*2 - 1.) * y_pred)  
      if use_focal:
        proba = torch.sigmoid(y_pred)
        # ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        proba_t = proba * y_true + (1. - proba) * (1. - y_true) # formula for binary classification
        alpha_t = alpha * y_true + (1. - alpha) * (1. - y_true)
        loss = alpha_t * (1. - proba_t) ** gamma * ce_loss
      else:
         loss = ce_loss
      loss = loss.sum(-1)
      # Check reduction option and return loss accordingly
      if reduction == "none":
          pass
      elif reduction == "mean":
          loss = loss.mean()
      elif reduction == "sum":
          loss = loss.sum()
      else:
          raise ValueError(
              f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
          )

      return loss * scale
   
   return sigmoid_loss_fixed