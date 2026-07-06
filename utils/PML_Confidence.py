import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mllt.models.losses.cross_entropy_loss import binary_cross_entropy
class PML_Confidence(nn.Module):
    def __init__(self, train_givenY, dataset, mu_range, tau=1, beta=1, m=0.9, phi=0.9, total_epoch=10, mu=0.9):
        super().__init__()
        self.mu = mu
        self.feat_mean = None
        self.co_occurrence = None
        self.xi = torch.zeros(train_givenY.shape[1], train_givenY.shape[1])
        self.m = m
        self.phi = phi
        self.beta = beta
        self.mu_range = mu_range
        self.total_epoch = total_epoch
        self.tau = tau
        self.confidence = train_givenY.float()
        self.class_1 = (train_givenY.float().sum(0) / train_givenY.float().sum(0).sum(0)).unsqueeze(0)
        self.class_0 = ((1-train_givenY.float()).sum(0) / (1-train_givenY.float()).sum(0).sum(0)).unsqueeze(0)
        self.cls_criterion = binary_cross_entropy
        self.reduction = 'mean'
        self.dataset = dataset
        self.freq_inv = torch.ones(self.class_1.shape).cuda() / self.class_1
        self.init_bias = 0.05
        if self.dataset == 'voc-lt':
            ################## 5
            self.neg_scale = 5
            # self.neg_scale = 1
            self.map_alpha = 0.1
            self.map_beta = 10.0
            self.map_gamma = 0.3
        elif self.dataset == 'coco-lt':
            ################## 5
            self.neg_scale = 2
            # self.neg_scale = 1
            self.map_alpha = 0.1
            self.map_beta = 10.0
            self.map_gamma = 0.2
        elif self.dataset == 'cub200':
            ################## 5
            self.neg_scale = 5
            # self.neg_scale = 1
            self.map_alpha = 0.1
            self.map_beta = 10.0
            self.map_gamma = 0.3
    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach() / repeat_rate
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def forward(self, logits, feature, partial_label, index, model, epoch, targets=None):

        if self.feat_mean is None:
            self.feat_mean = (1 - self.m) * feature.detach().mean(0)
        else:
            self.feat_mean = self.m * self.feat_mean + (1 - self.m) * feature.detach().mean(0)

        # smooth average co-occurrence matrix
        prediction_init = F.sigmoid(logits.detach())
        pp = prediction_init.unsqueeze(2) * prediction_init.unsqueeze(1)  # p_l p_k
        v = prediction_init * (1.0 - prediction_init)  # p_l(1-p_l), shape (B,K)
        vv = v.unsqueeze(2) * v.unsqueeze(1)  # p_l(1-p_l) p_k(1-p_k), (B,K,K)
        tilde = pp + self.xi.cuda().unsqueeze(0) * vv  # broadcast xi to (B,K,K)
        A = tilde.mean(dim=0)
        if self.co_occurrence is None:
            self.co_occurrence = A
        else:
            self.co_occurrence = self.phi * self.co_occurrence + (1 - self.phi) * A

        bias = model.module.head(self.feat_mean.unsqueeze(0)).detach()
        bias_sigmoid = F.sigmoid(bias)

        # compute \xi
        pi_outer = bias_sigmoid.squeeze(0)[:, None] * bias_sigmoid.squeeze(0)[:, None]
        numer = A - pi_outer
        var = bias_sigmoid.squeeze(0) * (1.0 - bias_sigmoid.squeeze(0))
        denom = torch.sqrt(var[:, None] * var[None, :] + 1e-12) + 1e-12
        self.xi = numer / denom

        # label-condition dependency term
        pi_l = bias_sigmoid.squeeze(0)[:, None]
        pi_k = bias_sigmoid.squeeze(0)[None, :]
        num = pi_k * (pi_l - self.co_occurrence.t()).clamp(min=1e-12)
        den = (pi_l * (pi_k - self.co_occurrence) + 1e-12).clamp(min=1e-12)
        gamma_hat = torch.log(num / den).fill_diagonal_(0.0)
        dep_term = self.beta * (prediction_init @ gamma_hat.t())

        ################## 6
        logits_output = logits - torch.log((bias_sigmoid / (1 - bias_sigmoid)) ** self.tau + 1e-12) + dep_term

        # batch_LA
        '''
        class_batch_1 = (partial_label.sum(0) / partial_label.sum(0).sum(0)).unsqueeze(0)
        class_batch_0 = ((1-partial_label).sum(0) / (1-partial_label).sum(0).sum(0)).unsqueeze(0)
        logits_output = logits - torch.log((class_batch_1 / class_batch_0) ** self.tau + 1e-12)
        '''
        # total_MLA
        # logits_output = logits - torch.log((self.class_1 / self.class_0) ** self.tau + 1e-12)

        # total_LA
        # logits_output = logits - torch.log(self.class_1 ** self.tau + 1e-12)

        prediction = F.sigmoid(logits_output.detach())
        prediction_partial = prediction * partial_label
        prediction_adj = prediction_partial / prediction_partial.sum(dim=1, keepdim=True)

        self.confidence_move_update(prediction_adj, index, epoch)
        # self.confidence_move_update(prediction_partial, index, epoch)

        q_soft = self.confidence[index, :].clone().detach()
        self.init_bias = torch.log((bias_sigmoid / (1 - bias_sigmoid)) ** self.tau + 1e-12)

        ################## 4
        logits += self.init_bias
        logits = logits - dep_term
        weight = self.rebalance_weight(partial_label)
        cls_score = logits * (1 - partial_label) * self.neg_scale + logits * partial_label
        weight = weight / self.neg_scale * (1 - partial_label) + weight * partial_label * q_soft

        loss = self.cls_criterion(cls_score, partial_label.float(), weight, reduction=self.reduction)

        return loss

    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, batch_index, epoch):
        [mu_s, mu_e] = [float(item) for item in self.mu_range.split(',')]
        self.mu = mu_s + (mu_e - mu_s) * min(epoch / self.total_epoch, 1)
        self.confidence[batch_index, :] = self.confidence[batch_index, :] * self.mu + temp_un_conf * (1 - self.mu)
        return None


