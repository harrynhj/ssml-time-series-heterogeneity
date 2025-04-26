import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.utils.checkpoint as cp
import torch.autograd as autograd
import numpy as np
import copy

from rnnmodel import LSTM
from copy import deepcopy
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Meta(nn.Module):
    def __init__(self, args, config, device):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.train_way
        self.k_spt = args.train_shot
        self.k_qry = args.train_query
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.config = config
        self.device = device
        self.net = LSTM(*config)
        self.domain_net = nn.Linear(128, 1)
        self.label0 = torch.zeros(8, 1).cuda()
        self.label1 = torch.ones(8, 1).cuda()
        self.tau = args.tau
        #self.tau = -999999

    def single_task_loss(self, data_in):
        # Unpack data for clarity
        support_x = data_in[0]
        support_y = data_in[1]
        meta_x = data_in[2] # Query set for meta-update
        meta_y = data_in[3] # Query labels for meta-update
        support_xu = data_in[4] # Unlabeled support set
        support_xaug = data_in[5] # Augmented unlabeled support set
        meta_xu = data_in[6] # Unlabeled query set
        meta_xaug = data_in[7] # Augmented unlabeled query set

        meta_loss_list = [] # Renamed to avoid conflict with 'loss' variable

        # --- Step 0: Initial Forward Pass on Labeled Support Data ---
        # Ensure self.net returns two values (output, latent_state)
        # Fix 1: Receive both output and latent state from the first call
        out_spt, latent_spt = self.net(support_x)

        # Calculate domain loss for labeled support data using its latent state
        latent_reversed_spt = ReverseLayerF.apply(latent_spt, 0.5)
        domain_spt = self.domain_net(latent_reversed_spt)
        dloss0 = F.binary_cross_entropy_with_logits(domain_spt, self.label0)

        # --- Step 0: Predictions for Unlabeled Support Data ---
        # Get predictions (logits) and latent state for unlabeled and augmented unlabeled
        # We need the latent state from augmented data for its domain loss (dloss1)
        pseudo_logits, _ = self.net(support_xu) # Ignore latent state for pseudo label generation
        pseudo_logits = pseudo_logits.double()
        pred_logits, latent_aug = self.net(support_xaug) # Get logits and latent for augmented

        # Calculate domain loss for augmented unlabeled support data using its latent state
        # Fix 3 (Logic Correction): Use latent_aug for dloss1, not latent_spt
        latent_reversed_aug = ReverseLayerF.apply(latent_aug, 0.5)
        domain_aug = self.domain_net(latent_reversed_aug)
        dloss1 = F.binary_cross_entropy_with_logits(domain_aug, self.label1)

        # --- Step 0: Calculate Loss for Initial Parameters ---
        # Create mask based on unlabeled logits confidence
        mask = torch.gt(torch.abs(pseudo_logits), self.tau)
        # Create boolean pseudo-labels
        pseudo_labels_bool = torch.gt(pseudo_logits, 0)
        # Fix 2: Convert boolean pseudo-labels to float for loss calculation
        pseudo_labels_float = pseudo_labels_bool.float()

        # Combine Labeled loss + Pseudo-labeled loss + Domain losses
        loss = F.binary_cross_entropy_with_logits(out_spt, support_y) + \
               F.binary_cross_entropy_with_logits(pred_logits * mask, pseudo_labels_float * mask) + \
               dloss0 + dloss1 # Assuming dloss1 is intended here

        # Calculate gradients for the first update step
        self.net.zero_grad() # Clear potential gradients if any
        grad = autograd.grad(loss, self.net.parameters(), create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        # --- MAML Inner Loop (Steps 1 to update_step-1) ---
        # Calculate loss on query set using adapted parameters (fast_weights) for meta-update
        # This first query calculation is for the loss *after* 1 update step
        out_qry, _ = self.net(meta_x, vars=fast_weights, train=True) # Ignore latent
        pseudo_qry_logits, _ = self.net(meta_xu, vars=fast_weights, train=True) # Ignore latent
        pseudo_qry_logits = pseudo_qry_logits.double()
        pred_qry_logits, _ = self.net(meta_xaug, vars=fast_weights, train=True) # Ignore latent

        mask_qry = torch.gt(torch.abs(pseudo_qry_logits), self.tau)
        pseudo_qry_labels_bool = torch.gt(pseudo_qry_logits, 0)
        pseudo_qry_labels_float = pseudo_qry_labels_bool.float() # Convert to float

        meta_loss_list.append(
            F.binary_cross_entropy_with_logits(out_qry, meta_y) + \
            F.binary_cross_entropy_with_logits(pred_qry_logits * mask_qry, pseudo_qry_labels_float * mask_qry)
        )

        # Inner loop for subsequent update steps (if update_step > 1)
        for k in range(1, self.update_step):
            # Calculate loss on support set using current fast_weights
            out_spt_k, latent_spt_k = self.net(support_x, vars=fast_weights, train=True)

            latent_reversed_spt_k = ReverseLayerF.apply(latent_spt_k, 0.5)
            domain_spt_k = self.domain_net(latent_reversed_spt_k)
            dloss0_k = F.binary_cross_entropy_with_logits(domain_spt_k, self.label0)

            pseudo_logits_k, _ = self.net(support_xu, vars=fast_weights, train=True)
            pseudo_logits_k = pseudo_logits_k.double()
            pred_logits_k, latent_aug_k = self.net(support_xaug, vars=fast_weights, train=True)

            latent_reversed_aug_k = ReverseLayerF.apply(latent_aug_k, 0.5)
            domain_aug_k = self.domain_net(latent_reversed_aug_k)
            dloss1_k = F.binary_cross_entropy_with_logits(domain_aug_k, self.label1)

            mask_k = torch.gt(torch.abs(pseudo_logits_k), self.tau)
            pseudo_labels_bool_k = torch.gt(pseudo_logits_k, 0)
            pseudo_labels_float_k = pseudo_labels_bool_k.float() # Convert to float

            loss_k = F.binary_cross_entropy_with_logits(out_spt_k, support_y) + \
                     F.binary_cross_entropy_with_logits(pred_logits_k * mask_k, pseudo_labels_float_k * mask_k) + \
                     dloss0_k + dloss1_k

            # Calculate gradients for the next step
            self.net.zero_grad() # Make sure gradients from previous use of fast_weights are cleared
            grad_k = autograd.grad(loss_k, fast_weights, create_graph=True)
            # Update fast_weights
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_k, fast_weights)))

            # --- Calculate Loss on Query Set for Meta Update ---
            # Evaluate the updated fast_weights on the query set
            out_qry_k, _ = self.net(meta_x, vars=fast_weights, train=True)
            pseudo_qry_logits_k, _ = self.net(meta_xu, vars=fast_weights, train=True)
            pseudo_qry_logits_k = pseudo_qry_logits_k.double()
            pred_qry_logits_k, _ = self.net(meta_xaug, vars=fast_weights, train=True)

            mask_qry_k = torch.gt(torch.abs(pseudo_qry_logits_k), self.tau)
            pseudo_qry_labels_bool_k = torch.gt(pseudo_qry_logits_k, 0)
            pseudo_qry_labels_float_k = pseudo_qry_labels_bool_k.float() # Convert to float

            meta_loss_list.append(
                F.binary_cross_entropy_with_logits(out_qry_k, meta_y) + \
                F.binary_cross_entropy_with_logits(pred_qry_logits_k * mask_qry_k, pseudo_qry_labels_float_k * mask_qry_k)
            )

        # Return the list of losses calculated on the query set after each update step
        return meta_loss_list


    def forward(self, data, meta_train=True, fast_weights=None):
        support_x = data[0]
        support_y = data[1]
        meta_x = data[2]
        meta_y = data[3]
        support_xu = data[4]
        support_xaug = data[5]
        meta_xu = data[6]
        meta_xaug = data[7]
        if (meta_train):
            """
            :param support_x:   [b, setsz, c_, h, w]
            :param support_y:   [b, setsz]
            :param meta_x:      [b, setsz, c_, h, w]
            :param meta_y:      [b, setsz]
            """
            # assert(len(support_x.shape) == 5)
            # task_num_now = support_x.size(0)
            task_num_now = len(support_x)
            n_task_meta_loss = list(map(self.single_task_loss,
                                        zip(support_x, support_y, meta_x, meta_y, support_xu, support_xaug, meta_xu,
                                            meta_xaug)))
            re = n_task_meta_loss[0][-1].view(1, 1)
            for i in range(1, task_num_now):
                re = torch.cat([re, n_task_meta_loss[i][-1].view(1, 1)], dim=0)
            return re
        elif fast_weights is None:
            """
            :param support_x:   [b, setsz,   c_, h, w]
            :param support_y:   [b, setsz  ]
            :param qx:          [b, querysz, c_, h, w]
            :param qy:          [b, querysz]
            :return:            [b, acc_dim]
            """
            fast_weights = list(self.net.parameters())
            for _ in range(self.update_step_test):
                # out = self.net(support_x, vars = fast_weights, train=True)
                # pseudo = self.net(support_xu, vars=fast_weights, train=True)
                # pred = self.net(support_xaug, vars=fast_weights, train=True)
                # loss = F.multilabel_soft_margin_loss(out, support_y) + F.multilabel_soft_margin_loss(pseudo, pred)
                out, _ = self.net(support_x, vars=fast_weights, train=True)
                loss = F.binary_cross_entropy_with_logits(out, support_y)
                self.net.zero_grad()
                grad = autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            return fast_weights
        else:
            out_q, _ = self.net(meta_x, vars=fast_weights)
            return out_q


def main():
    pass


if __name__ == '__main__':
    main()
