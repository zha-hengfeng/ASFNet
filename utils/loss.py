import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''

    def __init__(self, weight=None, ignore_label=255, aux=False):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super().__init__()
        self.aux = aux
        # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)
        self.loss_2 = nn.NLLLoss(weight, ignore_index=ignore_label)
        self.loss_3 = nn.NLLLoss(weight, ignore_index=ignore_label)
        self.loss_4 = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, preds, target):
        output = preds[0]             # no-aux
        loss = self.loss(F.log_softmax(output, 1), target)

        if self.aux and len(preds) > 1:
            pred, out_2, out_3, out_4 = preds
            # loss_34 = self.criterion_34(b34, target)
            loss_2 = self.loss_2(out_2, target)
            loss_3 = self.loss_3(out_3, target)
            loss_4 = self.loss_4(out_4, target)

            # loss = loss + 0.05 * loss_2 + 0.15 * loss_3 + 0.4 * loss_4
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False, aux=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.aux = aux
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
            if aux:
                self.criterion_2 = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                           weight=weight,
                                                           ignore_index=ignore_label)
                self.criterion_3 = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                           weight=weight,
                                                           ignore_index=ignore_label)
                self.criterion_4 = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                             weight=weight,
                                                             ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, preds, target):

        pred = preds[0]
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # logger.info('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            #prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~ valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        #target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.masked_fill_(~ valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        loss = self.criterion(pred, target)

        if self.aux and len(preds) > 1:
            pred, out_2, out_3, out_4 = preds
            # loss_34 = self.criterion_34(b34, target)
            loss_2 = self.criterion_2(out_2, target)  # use for stage-3
            loss_3 = self.criterion_3(out_3, target)
            loss_4 = self.criterion_4(out_4, target)

            loss = loss + 0.10*loss_2 + 0.25*loss_3 + 0.40*loss_4     # use for stage-3
            # loss = 0.4*loss + 0.20*loss_3 + 0.40*loss_4

        return loss
