'''
Description:
Author: Yi Yang
Date: 2023-03-15 14:29
'''

import torch.nn.functional as F
import torch
import utils as utils
import torch.nn as nn
from einops import rearrange, repeat

class ChannelAttentionMPL(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionMPL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, 32, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(32, in_planes,1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # self.sigmoid = nn.Softmax(dim=1)


    def forward(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg)))
        max_out = self.fc2(self.relu1(self.fc1(max)))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        out = x + attention * x
        return out, attention





class EmotionAttention(nn.Module):
    def __init__(self, in_planes=[5,62], ratio=16):
        super(EmotionAttention, self).__init__()
        self.frequency_mix = nn.Sequential(
            # FeedForward(num_patch, token_dim, dropout),
            # nn.LayerNorm(62),
            # nn.BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ChannelAttentionMPL(in_planes=in_planes[0])
        )

        self.channel_mix = nn.Sequential(
            # nn.LayerNorm(5),
            ChannelAttentionMPL(in_planes=in_planes[1]),
        )
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.view(x.size(0),5, 62)
        residual = x
        # out, att_weight2 = self.channel_mix(x.transpose(1, 2))
        out, att_weight1 = self.frequency_mix(x)
        out, att_weight2 = self.channel_mix(out.transpose(1,2))
        out = out.transpose(1,2)
        out = out.view(out.size(0),-1)
        out = out + residual.view(residual.size(0),-1)
        out = self.module(out)
        return out, [att_weight1,att_weight2]



def pretrained_CFE(pretrained=False):
    model = EmotionAttention()
    if pretrained:
        pass
    return model


class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(

            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

        )

    def forward(self, x):
        x = self.module(x)
        return x


def domain_discrepancy(out1, out2, loss_type):
    def huber_loss(e, d=1):
        t =torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'Huber':
        loss = huber_loss(diff)
    else:
        loss = torch.mean(diff*diff)
    return loss


class MSMDAERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)

        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

        self.weight_d = 0.3
        self.src_ca_last1 = [1. for _ in range(number_of_source)]
        self.tar_ca_last1 = [1.]
        self.src_ca_last2 = [1. for _ in range(number_of_source)]
        self.tar_ca_last2 = [1.]
        self.domain_loss_type = 'L1'
        self.number_of_source = number_of_source
        self.id = id


    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0

        data_src_DSFE = []
        data_tgt_DSFE = []
        att_loss = 0
        cls_loss = 0
        tcls_loss = 0
        if self.training == True:
            # common feature extractor
            data_src_CFE,att_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE,att_tgt_CFE = self.sharedNet(data_tgt)


            data_src_CFE = torch.chunk(data_src_CFE,number_of_source,0)
            label_src = torch.chunk(label_src,number_of_source,0)
            att_src_CFE_last = torch.chunk(att_src_CFE[-1], number_of_source, 0)
            att_src_CFE_last2 = torch.chunk(att_src_CFE[-2], number_of_source, 0)
            pred_tgt =[]
            with torch.no_grad():
                for i in range(number_of_source):
                    DSFE_name = 'self.DSFE' + str(i)
                    data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                    DSC_name = 'self.cls_fc_DSC' + str(i)
                    pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                    pred_tgt_i = F.softmax(pred_tgt_i, dim=1)
                    pred_tgt.append(pred_tgt_i.unsqueeze(1))
                pred_tgt = torch.cat(pred_tgt,dim=1)
                pred_tgt_w = pred_tgt.mean(1)
                max_prob, label_tgt = pred_tgt_w.max(1)  # (B)
                label_tgt_mask = (max_prob >= 0.95).float()

            for i in range(number_of_source):
                # Each domian specific feature extractor
                # to extract the domain specific feature of target data

                DSFE_name = 'self.DSFE' + str(i)
                data_src_DSFE_i = eval(DSFE_name)(data_src_CFE[i])
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_src_DSFE_i)
                data_src_DSFE.append(data_tgt_DSFE_i)

                mmd_loss += utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i)

                # Each domian specific classifier

                DSC_name = 'self.cls_fc_DSC' + str(i)
                pred_src_i = eval(DSC_name)(data_src_DSFE_i)
                cls_loss += F.nll_loss(F.log_softmax(
                    pred_src_i, dim=1), label_src[i].squeeze())


                pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                tcls_loss_i = F.nll_loss(F.log_softmax(
                    pred_tgt_i, dim=1), label_tgt,reduction='none')
                tcls_loss += (tcls_loss_i * label_tgt_mask).mean()

                ema_alpha = 0.8

                mean_tar_ca1 = self.tar_ca_last1[0] * ema_alpha + (1. - ema_alpha) * torch.mean(att_tgt_CFE[-1], 0)
                self.tar_ca_last1[0] = mean_tar_ca1.detach()

                mean_src_ca1 = self.src_ca_last1[i] * ema_alpha + (1. - ema_alpha) * torch.mean(att_src_CFE_last[i], 0)
                att_loss += self.weight_d / self.number_of_source * domain_discrepancy(mean_src_ca1, mean_tar_ca1,
                                                                                       self.domain_loss_type)

                mean_tar_ca2 = self.tar_ca_last2[0] * ema_alpha + (1. - ema_alpha) * torch.mean(att_tgt_CFE[-2], 0)
                self.tar_ca_last2[0] = mean_tar_ca2.detach()

                mean_src_ca2 = self.src_ca_last2[i] * ema_alpha + (1. - ema_alpha) * torch.mean(att_src_CFE_last2[i], 0)
                att_loss += self.weight_d / self.number_of_source * domain_discrepancy(mean_src_ca2, mean_tar_ca2,
                                                               self.domain_loss_type)

                self.src_ca_last1[i] = mean_src_ca1.detach()
                self.src_ca_last2[i] = mean_src_ca2.detach()


            return cls_loss+0.2*tcls_loss, mmd_loss, att_loss

        else:
            data_CFE,_= self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred



