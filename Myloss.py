# encoding:utf-8
#
# commet by zhangqi 2020,04.12
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class yoloLoss(nn.Module):
    '''
    传说中Yolo十分复杂的损失函数
    '''

    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''

        # 0维的size，即框的个数
        N = box1.size(0)
        M = box2.size(0)

        # 用来生成inter部分的面积
        # 获取候选框左上角的x、y坐标的最大值
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # 获取候选框右下角的x、y坐标的最小值
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # 此时只剩长框的长、宽
        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0，两个box没有重叠区域
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        # 利用坐标计算候选框的面积
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        """

        # batchsize的大小，用来归一化
        # N = pred_tensor.size()[0]
        N = pred_tensor.size(0)
        # 候选框内有目标
        coo_mask = target_tensor[:, :, :, 4] > 0
        # 候选框内无目标
        noo_mask = target_tensor[:, :, :, 4] == 0
        # 将coo_mask、noo_mask的维度转化为target_tensor
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # 预测下的tensor
        # 将coo_mask条件下的pre_tensor转化为：?*30的形式
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        # 调用contiguous()会使得原tensor被强制拷贝一份，避免原始数据被修改
        # index：0~9表示x1,y1,w1,h1,c1,x2,y2,w2,h2,c2
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        # 类别信息的index从10开始
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]

        # 真实下的tensor
        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # compute not contain obj loss
        # 将pre_tensor、target_tensor转化为：?*30的形式
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        # 这里为GPU张量类型，因为未使用GPU训练，所以进行修改
        # noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask = torch.ByteTensor(noo_pred.size())
        # 对noo_pred_mask进行初始化
        noo_pred_mask.zero_()
        # 通过对置信度数值赋1，为后面提取置信度提供方便
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        # 提取置信度
        noo_pred_c = noo_pred[noo_pred_mask]  # noo_pred只需要计算c的损失：size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        # 计算候选框内没有目标时的损失
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # compute contain obj loss
        # 这里为GPU张量类型，因为未使用GPU训练，所以进行修改
        # coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask = torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        # 这里为GPU张量类型，因为未使用GPU训练，所以进行修改
        # coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask = torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()

        # box_target_iou = torch.zeros(box_target.size()).cuda()
        box_target_iou = torch.zeros(box_target.size())

        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            box1 = box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # 获取box1左上角的坐标
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            # 获取box1右下角的坐标
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            # 获取box2左上角的坐标
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            # 获取box2右下角的坐标
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            # 计算iou
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            # 获取最大iou的index
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()
            # 负责检测坐标的box响应
            coo_response_mask[i + max_index] = 1
            # 不负责检测坐标的bbox的响应
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        # 包含目标的bbox的回归误差，第3项
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        # 坐标回归误差，第1、2项
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)

        # 2.not response loss
        # 不包含目标的bbox的回归误差，第4项
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 类别预测误差，第五项
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        # return (self.l_coord * loc_loss + 2 * contain_loss +
        # not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N
        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * nooobj_loss + class_loss) / N
