import torch
import numpy as np

grid_num = 14

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):
    '''
    pred (tensor) 1x14x14x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    # 剔除无用的维度
    pred = pred.squeeze(0)  # 14x14x30
    # 在指定维度增加一个维度
    # 在获取置信度的过程中会自动丢弃最后一个维度
    contain1 = pred[:, :, 4].unsqueeze(2)
    print("the size of contain1: {}".format(contain1.size()))
    contain2 = pred[:, :, 9].unsqueeze(2)
    print("the size of contain2: {}".format(contain2.size()))
    contain = torch.cat((contain1, contain2), dim=2)
    print("the size of contain: {}".format(contain.size()))
    mask1 = contain > 0.1  # 大于阈值，返回True、False
    print("the size of mask1: {}".format(mask1.size()))
    # we always select the best contain_prob what ever it > 0.9，返回True、False
    mask2 = (contain == contain.max())
    print("the size of mask2: {}".format(mask2.size()))

    # lt（小于）、gt（大于）、eq（等于）、le（小于等于）、ge（大于等于），返回True、False
    mask = (mask1 + mask2).gt(0)
    print("the size of mask: {}".format(mask.size()))
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    # print("候选框")
                    # print(i, j, b)
                    # 获取框的位置
                    box = pred[i, j, b * 5:b * 5 + 4]
                    # 获取框的预测概率
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    # 获取框的x,y索引，cell左上角  up left of cell
                    xy = torch.FloatTensor([j, i]) * cell_size
                    # return cxcy relative to image
                    box[:2] = box[:2] * cell_size + xy
                    # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    # 获取类别预测的概率
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    print("max_prob: {} | cla_index: {}".format(max_prob, cls_index))
                    # 满足阈值判断则将框的坐标、类别、概率分别存入boxes, cls_indexs, probs
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)

    # 获取列表的长度
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, dim=0)  # (n,4)
        probs = torch.cat(probs, dim=0)  # (n,)
        cls_indexs = torch.stack(cls_indexs, dim=0)  # (n,)
    print("boxes.size: {}".format(boxes.size()))
    print("probs.size: {}".format(probs.size()))
    print("cls_indexs.size: {}".format(cls_indexs.size()))
    print("the value of probs: {}".format(probs))

    keep = mynms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def mynms(bboxes, scores, threshold=0.1):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 对于概率值进行排序，order中存放索引
    # _, order = scores.sort(0, descending=True)
    _, order = torch.sort(scores, dim=0, descending=True)
    keep = []
    # 判断order的元素个数
    print("dim of order: {}".format(order.dim()))
    # 返回数组中元素的个数
    while order.numel() > 0:
        if order.dim() >= 1:
            i = order[0]
        else:
            i = order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
        # xx1 = x1[order[1:]].clamp(min=x1[i])
        # yy1 = y1[order[1:]].clamp(min=y1[i])
        # xx2 = x2[order[1:]].clamp(max=x2[i])
        # yy2 = y2[order[1:]].clamp(max=y2[i])
        # w = (xx2 - xx1).clamp(min=0)
        # h = (yy2 - yy1).clamp(min=0)
        # xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        # yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        # xx2 = torch.clamp(x2[order[1:]], min=x2[i])
        # yy2 = torch.clamp(y2[order[1:]], min=y2[i])
        # w = torch.clamp((xx2-xx1), min=0)
        # h = torch.clamp((yy2-yy1), min=0)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        print("iou: {}".format(iou))
        ids = (iou <= threshold).nonzero().squeeze()
        print("ids: {}".format(ids))
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的1*14*14*30的数据转换为bbox的(392,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果,bboxes.shape = (-1, 6), 0:4是(x1,y1,x2,y2), 4是conf， 5是cls
    """
    if matrix.size()[0:2] != (14, 14):
        raise ValueError("Error: Wrong labels size: ", matrix.size(), " != (14,14)")
    matrix = matrix.numpy()
    # bboxes存储392个预测的bbox的x1,y1,x2,y2,conf,cls
    bboxes = np.zeros((392, 6))
    # 先把14*14*30的数据转变为bbox的(392,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    matrix = matrix.reshape(196, -1)     # 14*14*30 --> 196*30
    bbox = matrix[:, :10].reshape(392, 5)    # 196*10 --> 392*5
    r_grid = np.array(list(range(14)))
    # 对数字进行重复，repeats：次数，axis：重复的维度
    r_grid = np.repeat(r_grid, repeats=28, axis=0)  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]
    c_grid = np.array(list(range(14)))
    # [np.newaxis, :]，扩充一个维度
    c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]    # [[0 0 1 1 2 2 3 3 4 4 5 5 6 6]]
    c_grid = np.repeat(c_grid, repeats=14, axis=0).reshape(
        -1)  # [0 0 1 1 2 2 3 3 4 4 5 5 6 6 0 0 1 1 2 2 3 3 4 4 5 5 6 6...]
    # 将坐标从(px,py,w,h)转换为(x1,y1,x2,y2)，但此时还是相对值
    bboxes[:, 0] = np.maximum((bbox[:, 0] + c_grid) / 14.0 - bbox[:, 2] / 2.0, 0)
    bboxes[:, 1] = np.maximum((bbox[:, 1] + r_grid) / 14.0 - bbox[:, 3] / 2.0, 0)
    bboxes[:, 2] = np.minimum((bbox[:, 0] + c_grid) / 14.0 + bbox[:, 2] / 2.0, 1)
    bboxes[:, 3] = np.minimum((bbox[:, 1] + r_grid) / 14.0 + bbox[:, 3] / 2.0, 1)
    bboxes[:, 4] = bbox[:, 4]
    # 按指定的维度搜索最大值的索引
    cls = np.argmax(matrix[:, 10:], axis=1)
    # 每个grid有两个候选框，所以需要将cls复制一次
    cls = np.repeat(cls, repeats=2, axis=0)
    bboxes[:, 5] = cls
    # 对所有392个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox
    keepid = nms_multi_cls(bboxes, thresh=0.1, n_cls=20)
    ids = []
    for x in keepid:
        ids = ids + list(x)
    ids = sorted(ids)
    return bboxes[ids, :]


def nms_1cls(dets, thresh):
    """
    单类别NMS：这种非极大值抑制是在类内进行的
    :param dets: ndarray,nx5,dets[i,0:4]分别是bbox坐标(x1,y1,x2,y2)；dets[i,4]是置信度score
    :param thresh: NMS算法设置的iou阈值
    """
    # 从检测结果dets中获得x1,y1,x2,y2和scores的值
    # x1,y1,x2,y2,scores为列表
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照置信度score的值降序排序的"下标序列"
    # [::-1]顺序相反操作，这里会使得所有的框索引倒置，why?
    # 答：为了进行非极大值一直，将所有概率值从大到小排列，并得到排列后概率值对应索引构成的数组
    # 这里的order为类内框重新生成的索引0~
    # det为6，48，65，86所对应的候选框，根据里面scores的大小，由大到小重新编码为0，1，2，3
    order = scores.argsort()[::-1]

    # keep用来保存最后保留的检测框的index
    keep = []
    while order.size > 0:
        # 当前置信度最高bbox的index
        i = order[0]
        # 添加当前剩余检测框中得分最高的index到keep中
        keep.append(i)
        # 得到此bbox和剩余其他bbox的相交区域，左上角和右下角
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积/(面积1+面积2-重叠面积)
        # iou：列表，size为x1,y1,x2,y2,scores的大小-1
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的bbox，inds为对应的索引
        inds = np.where(iou <= thresh)[0]
        # 将原order修改为IoU小于阈值的bbox的索引
        order = order[inds + 1]
    return keep


def nms_multi_cls(dets, thresh, n_cls):
    """
    多类别的NMS算法
    :param dets:ndarray,nx6,dets[i,0:4]是bbox坐标；dets[i,4]是置信度score；dets[i,5]是类别序号；
    :param thresh: NMS算法的阈值；
    :param n_cls: 是类别总数
    """
    # 储存结果的列表，keeps_index[i]表示第i类保留下来的bbox下标list
    keeps_index = []
    for i in range(n_cls):
        # np.where(dets[:, 5] == i)[0]，返回满足条件的行索引构成的列表
        # 即边界框的索引或者编号构成的列表：order_i
        # 例如：6，48，65，86
        order_i = np.where(dets[:, 5] == i)[0]
        # 返回符合dets[:, 5] == i的array，即从96个候选框中剔除不属于第i类的候选框
        # det为6，48，65，86所对应的候选框
        det = dets[dets[:, 5] == i, 0:5]
        # 判断det的行数
        if det.shape[0] == 0:
            keeps_index.append([])
            continue    # 直接进入下一个for循环
        keep = nms_1cls(det, thresh)
        # 根据返回的索引值keep通过order_i[keep]获取保留的候选框的index
        keeps_index.append(order_i[keep])
    return keeps_index
