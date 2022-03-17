import os
import cv2
import torch
import argparse
from PIL import Image
import numpy as np
from Myresnet import resnet50
from Myutils import labels2bbox, VOC_CLASSES, Color
import torchvision.transforms as transforms

data_dir = "D:\\PythonProject\\Yolov1\\images"
run_dir = "D:\\PythonProject\\Yolov1\\run"

class TestInterface(object):
    """
    网络测试接口，
    main(): 网络测试主函数
    """

    def __init__(self, opts):
        self.opts = opts
        print("=======================Start inferring.=======================")

    def main(self):
        """
        具体测试流程根据不同项目有较大区别，需要自行编写代码，主要流程如下：
        1. 获取命令行参数
        2. 获取测试集
        3. 加载网络模型
        4. 用网络模型对测试集进行测试，得到测试结果
        5. 根据不同项目，计算测试集的评价指标， 或者可视化测试结果
        """
        opts = self.opts
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        img_list = os.listdir(opts.dataset_dir)
        trans = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        # 加载权重文件
        # model = torch.load(opts.weight_path, map_location='cpu')
        model = resnet50()
        if opts.use_GPU:
            model.to(opts.GPU_id)
        for img_name in img_list:
            img_path = os.path.join(opts.dataset_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            # 在指定位置增加维度，转化为四维张量[N, C, H, W]
            img = torch.unsqueeze(img, dim=0)
            print(img_name, img.shape)
            if opts.use_GPU:
                img = img.to(opts.GPU_id)
            # tensor.detach()
            # 返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False。
            # 修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存，但如果对其中一个tensor执行某些内置操作，则会报错。
            # 例如resize_、resize_as_、set_、transpose_
            # .detach().cpu()，阻断反向传播，并将数据转移到cpu上
            preds = torch.squeeze(model(img), dim=0).detach().cpu()
            print("size:{}".format(preds.size()))
            # permute()，将tensor的维度换位：30*7*7->7*7*30
            # preds = preds.permute(1, 2, 0)
            bbox = labels2bbox(preds)
            draw_img = cv2.imread(img_path)
            self.draw_bbox(draw_img, bbox)

    def draw_bbox(self, img, bbox):
        """
        根据bbox的信息在图像上绘制bounding box
        :param img: 绘制bbox的图像
        :param bbox: 是(n,6)的尺寸，0:4是(x1,y1,x2,y2), 4是conf， 5是cls
        """
        h, w = img.shape[0:2]
        n = bbox.shape[0]
        for i in range(n):
            confidence = bbox[i, 4]
            # 提出置信度小于0.2的bbox，置信度=Pr(object)*Pr(Cls(i)|object)
            if confidence < 0.2:
                continue
            p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
            p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
            cls_name = VOC_CLASSES[int(bbox[i, 5])]
            print(cls_name, p1, p2)
            cv2.rectangle(img, p1, p2, Color[int(bbox[i, 5])], thickness=2)
            cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
            cv2.putText(img, str(confidence), (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        cv2.imshow("bbox", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # 网络测试代码
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_GPU", action="store_true", default=True, help="identify whether to use gpu")
    parser.add_argument("--GPU_id", type=int, default=None, help="device id")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\PythonProject\Yolov1\images")
    parser.add_argument("--weight_path", type=str,
                        default=r"D:\PythonProject\Yolov1\yolo.pth",
                        help="load path for model weight")
    opts = parser.parse_args()
    test_interface = TestInterface(opts)
    test_interface.main()  # 调用测试接口
