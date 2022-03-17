import torch
import os
import cv2
import numpy as np
from Myresnet import resnet50
from Myutilscopy import decoder, VOC_CLASSES, Color, labels2bbox
import torchvision.transforms as transforms

data_dir = "D:\\PythonProject\\Yolov1\\images"

run_dir = "D:\\PythonProject\\Yolov1\\run"

# Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
transform = transforms.Compose([
        # transforms.Resize((448, 448)),
        transforms.ToTensor(),
])


# start predict one image
def predict(model, img_name, img_path, run_dir):
    result = []
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    # 图像每个像素三通道的数值分别减去mean，类型也转化了
    img = img - np.array(mean, dtype=np.float32)

    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    print("the input of net: {}".format(img.size()))
    preds = torch.squeeze(model(img), dim=0).detach().cpu()
    bbox = labels2bbox(preds)
    h, w = image.shape[0:2]
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
        cv2.rectangle(image, p1, p2, Color[int(bbox[i, 5])], thickness=2)
        cv2.putText(image, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(image, str(confidence), (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    thickness=1)
    cv2.imshow("bbox", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # 加载模型
    print('load model...')
    model = resnet50()

    # 加载权重
    model.load_state_dict(torch.load('yolo.pth', map_location='cpu'))

    # 如果在预测的时候忘记使用model.eval()，会导致不一致的预测结果
    model.eval()

    print('predicting...')
    img_list = os.listdir(data_dir)
    for img_name in img_list:
        img_path = os.path.join(data_dir, img_name)
        predict(model, img_name, img_path, run_dir)
