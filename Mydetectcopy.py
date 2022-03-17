import torch
import os
import cv2
import numpy as np
from Myresnet import resnet50
from Myutilscopy import decoder, VOC_CLASSES, Color
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
    pred = model(img)  # 1x14x14x30
    print("the output of net(Mypredict.py): {}".format(pred.size()))

    boxes, cls_indexs, probs = decoder(pred)
    print("结果框的相对位置、类别")
    print(boxes)
    print(cls_indexs)

    # 将相对框恢复到绝对框
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = np.float32(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], img_path, prob])

        for left_up, right_bottom, class_name, _, prob in result:
            color = Color[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, thickness=2)
            label = class_name + str(round(prob, 2))

            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                          color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    save_path = os.path.join(run_dir, img_name)
    # cv2.imwrite(save_path, image)
    cv2.imshow("result", image)
    cv2.waitKey()


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
