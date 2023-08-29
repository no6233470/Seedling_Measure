import colorsys
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from unet_model import Unet as unet
from utils import cvtColor, preprocess_input, resize_image


class Unet(object):
    _defaults = {
        "model_path"    : 'model_data/unet_raw.pth',
        "num_classes"   : 3+1,
        "backbone"      : "vgg",
        "input_shape"   : [512, 512],
        "mix_type"      : 2,
        "cuda"          : False,
    }

    #   初始化UNET
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #   获得所有的分类
    def generate(self, onnx=False):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #   检测图片
    def detect_image(self, image):
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #   图片传入网络进行预测
            pr = self.net(images)[0]
            #   取出每一个像素点的种类
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #   取出每一个像素点的种类
            pr = pr.argmax(axis=-1)

            seg_img1 = (np.expand_dims(pr == 1, -1) * np.array(old_img, np.float32)).astype('uint8')
            seg_img2 = (np.expand_dims(pr == 2, -1) * np.array(old_img, np.float32)).astype('uint8')
            seg_img3 = (np.expand_dims(pr == 3, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = [seg_img1,seg_img2,seg_img3]
        return image
