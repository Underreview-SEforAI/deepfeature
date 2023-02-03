import numpy as np
import cv2
import torch
import random
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur
from torch.nn.functional import mse_loss, conv2d



class benign_aug():
    def __init__(self, params, model, seed):
        self.params = params
        self.model = model
        self.seed = seed
    
    def random_shift(self, img: torch.Tensor):
        # random.seed(self.seed)
        c, w, h = img.shape
        imgc = random.uniform(*self.params['shift_x'])
        yc = random.uniform(*self.params['shift_y'])

        img_shift = int(random.choice([-1,1]) * imgc * w)
        y_shift = int(random.choice([-1,1]) * yc * h)
        img = F.affine(img, angle = 0, translate = [img_shift, y_shift], scale=1, shear=0)

        return img

    def random_rotation(self, img: torch.Tensor):
        # random.seed(self.seed)
        angle = random.uniform(*self.params['rotate'])
        img = F.affine(img, angle = angle, translate = [0, 0], scale=1, shear=0)

        return img

    def random_scale(self, img: torch.Tensor):
        # random.seed(self.seed)
        scaling_factor = random.uniform(*self.params['scale'])
        img = F.affine(img, angle = 0, translate = [0, 0], scale=scaling_factor, shear=0)

        return img

    def random_blur(self, img: torch.Tensor):
        def get_gaussian_filter(kernel_size = 3):
            kernel = cv2.getGaussianKernel(kernel_size,0).dot(cv2.getGaussianKernel(kernel_size,0).T)
            kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 1, 3, k_s, k_s
            kernel = torch.nn.Parameter(data = kernel, requires_grad=False).cuda()
            return kernel

        def get_mean_filter(kernel_size = 3):
            kernel = torch.nn.Parameter(data = torch.ones(1, 1, kernel_size, kernel_size),
                                        requires_grad = False).cuda()
            return kernel

        if self.params['blur_mode'] == 'hard':
            op_list = [0, 1, 2, 3, 4, 5, 6]
        elif self.params['blur_mode'] == "easy":
            op_list = [4]
        else:
            op_list = [-1] 

        shape = img.shape
        blur = []

        # random.seed(self.seed)
        blur_op = random.choice(op_list)
        # print("blur_op", blur_op)
        if blur_op == 0:
            kernel = get_mean_filter(2)
        if blur_op == 1:
            kernel = get_mean_filter(3)
        if blur_op == 2:
            kernel = get_mean_filter(4)
        if blur_op == 3:
            kernel = get_mean_filter(5)
        if blur_op == 4:
            kernel = get_gaussian_filter(3)
        if blur_op == 5:
            kernel = get_gaussian_filter(5)
        if blur_op == 6:
            kernel = get_gaussian_filter(7)

        if blur_op != -1 and img.shape[0] != 1:
            img[0] = conv2d(img[0].unsqueeze(0), kernel, stride = 1, padding='same').squeeze()
            img[1] = conv2d(img[1].unsqueeze(0), kernel, stride = 1, padding='same').squeeze()
            img[2] = conv2d(img[2].unsqueeze(0), kernel, stride = 1, padding='same').squeeze()
        elif blur_op != -1 and img.shape[0] == 1:
            img = conv2d(img, kernel, stride = 1, padding='same')

        return img

    def random_shear(self, img: torch.Tensor):
        # random.seed(self.seed)
        shear_angle = random.uniform(*self.params['shear'])
        img = F.affine(img, angle = 0, translate = [0, 0], scale=1, shear=shear_angle)

        return img

    def random_contrast(self, img: torch.Tensor):
        # random.seed(self.seed)
        factor = random.uniform(*self.params['contrast'])
        img = F.adjust_contrast(img, factor)

        return img

    def random_brightness(self, img: torch.Tensor):
        # random.seed(self.seed)
        factor = random.uniform(*self.params['brightness'])
        img = F.adjust_brightness(img, factor)

        return img



    def __call__(self, imgs: torch.Tensor):

        # _, clean_feature_map = self.model(imgs)

        random.seed(self.seed)
        aug_imgs = torch.zeros_like(imgs)

        # aug_imgs = self.random_blur(imgs)
        for idx in range(imgs.shape[0]):
            aug_idx = random.randint(0, 6)
            # aug_idx = 6
            if aug_idx == 0:
                aug_imgs[idx] = self.random_shift(imgs[idx])
            elif aug_idx == 1:
                aug_imgs[idx] = self.random_rotation(imgs[idx])
            elif aug_idx == 2:
                aug_imgs[idx] = self.random_scale(imgs[idx])
            elif aug_idx == 3:
                aug_imgs[idx] = self.random_shear(imgs[idx])
            elif aug_idx == 4:
                aug_imgs[idx] = self.random_blur(imgs[idx])
            elif aug_idx == 5:
                aug_imgs[idx] = self.random_contrast(imgs[idx])
            elif aug_idx == 6:
                aug_imgs[idx] = self.random_brightness(imgs[idx])
            
        # diff = []
        # outputs, adv_feature_map = self.model(aug_imgs)
        # for i in range(aug_imgs.shape[0]):
        #     diff.append(mse_loss(adv_feature_map[i], clean_feature_map[i]).cpu().detach().numpy())



        return aug_imgs






