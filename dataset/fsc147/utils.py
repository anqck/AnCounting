from torchvision import transforms
import torch

import cv2
import numpy as np


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


Normalize = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample

class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


class resizeImageWithGT_HW(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        # 3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, height=512, width = 512):
        self.height = height
        self.width =  width

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size

        scale_factor_height = float(self.height)/ H
        scale_factor_width = float(self.width)/ W

        # new_H = 8*int(H*scale_factor_height/8)
        # new_W = 8*int(W*scale_factor_width/8)

        new_H = self.height
        new_W = self.width

        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            # box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box[0]*scale_factor_height, box[1]*scale_factor_width, box[2]*scale_factor_height, box[3]*scale_factor_width
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample

