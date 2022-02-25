import math
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(feature_model, image, _boxes,feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
    # print(image.shape, _boxes.shape)
    N, M = image.shape[0], _boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        # boxes = boxes.squeeze(0)
        boxes = _boxes[ix][0]
        cnter = 0
        Cnter1 = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))            
            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                #print(y1,y2,x1,x2,max_h,max_w)
                if j == 0:
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)
                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
                else:
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                    examples_features = torch.cat((examples_features,feat),dim=0)
            """
            Convolving example features over image features
            """
            h, w = examples_features.shape[2], examples_features.shape[3]
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            combined = features.permute([1,0,2,3])
            # computing features for scales 0.9 and 1.1 
            for scale in exemplar_scales:
                    h1 = math.ceil(h * scale)
                    w1 = math.ceil(w * scale)
                    if h1 < 1: # use original size if scaled size is too small
                        h1 = h
                    if w1 < 1:
                        w1 = w
                    examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
                    features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    examples_features_scaled)
                    features_scaled = features_scaled.permute([1,0,2,3])
                    combined = torch.cat((combined,features_scaled),dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
    return All_feat

# def extract_features(feature_model, image, boxes_,feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
#     # print(image.shape, boxes_.shape)
#     N, M = image.shape[0], boxes_.shape[2]

#     """
#     Getting features for the image N * C * H * W
#     """
#     Image_features = feature_model(image)
#     """
#     Getting features for the examples (N*M) * C * h * w
#     """
#     for ix in range(0,N):
#         # boxes = boxes[0]
#         # boxes = boxes_[ix].squeeze(0)
#         boxes = boxes_[ix][0]
#         # print(boxes.shape)
#         cnter = 0
#         Cnter1 = 0
#         for keys in feat_map_keys:
#             image_features = Image_features[keys][ix].unsqueeze(0)
#             if keys == 'map1' or keys == 'map2':
#                 Scaling = 4.0
#             elif keys == 'map3':
#                 Scaling = 8.0
#             elif keys == 'map4':
#                 Scaling =  16.0
#             else:
#                 Scaling = 32.0
#             boxes_scaled = boxes / Scaling
#             boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
#             boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
#             boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
#             feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
#             # make sure exemplars don't go out of bound
#             boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
#             boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
#             boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
#             box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
#             box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
#             max_h = math.ceil(max(box_hs))
#             max_w = math.ceil(max(box_ws)) 

#             for j in range(0,M):
#                 # print(boxes_scaled.shape)
#                 y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
#                 y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
#                 #print(y1,y2,x1,x2,max_h,max_w)
#                 if j == 0:
#                     examples_features = image_features[:,:,y1:y2, x1:x2]
#                     if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
#                         #examples_features = pad_to_size(examples_features, max_h, max_w)
#                         examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
#                 else:
#                     feat = image_features[:,:,y1:y2, x1:x2]
#                     if feat.shape[2] != max_h or feat.shape[3] != max_w:
#                         feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
#                         #feat = pad_to_size(feat, max_h, max_w)
#                     examples_features = torch.cat((examples_features,feat),dim=0)
#             """
#             Convolving example features over image features
#             """
#             h, w = examples_features.shape[2], examples_features.shape[3]
#             features =    F.conv2d(
#                     F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
#                     examples_features
#                 )
#             combined = features.permute([1,0,2,3])
#             # computing features for scales 0.9 and 1.1 
#             for scale in exemplar_scales:
#                     h1 = math.ceil(h * scale)
#                     w1 = math.ceil(w * scale)
#                     if h1 < 1: # use original size if scaled size is too small
#                         h1 = h
#                     if w1 < 1:
#                         w1 = w
#                     examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
#                     features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
#                     examples_features_scaled)
#                     features_scaled = features_scaled.permute([1,0,2,3])
#                     combined = torch.cat((combined,features_scaled),dim=1)
#             if cnter == 0:
#                 Combined = 1.0 * combined
#             else:
#                 if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
#                     combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
#                 Combined = torch.cat((Combined,combined),dim=1)
#             cnter += 1
#         if ix == 0:
#             All_feat = 1.0 * Combined.unsqueeze(0)
#         else:
#             All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
#     return All_feat

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)