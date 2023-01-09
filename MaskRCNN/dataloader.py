from logging import root
import os
import cv2
import torch
import numpy as np
from torch._C import dtype
import torch.utils.data
from PIL import Image
from ElpPy.dataproc import GTPoseLoader
from MaskRCNN import utils
from MaskRCNN import transforms as T 
from MaskRCNN.engine import train_one_epoch, evaluate
import bisect
 
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)
 
class PlannerCircleBlenderDataset(torch.utils.data.Dataset):
    def __init__(self, roots, ds_names, transforms=None) -> None:
        super().__init__()
        
        assert(isinstance(roots, list) and isinstance(ds_names, list))
        
        
        self.dsloaders = []
        self.total_ds = len(roots)  # 数据集的个数
        self.total_data = 0 # 所有数据集图像的总数
        self.num_ds = []  # 每个数据集的图像个数
        self.cum_ds = []  # 数据集的累加个数,用于从索引获得数据集的索引
        # print(roots, ds_names)
        for each_root, each_ds_name in zip(roots, ds_names):
            tmp = GTPoseLoader()
            tmp.load(each_root, each_ds_name)
            tmp_num = len(tmp)
            self.dsloaders.append(tmp)
            
            self.num_ds.append(tmp_num)
            self.total_data += tmp_num
            
            if len(self.cum_ds) == 0:
                self.cum_ds.append(tmp_num)
            else:
                self.cum_ds.append(tmp_num + self.cum_ds[-1])
            
        self.roots = roots
        self.transforms = transforms
    
    def __getitem__(self, idx):
        
        # ds_idx: 数据集索引 img_idx: 图像索引
        ds_idx = bisect.bisect_left(self.cum_ds, idx + 1) 
        if ds_idx == 0:
            img_idx = idx
        else:
            img_idx = idx - self.cum_ds[ds_idx - 1]
        # print(self.cum_ds)
        # print(ds_idx, img_idx)
        gt = self.dsloaders[ds_idx].__getitem__(img_idx)
        
        # print('idx', idx)
        # gt = self.dsloader.__getitem__(idx)
        # gt = self.dsloader[idx, False]
        # load images ad masks
        img_path = gt['color_full_path']
        
        if 'mask_full_path' not in gt.keys():
            mask_path = os.path.join(self.roots[ds_idx], 'mask', 'MASK_{0}.png'.format(idx))
        else:
            mask_path = gt['mask_full_path']
            
        img = Image.open(img_path).convert("RGB")
        # print(img.size())
        # img.show()
        # exit()
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
        
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["color_full_path"] = img_path
        
        # print('boxes', boxes)
        
        # print('idx:', idx)
        # print("area:", area)
        
        
        isValid = 1
        for each_box in boxes:
            width = each_box[2] - each_box[0]
            height = each_box[3] - each_box[1]
            if width <= 6 or height <= 4:
                isValid = 0
                print('find invalid {0} image at {1}'.format(idx, img_path))
                break
        
        tensor_valid = torch.as_tensor(isValid, dtype=torch.uint8)
        target['isValid'] = tensor_valid
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        
 
        return img, target
    
    def __len__(self):
        return self.total_data
    
    def drawMask(self, idx):
        img, target = self.__getitem__(idx)
        # print(img)
        img = np.ascontiguousarray((img.permute(1,2,0).numpy() * 255).astype(dtype='uint8'))
        print('img shape', img.shape)
        boxes = target['boxes']

        
        
        for each_box in boxes:
            # img = 
            cv2.rectangle(img, (int(each_box[0] + 0.5), int(each_box[1] + 0.5)),
                          (int(each_box[2] + 0.5), int(each_box[3] + 0.5)), (255, 0, 0), 2)
        return img
 
 
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
 
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)