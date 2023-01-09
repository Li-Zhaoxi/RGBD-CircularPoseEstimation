from MaskRCNN.dataloader import PlannerCircleBlenderDataset, get_transform
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm



ds_roots = []
ds_names = []

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight1/')
# ds_names.append('CircularPose-RealRing40Traj')

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight2/')
# ds_names.append('CircularPose-RealRing40Traj')

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight3/')
# ds_names.append('CircularPose-RealRing40Traj')

ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark1/')
ds_names.append('CircularPose-RealRing40Traj')

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark2/')
# ds_names.append('CircularPose-RealRing40Traj')

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/')
# ds_names.append('CircularPose-RealRing40Traj')



dataset = PlannerCircleBlenderDataset(ds_roots, ds_names, get_transform(train=False))


device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

# model = torch.load('./MaskRCNN/model_15circleblender.pkl')
model = torch.load('./MaskRCNN/model_circular.pkl')
model.eval()

total_num = len(dataset)

for idx_image in tqdm(range(total_num)):
    # idx_image = 144
    img, target = dataset[idx_image]
    with torch.no_grad():
        prediction = model([img.to(device)])
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].mul(255).byte().cpu().numpy()
    
    img_path = target['color_full_path']
    
    # data = {}
    # data['boxes'] = boxes
    # data['labels'] = labels
    # data['scores'] = scores
    # data['masks'] = masks
    
    # # 结果可视化
    # img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # masks = prediction[0]['masks']
    # premask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    # img.show()
    # premask.show()
    # exit()
    
    np.savez(img_path, boxes=boxes, labels=labels, scores = scores, masks = masks)
    
    # print(prediction)
    # exit()
    


# # pick one image from the test set
# img, _ = dataset[100]



    
# print(prediction)

# img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

# masks = prediction[0]['masks']

# print(masks.shape)
 
# premask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
# # premask1 = Image.fromarray(prediction[0]['masks'][1, 0].mul(255).byte().cpu().numpy())
# # premask2 = Image.fromarray(prediction[0]['masks'][2, 0].mul(255).byte().cpu().numpy())
# img.show()
# premask.show()
# # premask1.show()
# # premask2.show()