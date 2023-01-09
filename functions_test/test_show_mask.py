from PIL import Image
import cv2
from MaskRCNN.dataloader import PennFudanDataset, PlannerCircleBlenderDataset, get_transform
from PIL import Image

# root_path = '/home/expansion/lizhaoxi/datasets/PennFudanPed/'
# img_path = '/home/expansion/lizhaoxi/datasets/PennFudanPed/PNGImages/FudanPed00001.png'
# mask_path = '/home/expansion/lizhaoxi/datasets/PennFudanPed/PedMasks/FudanPed00001_mask.png'

# img = Image.open(img_path)

# mask = Image.open(mask_path)
# mask = mask.convert("P")
# mask.putpalette([
#     0, 0, 0, # black background
#     255, 0, 0, # index 1 is red
#     255, 255, 0, # index 2 is yellow
#     255, 153, 0, # index 3 is orange
# ])
# img.show()
# mask.show()



# root_path_train = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneTrain/'
# root_path_test = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/'
# dataset_name = 'CircularPose-15PlaneBlender'


# dataset = PlannerCircleBlenderDataset(root_path_train, dataset_name, get_transform(train=False))

# img = dataset.drawMask(8403)

# # print(img.shape)
# # print(img)

# img = Image.fromarray(img)
# img.show()


ds_roots = []
ds_names = []

ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight1/')
ds_names.append('CircularPose-RealRing40Traj')

ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark1/')
ds_names.append('CircularPose-RealRing40Traj')


dsloader = PlannerCircleBlenderDataset(ds_roots, ds_names, get_transform(train=False))

# gt = dsloader[2]
# print(dsloader[20])
# print(dsloader[60])

img = dsloader.drawMask(20)
img = Image.fromarray(img)
img.show()

img = dsloader.drawMask(40)
img = Image.fromarray(img)
img.show()