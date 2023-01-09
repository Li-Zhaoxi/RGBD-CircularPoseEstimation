import cv2
from ElpPy.dataproc import GTPoseLoader
from MaskRCNN.dataloader import PlannerCircleBlenderDataset
from PIL import Image
# dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneTrain/'
# dataset_name = 'CircularPose-15PlaneBlender'

dataset_root_path = '/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark3/'
dataset_name = 'CircularPose-RealRing40Traj'

loader = GTPoseLoader()

loader.load(dataset_root_path, dataset_name)

imgT = loader.drawGT(30)

imgT = cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)
img = Image.fromarray(imgT)
img.show()
# gt = loader.__getitem__(0, False)

# print(gt)


# img_draw = loader.drawGT(1000)

# cv2.imshow('imgT', img_draw)
# cv2.waitKey()



# ds_roots = []
# ds_names = []

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingLight1/')
# ds_names.append('CircularPose-RealRing40Traj')

# ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/CircularPose-RingDark1/')
# ds_names.append('CircularPose-RealRing40Traj')


# dsloader = PlannerCircleBlenderDataset(ds_roots, ds_names)

# # gt = dsloader[2]
# print(dsloader[20])
# print(dsloader[60])
