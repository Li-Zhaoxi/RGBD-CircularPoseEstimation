import numpy as np


img_width = 1280
img_height = 720
mask_num = 10

generate_mask = np.random.normal(0, 0.1, size=(mask_num, img_height, img_width, 3))


np.savez('noise_masks', masks=generate_mask)




