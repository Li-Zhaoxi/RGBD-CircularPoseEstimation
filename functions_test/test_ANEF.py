from ElpPy.pose import ANEF, poseSingleCircle
import numpy as np
from scipy.io import savemat

from ElpPy.utils import GeneralCamera, GeneralEllipse

elp_data = [100, 200, 60, 30, np.pi/3]
gcam = GeneralCamera(960, 960, 640, 360)

gelp = GeneralEllipse(elp_data=elp_data, elp_type='shape_image')
gcir1, gcir2 = poseSingleCircle(gcam, gelp, 0.8)

ptu, ptv = gelp.GenerateElpData()


# savemat('anef.mat', {'ptu': ptu, 'ptv': ptv, 'camK': gcam.cameraK(), 'gtnorm': gcir1.cnorm})
# exit()

print('before ANEF')
# print('gcir1.cnorm', gcir1.cnorm)
res_anef = ANEF(ptu, ptv, gcir1.cnorm, gcam, 0.5, 2)

print('after ANEF')

fit_elp = res_anef['gelp']

print(gelp.ellipse_shape_img())
print(fit_elp.ellipse_shape_img())
# {'gelp': GeneralEllipse(elp_data=elp_base, elp_type='shape_image'), 'err': err_base, 'norm': normal_base}