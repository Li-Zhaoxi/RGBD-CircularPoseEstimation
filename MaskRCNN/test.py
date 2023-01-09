from MaskRCNN.dataloader import PennFudanDataset, get_transform
import torch
from PIL import Image

root_path = '/home/expansion/lizhaoxi/datasets/PennFudanPed/'
dataset = PennFudanDataset(root_path, get_transform(train=False))


device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


model = torch.load('./MaskRCNN/model.pkl')

# pick one image from the test set
img, _ = dataset[100]

model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
    
    
img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
 
premask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


img.show()
premask.show()