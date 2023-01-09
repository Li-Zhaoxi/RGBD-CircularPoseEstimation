# https://blog.csdn.net/u013685264/article/details/100564660

from MaskRCNN.dataloader import PennFudanDataset, get_transform
import torch
from MaskRCNN import utils
from MaskRCNN.engine import train_one_epoch, evaluate
from MaskRCNN.model import get_instance_segmentation_model

root_path = '/home/expansion/lizhaoxi/datasets/PennFudanPed/'



# use the PennFudan dataset and defined transformations
dataset = PennFudanDataset(root_path, get_transform(train=True))
dataset_test = PennFudanDataset(root_path, get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
 
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
 
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
 
# the dataset has two classes only - background and person
num_classes = 2
 
# get the model using the helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
 
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
 
# training
num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
 
    # update the learning rate
    lr_scheduler.step()
 
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    

torch.save(model, './MaskRCNN/model.pkl')