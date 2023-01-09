# https://blog.csdn.net/u013685264/article/details/100564660

from MaskRCNN.dataloader import PlannerCircleBlenderDataset, get_transform
import torch
from MaskRCNN import utils
from MaskRCNN.engine import train_one_epoch, evaluate
from MaskRCNN.model import get_instance_segmentation_model


ds_roots = []
ds_names = []

ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/PlannerDatasets/light1/')
ds_names.append('CircularPose-GTOnly')

ds_roots.append('/home/expansion/lizhaoxi/datasets/Pose/PlannerDatasets/dark1/')
ds_names.append('CircularPose-GTOnly')




# use the PennFudan dataset and defined transformations
dataset = PlannerCircleBlenderDataset(ds_roots, ds_names, get_transform(train=False))
# dataset_test = PlannerCircleBlenderDataset(root_path_test, dataset_name, get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
 
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
num_epochs = 5
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
 
    # update the learning rate
    lr_scheduler.step()
 
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    torch.save(model, './MaskRCNN/model_circular_{0}.pkl'.format(epoch))
    

torch.save(model, './MaskRCNN/model_circular.pkl')