import argparse
import os
from ElpPy.dataproc import GTPoseLoader
from tqdm import tqdm
import numpy as np
import cv2
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Basic Features Generation.')
    parser.add_argument('--ds_root', type=str, 
                        default='/home/expansion/lizhaoxi/datasets/Pose/CircularPose-15PlaneBlender/',
                        help='The root path of the used dataset')
    parser.add_argument('--ds_name', type=str,
                        default='CircularPose-15PlaneBlender',
                        help='The dsname is used to generate usage datas in class GTPoseLoader')
    parser.add_argument(
        '--elsd', action='store_true',
        help='run ELSD')
    
    parser.add_argument(
        '--lsd', action='store_true',
        help='run LSD')
    
    parser.add_argument(
        '--harris', action='store_true',
        help='run LSD')
    
    parser.add_argument(
        '--maskrcnn', action='store_true',
        help='run LSD')
    parser.add_argument(
        '--cuda_idx', type=int, default=0,
        help='specific cuda device index')
    parser.add_argument(
        '--maskrcnn_pkl', type=str, default='./MaskRCNN/model_15circleblender.pkl',
        help='MaskRCNN model file')
    
    args = parser.parse_args()
    
    return args
    
    
def main(opt):
    dataset_root_path = opt.ds_root
    dataset_name = opt.ds_name
    
    use_elsd = opt.elsd
    use_lsd = opt.lsd
    use_harris = opt.harris
    use_maskrcnn = opt.maskrcnn
    
    loader = GTPoseLoader()
    loader.load(dataset_root_path, dataset_name)
    
    gt_num = len(loader)
    
    folder_methods_name = 'ExtractedFeatures'
    
    print('Total {0} RGBD data in {1} with name: {2}.'.format(
        gt_num, dataset_root_path, dataset_name))
    
    feature_root_path = os.path.join(dataset_root_path, folder_methods_name)
    if not os.path.exists(feature_root_path):
        os.mkdir(feature_root_path)
    
    if use_elsd:
        print('----> Progress ELSD.')
        save_root_path = os.path.join(dataset_root_path, folder_methods_name, 'ELSD')
        if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
    
        for idx_image in tqdm(range(gt_num)):
            gt = loader[idx_image]
            
            color_full_path = gt['color_full_path']
            
            cmd_str = './ELSD/bin/run_elsd {0}'.format(color_full_path)
            os.system(cmd_str)
            
            os.system('mv {0}.json {1}'.format(color_full_path, save_root_path))
            os.system('mv {0}.svg {1}'.format(color_full_path, save_root_path))
            os.system('mv {0}.ellipses.txt {1}'.format(color_full_path, save_root_path))
        
    if use_lsd:
        print('----> Progress LSD.')
        save_root_path = os.path.join(dataset_root_path, folder_methods_name, 'LSD')
        if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
    
        for idx_image in tqdm(range(gt_num)):
            gt = loader[idx_image]
            
            color_full_path = gt['color_full_path']
            
            cmd_str = './LSD/bin/run_lsd {0}'.format(color_full_path)
            os.system(cmd_str)
            
            os.system('mv {0}.json {1}'.format(color_full_path, save_root_path))    
            
    if use_harris:
        print('----> Progress Harris.')
        save_root_path = os.path.join(dataset_root_path, folder_methods_name, 'Harris')
        if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
    
        for idx_image in tqdm(range(gt_num)):
            gt = loader.__getitem__(idx_image, True)
            
            color_full_path = gt['color_full_path']
            imgG = gt['imgG']
            imgC = gt['imgC']
            
            imgG = np.float32(imgG)
    
            dst = cv2.cornerHarris(imgG, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            
            pi, pj = np.where(dst > 0.005*dst.max())
            
            save_dict = {}
            save_dict['pi'] = pi.tolist()
            save_dict['pj'] = pj.tolist()
            
            save_path = color_full_path + '.json'
            
            json_str = json.dumps(save_dict, indent=1)
            
            with open(save_path, 'w') as f:
                f.write(json_str)
                f.close()
            
            os.system('mv {0}.json {1}'.format(color_full_path, save_root_path))
            
            # imgC[pi, pj, :] = [0, 0, 255]
            # # imgC[dst > 0.005*dst.max()] = [255, 0, 0]
            
            # image = Image.fromarray(imgC)
            # image.show()
        
            # exit()
            
    if use_maskrcnn:
        print('----> Progress MaskRCNN.')
        import torch
        from MaskRCNN.dataloader import PlannerCircleBlenderDataset, get_transform

        
        save_root_path = os.path.join(dataset_root_path, folder_methods_name, 'MaskRCNN')
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
                
        dataset = PlannerCircleBlenderDataset(dataset_root_path, dataset_name, get_transform(train=False))
        
        str_cuda_dev = 'cuda:{0}'.format(opt.cuda_idx)
        device = torch.device(str_cuda_dev) if torch.cuda.is_available() else torch.device('cpu')
        
        model_pkl_path = opt.maskrcnn_pkl
        
        model = torch.load(model_pkl_path)
        model.eval()
        total_num = len(dataset)
        for idx_image in tqdm(range(total_num)):
            # idx_image = 144
            img, target = dataset[idx_image]
            gt = dataset.dsloader.__getitem__(idx_image)
            with torch.no_grad():
                prediction = model([img.to(device)])
            
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            masks = prediction[0]['masks'].mul(255).byte().cpu().numpy()
            
            color_full_path = gt['color_full_path']
            
            # data = {}
            # data['boxes'] = boxes
            # data['labels'] = labels
            # data['scores'] = scores
            # data['masks'] = masks
            
            np.savez(color_full_path, boxes=boxes, labels=labels, scores = scores, masks = masks)
            
            # print(prediction)
            # exit()
            
            os.system('mv {0}.npz {1}'.format(color_full_path, save_root_path))

        
        
        
if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
    
    