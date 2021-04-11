import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
import statistics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))
import shapenet_part_loader
import matplotlib.pyplot as plt

from model import PointCapsNet, CapsSegNet

#import h5py
from sklearn.svm import LinearSVC
import json


import open3d as o3d

## MONKEY PATCHING
PointCloud = o3d.geometry.PointCloud
Vector3dVector = o3d.utility.Vector3dVector
draw_geometries = o3d.visualization.draw_geometries
viz = o3d.visualization.Visualizer()
KDTreeFlann = o3d.geometry.KDTreeFlann


LATENT_CAPS_SIZE = 64
LATENT_VEC_SIZE = 64
NUM_POINTS = 2048
BATCH_SIZE = 1
CLASS_CHOICE = "Airplane"
DATASET = "shapenet_part"


def seg_and_viz(latent_caps, reconstructions):

    blue = lambda x:'\033[94m' + x + '\033[0m'
    cat_no={'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 
            'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 
            'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}    
    
    #generate part label one-hot correspondence from the catagory:
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset/shapenet/'))
    oid2cpid_file_name=os.path.join(dataset_main_path, DATASET,'shapenetcore_partanno_segmentation_benchmark_v0/shapenet_part_overallid_to_catid_partid.json')        
    oid2cpid = json.load(open(oid2cpid_file_name, 'r'))   
    object2setofoid = {}
    for idx in range(len(oid2cpid)):
        objid, pid = oid2cpid[idx]
        if not objid in object2setofoid.keys():
            object2setofoid[objid] = []
        object2setofoid[objid].append(idx)
    
    all_obj_cat_file = os.path.join(dataset_main_path, DATASET, 'shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt')
    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
#    objnames = [line.split()[0] for line in lines]
#    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()


    colors = plt.cm.tab10((np.arange(10)).astype(int))
    blue = lambda x:'\033[94m' + x + '\033[0m'

# load the model for point cpas auto encoder    
    #capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size, opt.num_points,)
    #if opt.model != '':
    #    capsule_net.load_state_dict(torch.load(opt.model))
    #if USE_CUDA:
    #    capsule_net = torch.nn.DataParallel(capsule_net).cuda()
    #capsule_net=capsule_net.eval()
 
    
# load the model for capsule wised part segmentation
    PART_SEG_MODEL_PATH = "checkpoints/part_seg_100percent.pth"
    print(PART_SEG_MODEL_PATH)

    caps_seg_net = CapsSegNet(latent_caps_size=LATENT_CAPS_SIZE, latent_vec_size=LATENT_VEC_SIZE , num_classes=50)    
    caps_seg_net.load_state_dict(torch.load(PART_SEG_MODEL_PATH))
    caps_seg_net = caps_seg_net.cuda()
    caps_seg_net = caps_seg_net.eval()    
    

    train_dataset = shapenet_part_loader.PartDataset(classification=False, class_choice=CLASS_CHOICE, npoints=NUM_POINTS, split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)        


    pcd_colored = PointCloud()                   
    pcd_ori_colored = PointCloud()        
    rotation_angle=-np.pi/4
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)           
    flip_transforms  = [[cosval, 0, sinval,-1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    flip_transformt  = [[cosval, 0, sinval,1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    
    
    correct_sum=0

    #points, part_label, cls_label= data

    #if(points.size(0)<opt.batch_size):
    #    break
    
    
    # use the pre-trained AE to encode the point cloud into latent capsules
    #points_ = Variable(points)
    #points_ = points_.transpose(2, 1)
    #if USE_CUDA:
    #    points_ = points_.cuda()
    #latent_caps, reconstructions= capsule_net(points_)
    reconstructions=reconstructions.transpose(1,2).data.cpu()
    
    
    #concatanete the latent caps with one-hot part label
    cur_label_one_hot = np.zeros((BATCH_SIZE, 16), dtype=np.float32)
    for i in range(BATCH_SIZE):
        cur_label_one_hot[i, cat_no[CLASS_CHOICE]] = 1
        iou_oids = object2setofoid[objcats[cat_no[CLASS_CHOICE]]]
        #for j in range(opt.num_points):
        #    part_label[i,j]=iou_oids[part_label[i,j]]
    cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()        
    expand =cur_label_one_hot.unsqueeze(2).expand(BATCH_SIZE, 16, LATENT_CAPS_SIZE).transpose(1,2) # latent_caps_size = 64
    expand,latent_caps=Variable(expand),Variable(latent_caps)
    expand,latent_caps=expand.cuda(),latent_caps.cuda()
    #print(expand.shape)
    #print(latent_caps.shape, "before concat")
    latent_caps=torch.cat((latent_caps,expand),2)
    #print(latent_caps.shape, "after concat")

    
    # predict the part class per capsule
    latent_caps=latent_caps.transpose(2, 1)
    output=caps_seg_net(latent_caps)        
    for i in range (BATCH_SIZE):
        iou_oids = object2setofoid[objcats[cat_no[CLASS_CHOICE]]]
        non_cat_labels = list(set(np.arange(50)).difference(set(iou_oids))) # there are 50 part classes in all the 16 catgories of objects
        mini = torch.min(output[i,:,:])
        output[i,:, non_cat_labels] = mini - 1000   
    pred_choice = output.data.cpu().max(2)[1]

    #print(pred_choice.shape)
    #print(pred_choice[0,:])
    
    # assign predicted capsule part label to its reconstructed point patch
    reconstructions_part_label=torch.zeros([BATCH_SIZE,NUM_POINTS],dtype=torch.int64)
    for i in range(BATCH_SIZE):
        for j in range(LATENT_CAPS_SIZE): # subdivisions of points from each latent cap
            for m in range(NUM_POINTS//LATENT_CAPS_SIZE): # all points in each subdivision
                reconstructions_part_label[i,LATENT_CAPS_SIZE*m+j]=pred_choice[i,j]


    
    # assign the part label from the reconstructed point cloud to the input point set with NN
    #pcd = PointCloud() 
    #pred_ori_pointcloud_part_label=torch.zeros([BATCH_SIZE,NUM_POINTS],dtype=torch.int64)   
    #for point_set_no in range (BATCH_SIZE):
    #    pcd.points = Vector3dVector(reconstructions[point_set_no,])
    #    pcd_tree = KDTreeFlann(pcd)
    #    for point_id in range (NUM_POINTS):
    #        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[point_set_no,point_id,:], 10)
    #        local_patch_labels=reconstructions_part_label[point_set_no,idx]
    #        pred_ori_pointcloud_part_label[point_set_no,point_id]=statistics.median(local_patch_labels)
    
    
    # calculate the accuracy with the GT
    #correct = pred_ori_pointcloud_part_label.eq(part_label.data.cpu()).cpu().sum()
    #correct_sum=correct_sum+correct.item()        
    #print(' accuracy is: %f' %(correct_sum/float(opt.batch_size*(batch_id+1)*opt.num_points)))

    
    
    # viz the part segmentation
    point_color=torch.zeros([BATCH_SIZE,NUM_POINTS,3])
    #point_ori_color=torch.zeros([opt.batch_size,opt.num_points,3])
    
    for point_set_no in range (BATCH_SIZE):
        iou_oids = object2setofoid[objcats[cat_no[CLASS_CHOICE]]]
        for point_id in range (NUM_POINTS):
            part_no=reconstructions_part_label[point_set_no,point_id]-iou_oids[0]
            point_color[point_set_no,point_id,0]=colors[part_no,0]
            point_color[point_set_no,point_id,1]=colors[part_no,1]
            point_color[point_set_no,point_id,2]=colors[part_no,2]
            
        pcd_colored.points=Vector3dVector(reconstructions[point_set_no,])
        pcd_colored.colors=Vector3dVector(point_color[point_set_no,])

        draw_geometries([pcd_colored])
            
        