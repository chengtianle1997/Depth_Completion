import os
from shutil import copyfile, copy

data_in = "Data"
data_des = "Data_Down10"
sample = 10

def DealImageFolder(filepath, destpath, sample):
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    img_seq = os.listdir(filepath)
    i = 0
    for img in img_seq:
        if i % sample == 0:
            #Copy this image
            ori_dir = filepath + "/" + img
            dest_dir = destpath + "/" + img
            copyfile(ori_dir, dest_dir)
        i = i + 1


def ScanDepthFolder(filepath, destpath, sample):
    print("---->Solving {} Folder......\n".format(filepath))
    print("-->Saving to {} Folder......\n".format(destpath))
    date_seq = os.listdir(filepath)
    counter = 1
    for date_folder in date_seq:
        print("Processing {}/{}...\n".format(counter, len(date_seq)))
        counter = counter + 1
        #Deal with groundtruth
        gt_branch = "/" + date_folder + "/proj_depth/groundtruth"
        gt_dir = filepath + gt_branch
        gt_dest_dir = destpath + gt_branch
        gt_image02_dir = gt_dir + "/image_02"
        gt_image02_des = gt_dest_dir + "/image_02"
        gt_image03_dir = gt_dir + "/image_03"
        gt_image03_des = gt_dest_dir + "/image_03"
        DealImageFolder(gt_image02_dir, gt_image02_des, sample)
        DealImageFolder(gt_image03_dir, gt_image03_des, sample)
        #Deal with raw
        raw_branch =  "/" + date_folder + "/proj_depth/velodyne_raw"
        raw_dir = filepath + raw_branch
        raw_dest_dir = destpath + raw_branch
        raw_image02_dir = raw_dir + "/image_02"
        raw_image02_des = raw_dest_dir + "/image_02"
        raw_image03_dir = raw_dir + "/image_03"
        raw_image03_des = raw_dest_dir + "/image_03"
        DealImageFolder(raw_image02_dir, raw_image02_des, sample)
        DealImageFolder(raw_image03_dir, raw_image03_des, sample)


def ScanRgbFolder(filepath, destpath, sample):
    print("---->Solving {} Folder......\n".format(filepath))
    print("-->Saving to {} Folder......\n".format(destpath))
    date_seq = os.listdir(filepath)
    counter = 1
    for date_folder in date_seq:
        print("Processing {}/{}...\n".format(counter, len(date_seq)))
        counter = counter + 1
        #Deal with rgb image
        image02_ori = filepath + "/" + date_folder + "/image_02/data"
        image02_dest = destpath + "/" + date_folder + "/image_02/data"
        image03_ori = filepath + "/" + date_folder + "/image_03/data"
        image03_dest = destpath + "/" + date_folder + "/image_03/data"
        DealImageFolder(image02_ori, image02_dest, sample)
        DealImageFolder(image03_ori, image03_dest, sample)

'''
#Scan Depth train & val folder
depth_train_folder = data_in + "/Depth/train" 
depth_val_folder = data_in + "/Depth/val"
depth_train_dest = data_des + "/Depth/train"
depth_val_dest = data_des + "/Depth/val"
ScanDepthFolder(depth_train_folder, depth_train_dest, sample)
ScanDepthFolder(depth_val_folder, depth_val_dest, sample)
#Scan Rgb Folder
rgb_train_folder = data_in + "/Rgb/train"
rgb_val_folder = data_in + "/Rgb/val"
rgb_train_dest = data_des + "/Rgb/train"
rgb_val_dest = data_des + "/Rgb/val"
ScanRgbFolder(rgb_train_folder, rgb_train_dest, sample)
ScanRgbFolder(rgb_val_folder, rgb_val_dest, sample)
'''
#Copy depth_selection
sel_ori = data_in + "/depth_selection"
sel_des = data_des + "/depth_selection"
copy(sel_ori, sel_des)
