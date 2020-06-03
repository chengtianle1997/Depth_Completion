# Depth_Completion
Depth completion method based on ERFNet and Stacked Hourglass network and experiments with several models. 

# Data Structure
All train and validation files are kept in folder '/Data'  
```
--Data  
  --Depth   
    (16 bit projected point cloud (PNG file) which is provided by KITTI/depth completion Dataset, including velodyne and groundtruth)  
  --Rgb   
    (RGB images)  
  --depth_selection   
    (depth selection folder which is provided by KITTI/depth completion Dataset)  
```  
# Train  
```
--mod 'erf' (Only  ERFNet)  'hourglass' (Only hourglass)  'mod' (Both ERFNet and hourglass)  
--data_path  The '/Data' folder path   
--input_type  Choose from 'rgb' for RGB-D input and 'depth' for Only depth(D) input  
--crop_w and --crop_h   Size of input images.  
```
Tips: You can only train and evaluate with GPU since we do not provide CPU method (Of course you can do it yourself)  


# Evaluate  
Remember to use the same param when you try to evaluate the model you have trained.  
And add --evaluate which means you will only evaluate your model.  
We have a conditional_save_pred function from line 574 in main.py, this function will help you to save the final result   
or other intermediate results as a 16 bit PNG file, which is similar to the depth input provided by KITTI.  
The conditional_save_conf function from line 179 in Models/model.py is used to visualize the depth confidence of two networks.  




