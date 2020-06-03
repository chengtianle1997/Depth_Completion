import os

Rgbdatadir = "Data/Rgb"

def ScanFolder(filepath):
    print("--->Solving {} Folder......\n".format(filepath))
    date_seq = os.listdir(filepath)
    counter = 1
    for date_folder in date_seq:
        print("Processing {}/{}...\n".format(counter,len(date_seq)))
        counter = counter + 1
        #Deal with Image02
        image02_dir = filepath + "/" + date_folder + "/image_02"
        image02_txt = image02_dir + "/timestamps.txt"
        image02_imdir = image02_dir + "/data"
        image02_seq = os.listdir(image02_imdir)
        image02_num = len(image02_seq)
        #delete timestamp
        if os.path.exists(image02_txt):
            os.remove(image02_txt)
        #delete head 5 images
        for i in range(0,5):
            image02_path = image02_imdir + "/" + str(i).rjust(10,'0') + ".png"
            if os.path.exists(image02_path):
                os.remove(image02_path)
        #delete tail 5 images
        for i in range(image02_num-5, image02_num):
            image02_path = image02_imdir + "/" + str(i).rjust(10,'0') + ".png"
            if os.path.exists(image02_path):
                os.remove(image02_path)
        
        #Deal with Image03
        image03_dir = filepath + "/" + date_folder + "/image_03"
        image03_txt = image03_dir + "/timestamps.txt"
        image03_imdir = image03_dir + "/data"
        image03_seq = os.listdir(image03_imdir)
        image03_num = len(image03_seq)
        #delete timestamp
        if os.path.exists(image03_txt):
            os.remove(image03_txt)
        #delete head 5 images
        for i in range(0,5):
            image03_path = image03_imdir + "/" + str(i).rjust(10,'0') + ".png"
            if os.path.exists(image03_path):
                os.remove(image03_path)
        #delete tail 5 images
        for i in range(image03_num-5, image03_num):
            image03_path = image03_imdir + "/" + str(i).rjust(10,'0') + ".png"
            if os.path.exists(image03_path):
                os.remove(image03_path)



ScanFolder("Data/Rgb/train")
ScanFolder("Data/Rgb/val")