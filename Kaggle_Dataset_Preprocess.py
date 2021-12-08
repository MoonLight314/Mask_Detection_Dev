import pandas as pd
import xml.etree.ElementTree as et
from tqdm import tqdm
import os


dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/annotations/"
data_file_path = []

for (root, directories, files) in tqdm(os.walk(dir_path)):
    for file in files:
        if '.xml' in file:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )



meta_data = pd.DataFrame({"filename":[], 
                        "mask":[],
                        "xmin":[],
                        "ymin":[],
                        "xmax":[],
                        "ymax":[]
                        })


for path in tqdm(data_file_path):

    xtree=et.parse( path )
    xroot=xtree.getroot()

    mask_flag = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for node in xroot:
        
        if node.tag == 'filename':
            fname = node.text

        if node.tag == 'object':
            name = node.find("name")
            mask_flag.append( name.text )

            box = node.find("bndbox")
            
            t = box.find("xmin")
            if t != None:
                xmin.append( t.text )

            t = box.find("ymin")
            if t != None:
                ymin.append( t.text )

            t = box.find("xmax")
            if t != None:
                xmax.append( t.text )

            t = box.find("ymax")
            if t != None:
                ymax.append( t.text )
            


    file_name = [fname] * len(xmin)

    tmp = pd.DataFrame({"filename":file_name , 
                        "mask":mask_flag,
                        "xmin":xmin,
                        "ymin":ymin,
                        "xmax":xmax,
                        "ymax":ymax
                        })

    meta_data = pd.concat( [meta_data,tmp] )

print('End')
meta_data.to_csv("meta_data.csv",index=False)