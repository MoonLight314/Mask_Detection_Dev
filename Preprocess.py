import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf



MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RATIO = 0.2






def verify_image_file():

    meta_data = pd.read_csv("merged_meta_data_211209_Rev_01.csv")

    train_left = meta_data['xmin'].tolist()
    train_right = meta_data['xmax'].tolist()
    train_top = meta_data['ymin'].tolist()
    train_bottom = meta_data['ymax'].tolist()
    train_mask = meta_data['mask'].tolist()
    file_path_train = meta_data['filename'].tolist()

    new_left = []
    new_right = []
    new_top = []
    new_bottom = []
    new_file_path = []
    new_mask = []

    for idx,image_path in tqdm(enumerate( file_path_train)):
        
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)   
            
            img = tf.image.crop_to_bounding_box( img , train_top[idx] , train_left[idx], train_bottom[idx] - train_top[idx] , train_right[idx] - train_left[idx] )

            """
            output_image = tf.image.encode_png(img)
            file_name = tf.constant('./Ouput_image.png')
            file = tf.io.write_file(file_name, output_image)    
            """
            
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.resnet50.preprocess_input(img)

            new_left.append(train_left[idx])
            new_right.append(train_right[idx])
            new_top.append(train_top[idx])
            new_bottom.append(train_bottom[idx])
            new_file_path.append(image_path)
            new_mask.append(train_mask[idx])
        
        except:
            continue
    
    print(len(new_file_path))

    result = pd.DataFrame(list(zip(new_file_path, new_mask , new_left , new_top , new_right , new_bottom)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    result.to_csv("merged_meta_data_211209_Rev_02.csv",index=False)

    return







def temp2():
    
    meta_data = pd.read_csv("merged_meta_data_211209.csv")

    filenames = meta_data['filename'].tolist()
    xmax = meta_data['xmax'].tolist()
    ymax = meta_data['ymax'].tolist()

    for idx,filename in tqdm( enumerate(filenames)):
        img = cv2.imread(filename)
        rows, cols, channels = img.shape

        if xmax[idx] >= cols:
            print(xmax[idx] - cols)
            xmax[idx] = xmax[idx] - 1
        
        if ymax[idx] >= rows:
            print(ymax[idx] - rows)
            ymax[idx] = ymax[idx] - 1

    meta_data['xmax'] = xmax
    meta_data['ymax'] = ymax

    meta_data.to_csv("meta_data_211209_Rev_01.csv",index=False)    
    
    return





def temp():
    
    meta_data = pd.read_csv("meta_data_211203.csv")

    filenames = meta_data['filename'].tolist()
    file_path = []
    masks = meta_data['mask'].tolist()
    new_masks = []

    for idx,filename in enumerate( filenames):
        file_path.append("./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/" + filename)

        if masks[idx] == 'mask_weared_incorrect':
            new_masks.append('with_mask')
        else:
            new_masks.append(masks[idx])

    meta_data['filename'] = file_path
    meta_data['mask'] = new_masks

    meta_data.to_csv("meta_data_211209.csv",index=False)
    
    return



def get_face_coor_info():
    
    pass_file_list = []
    lefts = []
    tops = []
    rights = []
    bottoms = []
    masks = []

    net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

    meta_data = pd.read_csv("meta_data_Face-Mask-Detection-master.csv" , encoding='CP949')
    file_path = meta_data['file_path'].tolist()

    for file in tqdm(file_path):

        try:
            img = cv2.imread(file)
            rows, cols, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0)#, (SIZE, SIZE))  # , (104.0, 177.0, 123.0))

            net.setInput(blob)
            detections = net.forward()

            detection = detections[0, 0]    
            i = np.argmax(detection[:,2])

            if i != 0:
                print(file , "Max index is not 0")
                continue

            if detection[i,2] < CONFIDENCE_FACE:
                print(file , "Low CONFIDENCE_FACE" , detection[i,2])
                continue
            

            left = detection[i,3] * cols
            top = detection[i,4] * rows
            right = detection[i,5] * cols            
            bottom = detection[i,6] * rows

            left = int(left - int((right - left) * MARGIN_RATIO))
            top = int(top - int((bottom - top) * MARGIN_RATIO))
            right = int(right + int((right - left) * MARGIN_RATIO))
            bottom = int(bottom + int((bottom - top) * MARGIN_RATIO / 2))

            if left < 0:
                left = 0

            if right > cols:
                right = cols

            if top < 0:
                top = 0

            if bottom > rows:
                bottom = rows

            pass_file_list.append(file)
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)

            if "with_mask" in file:
                masks.append("with_mask")
            elif "without_mask" in file:
                masks.append("without_mask")
        
        except:
            print(file , " Error")


    print(len(pass_file_list))

    result = pd.DataFrame(list(zip(pass_file_list, masks , lefts , tops , rights , bottoms)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    result.to_csv("meta_data_Usefull_Face-Mask-Detection-master.csv",index=False)

    return





def save_file_fist():
    dir_path = "./Dataset/Face-Mask-Detection-master"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )


    print( len(data_file_path) )

    meta_data = pd.DataFrame( data_file_path , columns=['file_path'])
    meta_data.to_csv("meta_data_Face-Mask-Detection-master.csv",index=False)




if __name__== '__main__':
    #temp2()
    #temp()
    #save_file_fist()
    #get_face_coor_info()
    verify_image_file()