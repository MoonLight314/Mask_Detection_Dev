import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model
import serial


MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
RESULT = ['with_mask' , 'without_mask']
MARGIN_RATIO = 0.2




CONFIG = {
    "SENSOR_X_RESOLUTION" : 32,
    "SENSOR_Y_RESOLUTION" : 32,
    "CHANNELS" : 3,
    "DATA_REQUEST_COMMAND" : [0x51 , 0x75 , 0x65, 0x72 , 0x79 , 0x43 , 0x61 , 0x6C , 0x63 , 0x54 , 0x0D , 0x0A],
    "SENSOR_DATA_TOTAL_LEN" : 2060,
    "TSM_SENSOR_NAME" : "COM3",
    "TSM_SENSOR_BAUD_RATE" : 115200,
    "RESOLUTION" : (575,575)
}




# 
def OpenThermalSensor():
    
    try:
        TSM_COM_Port = serial.Serial( CONFIG["TSM_SENSOR_NAME"] , CONFIG["TSM_SENSOR_BAUD_RATE"] )
        return TSM_COM_Port
    except:
        return False





#
def GetHumanTemp( TSM_COM_Port ):

    TSM_COM_Port.write( bytearray( CONFIG["DATA_REQUEST_COMMAND"] ) )

    total_len = 0
    Received_Data = bytearray()
    ImageData = np.zeros([CONFIG["SENSOR_X_RESOLUTION"] , CONFIG["SENSOR_Y_RESOLUTION"] , CONFIG["CHANNELS"]] , dtype=np.uint8)

    # Receive Thermal Sensor Data 
    while True:
        data = TSM_COM_Port.read( 1 )
        Received_Data = Received_Data + data
        total_len = total_len + len(data)

        if total_len >= CONFIG["SENSOR_DATA_TOTAL_LEN"]:
            break

    
    DegData , AmbTemp , HumanTemp , HumanPosRow , HumanPosCol = ExtractData( Received_Data )

    return HumanTemp






# 
def ExtractData( SensorData ):    

    DegData = np.array(range( CONFIG["SENSOR_X_RESOLUTION"] * CONFIG["SENSOR_Y_RESOLUTION"] ) , np.float)

    for idx in range(2,2050,2):
        DegData[int((idx / 2)-1)] = (( SensorData[idx+1] * 256 + SensorData[idx] ) - 2731 ) / 10.0

    AmbTemp = (( SensorData[2051] * 256 + SensorData[2050] ) - 2731 ) / 10.0
    HumanTemp = float(SensorData[2052]) + (float(SensorData[2053]) / 100.0)
    HumanPosRow = SensorData[2054]
    HumanPosCol = SensorData[2056]

    return DegData , AmbTemp , HumanTemp , HumanPosRow , HumanPosCol








# 
def Inference( TSM_COM_Port ):
    print(tf.__version__)

    # Load Face Detection Model
    net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

    # Loading Model
    print("Loading Saved Model...")

    model = load_model("CheckPoints_Mask_Detection_Val_Acc_0.97494")

    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) < 0:
        ret, frame = cap.read()
        rows, cols, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1.0)#, (SIZE, SIZE))  # , (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        detection = detections[0, 0]    
        i = np.argmax(detection[:,2])

        if i != 0:
            print("Max index is not 0")
            continue

        if detection[i,2] < CONFIDENCE_FACE:
            print("Low CONFIDENCE_FACE" , detection[i,2])
            continue

        if detection[i,3] >= 1.00 or detection[i,4] >= 1.00 or detection[i,5] >= 1.00 or detection[i,6] >= 1.00 or detection[i,3] <= 0 or detection[i,4] < 0 or detection[i,5] <= 0 or detection[i,6] <= 0:
            pass
        else:
            left = int(detection[i,3] * cols)
            top = int(detection[i,4] * rows)
            right = int(detection[i,5] * cols)
            bottom = int(detection[i,6] * rows)

            left = left - int((right - left) * MARGIN_RATIO)
            top = top - int((bottom - top) * MARGIN_RATIO)
            right = right + int((right - left) * MARGIN_RATIO)
            bottom = bottom + int((bottom - top) * MARGIN_RATIO)

            if left < 0:
                left = 0

            if right > cols:
                right = cols

            if top < 0:
                top = 0

            if bottom > rows:
                bottom = rows

            cropped = frame[top:bottom, left:right]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize( cropped , dsize=(224,224) )
            cropped = np.array(cropped).reshape(-1,224,224,3)

            cropped = tf.keras.applications.resnet50.preprocess_input(cropped)

            pred = model.predict( cropped )
            print(pred)

            HumanTemp = GetHumanTemp( TSM_COM_Port )
            
            Result = "Result : {0} , Temp : {1}".format(RESULT[int(np.argmax(np.reshape( pred , (1,-1) )))] , HumanTemp )

            cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("VideoFrame", frame)
        

    cap.release()
    cv2.destroyAllWindows()

    return



if __name__== '__main__':
    TSM_COM_Port = OpenThermalSensor()
    
    if TSM_COM_Port == False:
        print("Thermal Sensor Open Error")
    else:
        Inference( TSM_COM_Port )