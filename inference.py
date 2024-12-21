import cv2
import numpy as np
import os
from keras.models import load_model

#function to collect data and preprocess it
def collect_data(address):
    img_data=cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    img_data=cv2.resize(img_data,(28,28))
    img_data=img_data.reshape(-1,28,28,1)
    return img_data

#function to classify the image
def classification(user_data):
    model=load_model('model.h5')
    predict_model=model.predict(user_data)
    prediction=np.argmax(predict_model, axis=-1)
    
    print('The image is classified as: ', str(prediction))
    return 0

#calling function
address=input("enter the address of the image: ")
address = os.path.normpath(address)
user_data=collect_data(address)
classification(user_data)

