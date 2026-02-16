# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


#import model
loan_model = pickle.load(open("C:/Users/Lab/Desktop/ML/loan_model.sav",'rb'))

inputdata = (22.0,0,4,71948.0,0,3,35000.0,4,16.02,0.49,3.0,561,0)
input_data = np.array(inputdata)
input_data_reshape = input_data.reshape(1,-1)

prediction = loan_model.predict(input_data_reshape)

if(prediction[0]==0):
  print('Not Accept')
else:
  print('Accept')