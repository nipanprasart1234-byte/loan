# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:08:41 2026

@author: Lab
"""

import numpy as np
import pickle
import streamlit as st

#import model
loan_model = pickle.load(open("C:/Users/Lab/Desktop/ML/loan_model.sav",'rb'))

def loan_prediction(inputdata):
    
    
    input_data = np.array(inputdata,dtype=float)
    input_data_reshape = input_data.reshape(1,-1)
    prediction = loan_model.predict(input_data_reshape)

    if(prediction[0]==0):
      return 'Not Accept'
    else:
      return 'Accept'
      
    
def main():
    
    #title
    st.title('Loan Prediction')
    
    #input data
    person_age = st.text_input('person_age')
    person_gender = st.text_input('person_gender')
    person_education = st.text_input('person_education')
    person_income = st.text_input('person_income')
    person_emp_exp = st.text_input('person_emp_exp')
    person_home_ownership = st.text_input('person_home_ownership')
    loan_amnt = st.text_input('loan_amnt')
    loan_intent = st.text_input('loan_intent')
    loan_int_rate = st.text_input('loan_int_rate')
    loan_percent_income = st.text_input('loan_percent_income')
    cb_person_cred_hist_length = st.text_input('cb_person_cred_hist_length')
    credit_score = st.text_input('credit_score')
    previous_loan_defaults_on_file = st.text_input('previous_loan_defaults_on_file')
    
    #predict
    loan_accept = ''
    
    if st.button('Loan accept/not test'):
        loan_accept = loan_prediction([
            float(person_age),
            float(person_gender),
            float(person_education),
            float(person_income),
            float(person_emp_exp),
            float(person_home_ownership),
            float(loan_amnt),
            float(loan_intent),
            float(loan_int_rate),
            float(loan_percent_income),
            float(cb_person_cred_hist_length),
            float(credit_score),
            float(previous_loan_defaults_on_file)
        ])
    st.success(loan_accept)
    
if __name__=='__main__':
    main()