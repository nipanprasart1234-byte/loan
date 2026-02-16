# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:02:53 2026

@author: Lab
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#import model
loan_model = pickle.load(open("C:/Users/Lab/Desktop/ML/loan_model.sav",'rb'))
heart_model = pickle.load(open("C:/Users/Lab/Desktop/ML/heartmodel.sav",'rb'))

with st.sidebar:
    selcted = option_menu('Loan and heart prediction',
                          ['Loan','heart'],
                          default_index=0)
    
if(selcted == 'Loan'):
    st.title("Loan Prediction")
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
        loan_accept = loan_model.predict([[
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
        ]])
        if(loan_accept[0]== 0):
            loan_accept = 'not accept'
        else:
            loan_accept = 'Accept'
    st.success(loan_accept)
    
if(selcted == 'heart'):
    st.title('Heart Prediction')
    age = st.text_input('age')
    sex = st.text_input('sex')
    cp = st.text_input('cp')
    trestbps = st.text_input('trestbps')
    chol = st.text_input('chol')
    fbs = st.text_input('fbs')
    restecg = st.text_input('restecg')
    thalach = st.text_input('thalach')
    exang = st.text_input('exang')
    oldpeak = st.text_input('oldpeak')
    slope = st.text_input('slope')
    ca = st.text_input('ca')
    thal = st.text_input('thal')
    
    #predict
    heart_predict = ''
    
    if st.button('Heart Prediction'):
        heart_predict = heart_model.predict([[
            float(age),
            float(sex),
            float(cp),
            float(trestbps),
            float(chol),
            float(fbs),
            float(restecg),
            float(thalach),
            float(exang),
            float(oldpeak),
            float(slope),
            float(ca),
            float(thal)
        ]])
        
        if(heart_predict[0]== 0):
            heart_predict = 'not have heart Disease'
        else:
            heart_predict = 'You have haert Disease'
    st.success(heart_predict)