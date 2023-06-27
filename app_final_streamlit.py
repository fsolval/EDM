#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mon 5 23:56:08 2023

@author: albitadubonllamas
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
import pandas as pd
import folium
from streamlit_folium import folium_static
import json
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from streamlit_option_menu import option_menu



selected = option_menu(
    menu_title = None,
    options = ['Data Info.', 'Prediction Info.'],
    icons = ['data', 'diagram-2-fill'],
    default_index=0,
    orientation= 'horizontal',
)


df = pd.read_csv('vulnerabilidad.csv', delimiter=';')
df.head()
df = df.rename(columns={'Index_Gl_1': 'vulnerabilidad'})
df = df.dropna()

header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()
prediction = st.container()




if selected=='Data Info.':
    with header:
        st.title('*EDM* - VULNERABILITY IN NEIGHBOURHOODS')
        st.write('The  "vulnerability"  indicates the level of social, economic or environmental vulnerability of each neighbourhood. This implies that neighborhoods with greater vulnerability value may face greater challenges and difficulties in meeting their basic needs, as well as greater exposure to risks and shortages compared to ')
        st.write('Our aim in this project is to develop a predictive model that is able to predict wether a neighborhood is vulnerable or not. We will do so using a training dataset offered by the Valencian government')
        
        
    with datasets:
        st.header('DATA')
        st.write('**Description:** Geographical information of the neighborhoods of Valencia, with the result of poverty indices and the assessment of their vulnerability')
        columnas= df.columns
        df_seleccionado = df[columnas]
        st.write(df_seleccionado.head())
        
        
        st.subheader('Exploration')
        pull_counts= pd.DataFrame(df['vulnerabilidad'].value_counts())
    
        st.bar_chart(pull_counts)
        
        
    with features:
        
        #st.header('')
        sel_col0, disp_col0 = st.columns(2)
        selected_pair = sel_col0.selectbox('Selecciona un par de variables', options=[('Densitat_p', 'risc_pobre'), ('Densitat_p', 'turismes_e'), ('Densitat_p', 'renda_mitj'), ('renda_mitj', 'Zones verd')])
        
        fig, ax = plt.subplots()
        ax.scatter(df[selected_pair[0]], df[selected_pair[1]])
        ax.set_xlabel(selected_pair[0])
        ax.set_ylabel(selected_pair[1])
        ax.set_title(f'Gráfico de dispersión: {selected_pair[0]} vs {selected_pair[1]}')
        st.pyplot(fig)
        


if selected=='Prediction Info.':       

    with model_training:
        st.title('TRAINING THE DATA')
    
        st.write('In this part you can choose whit which model you would like to train the data. You will also have the choice for the hyperparameters:')
        
        
        sel_col, disp_col = st.columns(2)
        
        model = sel_col.selectbox('Number of trees', options=['Random Forest', 'Decision Tree', 'Stacking', 'Boosting'], index=0)
        
        
        
        if model== 'Random Forest':
            
            st.write('**Adjust the hyperparameters for RandomForest model:**')
            max_depth = sel_col.slider('Maxdepth of the model', min_value=5, max_value=25)
        
            n_estimators = sel_col.selectbox('Number of trees', options=[100, 200, 300, 'No limit'], index=0)
            
            
            
            if n_estimators == 'No limit':
                model = RandomForestClassifier(random_state=42, max_depth=max_depth)
            else:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth)
            
        
        if model == 'Decision Tree':
            
            st.write('**Adjust the hyperparameters for Decision Tree model:**')
            maxd = sel_col.slider('Maxdepth of the model', min_value=5, max_value=25)
            crit = sel_col.selectbox('Criterion messure:', options=['gini', 'entropy'])
            model = DecisionTreeClassifier(max_depth=maxd, criterion=crit)
        
        
        if model == 'Stacking':
            
            st.write('We will use as base models Random Forest, Logistic Regression and KNeighbors')
            
            meta = sel_col.selectbox('Would you like to use a Meta Model?:', options=['Yes', 'No'])
            
            model1 = RandomForestClassifier(random_state=42)
            model2 = LogisticRegression(random_state=42)
            model3 = KNeighborsClassifier()
    
            if meta == 'Yes':
                meta_model = LogisticRegression()
                model = StackingClassifier(estimators=[('rf', model1), ('lr', model2), ('knn', model3)], final_estimator=meta_model)
            
            if meta == 'No':
                meta_model = LogisticRegression()
                model = StackingClassifier(estimators=[('rf', model1), ('lr', model2), ('knn', model3)])
    
        
        if model == 'Boosting':
            
            
            st.write('We will use as base models Random Forest, Logistic Regression and KNeighbors')
               
            base_model = sel_col.selectbox('Which base model would you like to use', options=['Decision Tree', 'SVC'])
            
            if base_model == 'Decision Tree':
                base = DecisionTreeClassifier(max_depth=2)
                model = AdaBoostClassifier(base_estimator=base, n_estimators=100, random_state=42)
           
            
           
            elif base_model == 'SVM':
                base = SVC(kernel='linear', probability=True, random_state=42)
                model = AdaBoostClassifier(base_estimator=base, algorithm='SAMME', n_estimators=100, random_state=42)

        
        X = df.iloc[:,6:15]
        y = df[['vulnerabilidad']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Precission:", accuracy)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        st.write("Balanced precission:", balanced_accuracy)
        #confusion_mat = confusion_matrix(y_test, y_pred)

    
        #st.write("Matriz de confusión:")
        #st.write(confusion_mat)
        

    
      
    with prediction:
        st.header('OBTAIN YOUR PREDICTION')
        st.write("We have already created a predictive model. Now you can give some information about a neighbourhood of your interest and see if it's vulnerable or not")
        
        verdes = st.number_input('How many m2 of green areas your neighborhood has?:')
        densidad = st.number_input('Which is the population density of your neighborhood?:')
        #turismos = st.number_input('How many m2 of green areas your neighborhood has?:')
        #parados = st.number_input('How many m2 of green areas your neighborhood has?:')
       # renta = st.number_input('How many m2 of green areas your neighborhood has?:')
       # pobreza = st.number_input('How many m2 of green areas your neighborhood has?:')
       # equidad = st.number_input('How many m2 of green areas your neighborhood has?:')
        #social = st.number_input('How many m2 of green areas your neighborhood has?:')
        
        
        
        prediccion = model.predict(df.iloc[2:3,6:15].values)
    
        st.write('The prediction for the new observation is:', prediccion)
        
        


               
    
