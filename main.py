import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

tahlil=st.sidebar.text_area("Tahlil Sonucunu Giriniz")


modelsec=st.sidebar.selectbox("Model Seç",
                              ["Decision Tree","Random Forest"])

trainsize=st.sidebar.number_input("Train Size",
                                  min_value=0.0,max_value=1.0,value=0.75)

if modelsec=="Random Forest":
    agac=st.sidebar.number_input("Ağaç Sayısı",value=100)
else:
    dallanma=st.sidebar.number_input("Dal Sayısı",value=5)

getir=st.sidebar.button("Getir")

if getir:
    tahlil=tahlil.split(",")
    tahlil=np.array([tahlil])

    df=pd.read_csv('cancer.csv')
    df=df.drop("id",axis=1)
    df['diagnosis']=np.where(df['diagnosis']=="M",1,0)
    #iyi huylu 0 kötü huylu 1
    y=df[['diagnosis']]
    x=df.drop("diagnosis",axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=trainsize,random_state=20)
    if modelsec=="Decision Tree":
        tree=DecisionTreeClassifier(max_depth=dallanma)
        model=tree.fit(x_train,y_train)
        skor=model.score(x_test,y_test)
        sonuc=model.predict(tahlil)
    else:
        orman=RandomForestClassifier(n_estimators=agac)
        omodel=orman.fit(x_train,y_train)
        skor=omodel.score(x_test,y_test)
        sonuc =omodel.predict(tahlil)

    if sonuc[0]==0:
        mesaj="İyi Huylu"
        st.success(mesaj)
    elif sonuc[0]==1:
        mesaj="Kötü Huylu"
        st.error(mesaj)

