import streamlit as st 
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor ")

# Brand
company = st.selectbox('Brand',df['Company'].unique())

# Type of Laptop 
Type = st.selectbox('Type',df['TypeName'].unique())

# Ram 
Ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
Weight = st.number_input('Weight of the Laptop')

# Touchscreen
Touchscreen = st.selectbox('Touchsreen',['NO','YES'])

# IPS 
ips = st.selectbox('IPS',['NO','YES'])

# Screen Size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU',df['Cpu_brand'].unique())

# HDD 
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#SSD
ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024])

# Gpu
gpu = st.selectbox('GPU',df['Gpu_brand'].unique())

# OS

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # query
    
    if Touchscreen == 'YES':
        Touchscreen = 1
    else: 
        Touchscreen = 0

    if ips == 'YES':
        ips = 1
    else: 
        ips = 0
    
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2)+(y_res**2))**0.5/screen_size
    query = np.array([company,Type,Ram,Weight,Touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is :" + (str(int(np.exp(pipe.predict(query)[0])))))


