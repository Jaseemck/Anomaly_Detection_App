import streamlit as st
import pandas as pd
import os
from PIL import Image,ImageEnhance
import os, urllib
import pickle
import sklearn
import numpy as np

def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    #href = ''' <a href="https://github.com/Jaseemck/Anomaly_Detection_App/raw/master/paper.pdf" download="paper">Download Base Paper</a>'''
    #st.write(href,unsafe_allow_html=True)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Anomaly Detection in IoT Devices")
    st.sidebar.title("Group-4")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "EDA of Kaggle dataset", "EDA of KDD cup dataset", "Show the source code","Temp Anomaly Detection"])
    if app_mode == "Show instructions":
        st.sidebar.success('To view EDA of datasets, select "EDA of {} dataset".')
        st.sidebar.success('To view the sourcecode of the file, select "Show the Source Code".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("anomaly_detection_fl.py"))
    elif app_mode == "EDA of Kaggle dataset":
        readme_text.empty()
        eda_kaggle()
    elif app_mode == "EDA of KDD cup dataset":
        readme_text.empty()
        eda_kdd()
    elif app_mode == "Temp Anomaly Detection":
        readme_text.empty()
        st.sidebar.info('The Temperature Sensor data is taken from Raspberry pi. The anomalous temperatures are added manually.')
        tmp_anm()
    
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,11)
    loaded_model = pickle.load(open("model_demo.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def tmp_anm():
    st.title("Temperature Reading Anomaly Detection Taken from Raspberry Pi")
    st.subheader("Group-4")
    st.write("hai")
    
    html_temp = """
    <div style="background-color:tomato;padding:15px;">
    <h2> Classification App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    #temp = st.number_input("Enter Temp")
    temp = st.slider("Set Temperature",-100,200)
    to_predict_list= [61,4,8,11,2,6,11,5,11,1,temp]
    to_predict_list = list(map(float, to_predict_list))
    result = ValuePredictor(to_predict_list)
    prediction = str(result)

    if st.button("Detect"):
        st.text("Normality is {}".format(prediction.title()))

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Jaseemck/Anomaly_Detection_App/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def eda_kaggle():
    st.title("Anomaly Detection in IoT Devices")
    st.header("Explorative Data Analysis")
    st.subheader("Kaggle IoT Dataset")
    dataset1 = 'kaggle_iot_dataset.csv'
    @st.cache(persist=True)
    def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df

    data = explore_data(dataset1)

    if st.checkbox("Preview Dataset"):
        if st.button("Head"):
            st.write(data.head(10))
        elif st.button("Tail"):
            st.write(data.tail(10))
        else: 
            st.write(data.head(3))

    if st.checkbox("Show Column Names"):
        st.write(data.columns)

    data_dim = st.radio("What dimensions do you want to see?",("Rows","Columns"))
    if data_dim == 'Rows':
        st.text("Showing Rows")
        st.write(data.shape[0])
    elif data_dim == 'Columns':
        st.text("Showing Columns")
        st.write(data.shape[1])
    else:
        st.text("Shape of data")
        st.write(data.shape)

    col_option = st.selectbox("Select Column",("sourceType","sourceLocation","destinationServiceAddress","destinationServiceType","destinationLocation","accessedNodeAddress","accessedNodeType","operation","value","normality"))
    if col_option == 'sourceType':
        st.write(data['sourceType'].unique())
    elif col_option == 'sourceLocation':
        st.write(data['sourceLocation'].unique())
    elif col_option == 'destinationServiceAddress':
        st.write(data['destinationServiceAddress'].unique())
    elif col_option == 'destinationServiceType':
        st.write(data['destinationServiceType'].unique())
    elif col_option == 'destinationLocation':
        st.write(data['destinationLocation'].unique())
    elif col_option == 'accessedNodeAddress':
        st.write(data['accessedNodeAddress'].unique())
    elif col_option == 'accessedNodeType':
        st.write(data['accessedNodeType'].unique())
    elif col_option == 'operation':
        st.write(data['operation'].unique())
    elif col_option == 'value':
        st.write(data['value'].unique())
    elif col_option == 'normality':
        st.write(data['normality'].unique())
    else:
        st.write("Select Column")

    st.subheader("Visualization")
    if st.checkbox("Occurences of Source Location"):
        src_loc_img = Image.open("source_location.png")
        src_loc_width = st.slider("Set Image Width of src_loc Graph",300,500)
        st.image(src_loc_img,width=src_loc_width)
    if st.checkbox("Occurences of Source Type"):
        src_type_img = Image.open("source_type.png")
        src_type_width = st.slider("Set Image Width of src_type Graph",300,500)
        st.image(src_type_img,width=src_type_width)
    if st.checkbox("Occurences of Destination Location"):
        des_loc_img = Image.open("destination_location.png")
        des_loc_width = st.slider("Set Image Width of Dest_loc Graph",300,500)
        st.image(des_loc_img,width=des_loc_width)
    if st.checkbox("Occurences of Destination Service"):
        des_ser_img = Image.open("destination_service.png")
        des_ser_width = st.slider("Set Image Width of Dest_srv Graph",300,500)
        st.image(des_ser_img,width=des_ser_width)
    if st.checkbox("Occurences of Normality"):
        norm_img = Image.open("normality.png")
        norm_width = st.slider("Set Image Width of nrm Graph",300,500)
        st.image(norm_img,width=norm_width)

#------------------------------------------------------------------------------------

def eda_kdd():
    st.title("Anomaly Detection in IoT Devices")
    st.header("Explorative Data Analysis")
    st.subheader("KDD Dataset")

    
    @st.cache(persist=True)
    def explore_kdd():
        colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']
        df = pd.read_csv("kddcup.data_10_percent.gz",names=colnames+["threat_type"])[:100000]
        return df

    kdd_data = explore_kdd()


    if st.checkbox("Preview Dataset"):
        if st.button("Head"):
            st.write(kdd_data.head(10))
        elif st.button("Tail"):
            st.write(kdd_data.tail(10))
        else: 
            st.write(kdd_data.head(3))

    if st.checkbox("Show Column Names"):
        st.write(kdd_data.columns)

    data_dim = st.radio("What dimensions do you want to see?",("Rows","Columns"))
    if data_dim == 'Rows':
        st.text("Showing Rows")
        st.write(kdd_data.shape[0])
    elif data_dim == 'Columns':
        st.text("Showing Columns")
        st.write(kdd_data.shape[1])
    else:
        st.text("Shape of data")
        st.write(kdd_data.shape)

    col_option = st.selectbox("Select Column",("protocol_type","service","flag","src_bytes","dst_bytes","threat_type"))
    if col_option == 'protocol_type':
        st.write(kdd_data['protocol_type'].unique())
    elif col_option == 'service':
        st.write(kdd_data['service'].unique())
    elif col_option == 'flag':
        st.write(kdd_data['flag'].unique())
    elif col_option == 'src_bytes':
        st.write(kdd_data['src_bytes'].unique())
    elif col_option == 'dst_bytes':
        st.write(kdd_data['dst_bytes'].unique())
    elif col_option == 'threat_type':
        st.write(kdd_data['threat_type'].unique())
    else:
        st.write("Select Column")


    st.subheader("Visualization")
    if st.checkbox("Occurences of Attacks"):
        attacks_img = Image.open("attacks.png")
        attacks_width = st.slider("Set Image Width of attacks Graph",500,800)
        st.image(attacks_img,width=attacks_width)

    
if __name__ == "__main__":
    main()

