import streamlit as st
import os
import numpy as np
import pickle
import sklearn

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,11)
    loaded_model = pickle.load(open("model_demo.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def main():
    st.title("Anomaly Detection in IoT Devices")
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


if __name__ == '__main__':
    main()

