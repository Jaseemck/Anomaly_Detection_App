import streamlit as st

st.title("Anomaly Detection in IoT Devices")
st.header("EDA")
st.subheader("Group-4")

st.markdown("### This is a Markdown")
st.success("Successful")
st.info("Information!")
st.warning("Warning")
st.error("Error")
st.exception("NameError('name three not defined')")


st.write("Text with write")
st.write(range(10))

from PIL import Image
img = Image.open("logo.png")
st.image(img,width=300,caption="LOGO")

if st.checkbox("Show/Hide"):
    st.text("Showing or Hiding Widget")

status = st.radio("What is your status",("Active","Inactive"))
if status == 'Active':
    st.success("You are Active")
else:
    st.warning("Inactive, Activate")

occupation = st.selectbox("Your Occupation", ["Programmer","Datatscientist","Business man"])
st.write("You selectes", occupation)

location = st.multiselect("Where do you work?", ["London", "NewYork", "Delhi"])
st.write("You selected", len(location), "locations")

level = st.slider("What is your level",1,5)

st.button("Simple Button")
if st.button("About"):
    st.text("Streamlit is cool")

st.text("Display Raw Code")
st.code("import numpy as np")

import time
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p+1)

with st.spinner("Waiting..."):
    time.sleep(5)
st.success("Finished!")

st.balloons()


st.sidebar.header("About")
st.sidebar.text("This is shit")

@st.cache
def run_fxn():
    return range(100)
st.write(run_fxn())

st.pyplot
