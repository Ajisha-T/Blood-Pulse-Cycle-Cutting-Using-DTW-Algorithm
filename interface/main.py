import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt



from cycle_cutting_function import cycle_cutting

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

st.set_page_config(layout="wide", page_title="DTW cycle cutting")

st.sidebar.title(" DTW Based Cycle Cutting Algorithm")
st.empty()
st.write(
    "Upload a heart beat template and a sequence of heartbeat signals to find the good and bad cycles. this algorithm uses Dynamic Time warping for similarity search in the uploaded sequence using the given mother wavelet"
)

st.sidebar.write("## Upload data")

   
def btClick():
    if signal is not None and template is not None:
        cycle_cutting(signal,template)
        
        container1 = st.container()
        container1.subheader("Input signal")
        container1.image("input.png")

        with st.container():
            col1, col2 = st.columns(2)
        with col1:
            st.write("Detected Good Signals")
            st.image("good.png")
        with col2:
            st.write("Detected Bad Signals")
            st.image("bad.png")

        container2 = st.container()
        container2.title("Good and Bad Signals")
        container2.image("output.png")
        output = Image.open('output.png')
        st.download_button("Download this image", convert_image(output), "output.png", "image/png")

    


template = st.sidebar.file_uploader("Upload template file", type=["txt"])
signal = st.sidebar.file_uploader("Upload cycle", type=["txt"],key='32')
st.sidebar.button("Process Files",on_click=btClick)







