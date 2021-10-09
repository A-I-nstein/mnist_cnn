import cv2
import time
import shutil
import numpy as np
import streamlit as st
from tensorflow.keras import models
from streamlit_drawable_canvas import st_canvas

def get_prediction(image):

    image = pre_img(image)
    model = models.load_model('cnn_model.h5')
    prediction = model.predict(image)
    predicted_number = np.argmax(prediction)
    return predicted_number, prediction

def pre_img(img):

    img = cv2.resize(img.astype(np.uint8),(28,28),
        interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    predict_data = np.array([img])/255
    image = predict_data.reshape(1,28, 28, 1)
    return image

def main():

    # page configuration

    st.set_page_config(
    page_title = "Handwritten Letters Classifier",
    page_icon = ":pencil:"
    )

    # hide irrelevant stuffs

    hide_streamlit_style = """
                       <style>
                       #MainMenu {visibility: hidden;}
                       footer {visibility: hidden;}
                       </style>
                       """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # title and subheading

    st.markdown("<h1 style = 'text-align: center;'>Handwritten Number Classifier<h1>", unsafe_allow_html=True)
    st.markdown("<h3 style = 'text-align: center;'>Use this AMAZING app to classify Handwritten Numbers!!!<h3>", unsafe_allow_html=True)
    st.text("\n")
    st.text("\n")

    # configuring side bar

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 25)
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    explore_result = st.sidebar.checkbox("Explore result", False)

    # canvas to draw numbers

    canvas_result = st_canvas(
    stroke_width = stroke_width,
    stroke_color = "#fff",
    background_color = "#000",
    height = 300,
    width = 300,
    key = "canvas",
    update_streamlit=realtime_update,
    drawing_mode='freedraw'
    )

    # predict button

    st.text("\n")
    predict = st.button("Predict")

    # calling the predict function

    if canvas_result.image_data is not None and predict:
        st.text("Loading...")
        predicted_number, prediction = get_prediction(canvas_result.image_data)
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.001)
        st.markdown("<h3 style = 'text-align: center;'>Prediction : {}<h3>".format(predicted_number),
            unsafe_allow_html=True)
        if explore_result:
            st.bar_chart(prediction)

main()
