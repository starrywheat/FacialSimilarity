import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np
import pandas as pd
import plotly.express as px


def load_img(img):
    image_loaded = Image.open(img)
    image_arr = np.array(image_loaded.convert("RGB"))
    return image_arr


def charts(paternal_result, maternal_result):
    data = {
        "parent": ["Father", "Mother"],
        "feature": ["face", "face"],
        "similarity": [
            -1 + paternal_result["distance"],
            1 - maternal_result["distance"],
        ],
    }
    chart_data = pd.DataFrame(data=data)
    fig = px.bar(
        chart_data,
        x="similarity",
        y="feature",
        color="parent",
        title="Similarity",
        orientation="h",
        height=300,
    )
    st.plotly_chart(fig, theme="streamlit")


def compare_image(
    img_father,
    img_mother,
    img_child,
    distance_metrics: str = "cosine",
    model_name: str = "Facenet",
):

    st.subheader("These are the uploaded pictures")
    st.image(
        [img_father, img_child, img_mother],
        caption=["Father", "Child", "Mother"],
        width=200,
    )
    with st.spinner("Analyzing the images..."):
        try:
            paternal_result = DeepFace.verify(
                img1_path=load_img(img_father),
                img2_path=load_img(img_child),
                model_name=model_name,
                detector_backend="mtcnn",
                distance_metric=distance_metrics,
            )
        except ValueError:
            st.error(
                "Something wrong with either the father/child picture. Upload them again"
            )

        try:
            maternal_result = DeepFace.verify(
                img1_path=load_img(img_mother),
                img2_path=load_img(img_child),
                model_name=model_name,
                detector_backend="mtcnn",
                distance_metric=distance_metrics,
            )
        except ValueError:
            st.error(
                "Something wrong with either the mother/child picture. Upload them again"
            )

    charts(paternal_result, maternal_result)

    if maternal_result["distance"] < paternal_result["distance"]:
        st.success("The child looks more like mother.")
    else:
        st.success("The child looks more like father.")

    return maternal_result["distance"], paternal_result["distance"]


st.set_page_config(layout="wide", page_title="Family Facial Comparison")

img_father = st.file_uploader("Upload an image for father", type=["png", "jpg", "jpeg"])
img_mother = st.file_uploader("Upload an image for mother", type=["png", "jpg", "jpeg"])
img_child = st.file_uploader("Upload an image for child", type=["png", "jpg", "jpeg"])

distance_metrics = st.sidebar.radio(
    "Distance Metrics", ["cosine", "euclidean", "euclidean_l2"], index=0
)
model = st.sidebar.selectbox(
    "Image Models",
    [
        "Facenet",
        "VGG-Face",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ],
)
if img_father is not None and img_mother is not None and img_child is not None:
    compare_image(
        img_father,
        img_mother,
        img_child,
        distance_metrics=distance_metrics,
        model_name=model,
    )
