import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np


def load_img(img):
    image_loaded = Image.open(img)
    image_arr = np.array(image_loaded.convert("RGB"))
    return image_arr


def compare_image(img_father, img_mother, img_child):
    distance_metrics = "cosine"
    model_name = "Facenet"

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

    st.bar_chart()
    st.subheader(f"These are the {distance_metrics} distances:")
    st.write(f"Father-child: {paternal_result['distance']}")
    st.write(f"Mother-child: {maternal_result['distance']}")

    if maternal_result["distance"] < paternal_result["distance"]:
        st.success("The child looks more like mother.")
    else:
        st.success("The child looks more like father.")

    return maternal_result["distance"], paternal_result["distance"]


st.set_page_config(layout="wide", page_title="Family Facial Comparison")

img_father = st.file_uploader("Upload an image for father", type=["png", "jpg", "jpeg"])
img_mother = st.file_uploader("Upload an image for mother", type=["png", "jpg", "jpeg"])
img_child = st.file_uploader("Upload an image for child", type=["png", "jpg", "jpeg"])

if img_father is not None and img_mother is not None and img_child is not None:
    compare_image(img_father, img_mother, img_child)
