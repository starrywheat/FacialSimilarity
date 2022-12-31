import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deepface import DeepFace
from PIL import Image


def load_img(img: any) -> np.array:
    """
    This function loads the uploaded img obj and turn to np.array

    Returns:
        img_arr (np.array): img in numpy array
    """
    image_loaded = Image.open(img)
    image_arr = np.array(image_loaded.convert("RGB"))
    return image_arr


def run_deepface(img1: any, img2: any, model_name: str, distance_metrics: str) -> dict:
    """
    This function use deepface.verify to compare the img1 and img2 to get similarity

    Args:
        img1 (any): image 1
        img2 (any): image 2
        model_name (str): name of the deepL model for factial detection
        distance_metrics (str): name of the distance metrics

    Returns:
        result (dict): result of the analysis
    # TODO: calculate similarity for specific face features, e.g. eyes
    """
    try:
        result = DeepFace.verify(
            img1_path=load_img(img1),
            img2_path=load_img(img2),
            model_name=model_name,
            detector_backend="mtcnn",
            distance_metric=distance_metrics,
        )
        return result
    except ValueError:
        return None


def show_img(sample: bool = False) -> tuple:
    """
    This function displays the upload and loaded images.
    When the app first load, it will display sample family photos (Beckham family)
    After that, session_state['Init'] will set to false so that it can allow user to try their own.

    Args:
    sample (bool): If True, display the family pictures in sample

    Returns:
        img_father, img_mother, img_child: The files uploaded
    """
    # Display the uploaded images
    st.subheader("Upload pictures of your family to get started!")
    st.write("hint: For best results, use headshots.")
    st.write("This app doesn't store any your uploaded pictures.")

    col1, col2, col3 = st.columns(3)
    if sample:
        img_father = "data/father.jpeg"
        img_child = "data/child.jpeg"
        img_mother = "data/mother.jpeg"
        credit = " (Credits: IMDB)"
    else:
        img_father, img_child, img_mother = None, None, None
        credit = ""

    with col1:
        if img_father is None:
            img_father = st.file_uploader(
                "Upload an image for father 👨", type=["png", "jpg", "jpeg"]
            )
        if img_father is not None:
            st.image(img_father, caption="Father" + credit, width=200)
    with col2:
        if img_child is None:
            img_child = st.file_uploader(
                "Upload an image for child 🧒👧", type=["png", "jpg", "jpeg"]
            )
        if img_child is not None:
            st.image(img_child, caption="Child" + credit, width=200)
    with col3:
        if img_mother is None:
            img_mother = st.file_uploader(
                "Upload an image for mother 👩", type=["png", "jpg", "jpeg"]
            )
        if img_mother is not None:
            st.image(img_mother, caption="Mother" + credit, width=200)

    return img_father, img_mother, img_child


def distance2score(d1: float, d2: float, distance_metrics: str) -> list:
    """
    This function turn the distance metrics to a scaled score 0-100,
    which represent a rough estimate of the similiarity.

    Args:
        d1 (float): distance 1
        d2 (float): distance 2
        distance_metrics (str): name of the distance metrics

    Returns:
        list: the scaled scores
    """
    if distance_metrics == "cosine":
        s1 = 1.0 - d1
        s2 = 1.0 - d2
    else:
        s1 = 1.0 / (1.0 + d1)
        s2 = 1.0 / (1.0 + d2)

    score1 = round(s1 / (s1 + s2) * 100.0, 1)
    score2 = round(s2 / (s1 + s2) * 100.0, 1)

    return [score1, score2]


def charts(paternal_result: dict, maternal_result: dict, distance_metrics: str):
    """
    This function create a bar chart based on the analysis result

    Args:
        paternal_result (dict): father analysis result
        maternal_result (dict): mother analysis result
        distance_metrics (str): name of the distance metrics used
    """
    simscore = distance2score(
        paternal_result["distance"], maternal_result["distance"], distance_metrics
    )
    data = {
        "parent": ["Father", "Mother"],
        "feature": ["face", "face"],
        "distance": [paternal_result["distance"], maternal_result["distance"]],
        "similarity": simscore,
    }
    chart_data = pd.DataFrame(data=data)

    fig = px.bar(
        chart_data,
        x="similarity",
        y="feature",
        color="parent",
        title="Similarity",
        orientation="h",
        height=250,
    )
    st.plotly_chart(fig, theme="streamlit")


def compare_image(
    img_father: any,
    img_mother: any,
    img_child: any,
    distance_metrics: str = "cosine",
    model_name: str = "Facenet",
) -> tuple:
    """
    This function run analysis to compare the parent-child images.
    After successful run of the comparisons, this function also plot a bar chart
    to display the result visually

    Args:
        img_father (any): streamlit file uploader obj for father
        img_mother (any): streamlit file uploader obj for mother
        img_child (any):  streamlit file uploader obj for child
        distance_metrics (str, optional): name of the distance metrics used for comparison. Defaults to "cosine".
        model_name (str, optional): name of the model used for facial detection. Defaults to "Facenet".

    Returns:
        paternal_result, maternal_result: analysis results
    """
    # Run the analysis
    with st.spinner("Analyzing the images..."):
        paternal_result = run_deepface(
            img_father, img_child, model_name, distance_metrics
        )
        maternal_result = run_deepface(
            img_mother, img_child, model_name, distance_metrics
        )

    # Show the results
    if paternal_result is not None and maternal_result is not None:
        charts(paternal_result, maternal_result, distance_metrics=distance_metrics)

        if maternal_result["distance"] < paternal_result["distance"]:
            st.success("The child looks more like mother.")
        else:
            st.success("The child looks more like father.")

        return maternal_result["distance"], paternal_result["distance"]
    else:
        st.error("Something wrong with either the pictures. Upload them again")
