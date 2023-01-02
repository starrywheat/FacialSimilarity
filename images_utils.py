import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deepface import DeepFace
from deepface.commons import distance as dst
from PIL import Image


def load_img(img: any) -> np.array:
    """
    This function loads the uploaded img obj and turn to np.array

    Returns:
        img_arr (np.array): img in numpy array
    """
    if isinstance(img, Image.Image):
        image_arr = np.array(img.convert("RGB"))
    else:
        image_loaded = Image.open(img)
        image_arr = np.array(image_loaded.convert("RGB"))
    return image_arr


def run_deepface(
    img1: any, img2: any, feature: str, model_name: str, distance_metrics: str
) -> dict:
    """
    This function use deepface.verify (whole face) or deepface.represent (facial features)
    to compare the img1 and img2 to get similarity

    Args:
        img1 (any): image 1
        img2 (any): image 2
        feature (str): name of the facial feature
        model_name (str): name of the deepL model for factial detection
        distance_metrics (str): name of the distance metrics

    Returns:
        result (dict): result of the analysis
    """
    if feature == "whole face":
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=model_name,
                detector_backend="mtcnn",
                distance_metric=distance_metrics,
            )
            return result
        except ValueError:
            return None
    else:
        try:
            img1_representation = DeepFace.represent(
                img1,
                enforce_detection=False,
                model_name=model_name,
                detector_backend="mtcnn",
            )
            img2_representation = DeepFace.represent(
                img2,
                enforce_detection=False,
                model_name=model_name,
                detector_backend="mtcnn",
            )
            if distance_metrics == "cosine":
                distance = dst.findCosineDistance(
                    img1_representation, img2_representation
                )
            elif distance_metrics == "euclidean":
                distance = dst.findEuclideanDistance(
                    img1_representation, img2_representation
                )
            elif distance_metrics == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(img1_representation),
                    dst.l2_normalize(img2_representation),
                )
            return {"distance": np.float64(distance)}
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
        credit = " (Credits: The Guardian)"
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


def charts(deepface_results: dict):
    """
    This function create a bar chart based on the analysis result

    Args:
        deepface_results (dict): deepface analysis result
        distance_metrics (str): name of the distance metrics used
    """

    chart_data = pd.DataFrame(data=deepface_results)

    feature_emoji = {
        "nose": "nose 👃",
        "eyes": "eyes 👀",
        "mouth": "mouth 👄",
        "whole face": "whole face 😐",
    }
    chart_data["feature"] = chart_data["feature"].apply(lambda x: feature_emoji[x])

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


def compare_whole_face(
    img_father: any,
    img_mother: any,
    img_child: any,
    distance_metrics: str = "cosine",
    model_name: str = "Facenet",
) -> list:

    results = []
    feature = "whole face"
    # whole face
    print("Running on whole face")
    paternal_result = run_deepface(
        load_img(img_father), load_img(img_child), feature, model_name, distance_metrics
    )
    maternal_result = run_deepface(
        load_img(img_mother), load_img(img_child), feature, model_name, distance_metrics
    )
    simscore = distance2score(
        paternal_result["distance"], maternal_result["distance"], distance_metrics
    )
    results.append(
        {
            "parent": "Father",
            "feature": feature,
            "similarity": simscore[0],
            "distance": paternal_result["distance"],
        }
    )
    results.append(
        {
            "parent": "Mother",
            "feature": feature,
            "similarity": simscore[1],
            "distance": maternal_result["distance"],
        }
    )
    return results


# def compare_facial_features(
#     img_father: any,
#     img_mother: any,
#     img_child: any,
#     features: list,
#     distance_metrics: str = "cosine",
#     model_name: str = "Facenet",
# ):
#     results = []
#     if features != []:
#         # Detect the feature landmarks
#         facial_landmarks_father = detect_facial_landmarks(img_father)
#         facial_landmarks_mother = detect_facial_landmarks(img_mother)
#         facial_landmarks_child = detect_facial_landmarks(img_child)

#         # Get cropped image of the feature
#         for feature in features:
#             print(f"Running on {feature}")
#             cropped_father = crop_img(
#                 Image.open(img_father), facial_landmarks_father[0], feature=feature
#             )
#             cropped_mother = crop_img(
#                 Image.open(img_mother), facial_landmarks_mother[0], feature=feature
#             )
#             cropped_child = crop_img(
#                 Image.open(img_child), facial_landmarks_child[0], feature=feature
#             )
#             paternal_result = run_deepface(
#                 load_img(cropped_father),
#                 load_img(cropped_child),
#                 feature,
#                 model_name,
#                 distance_metrics,
#             )
#             maternal_result = run_deepface(
#                 load_img(cropped_mother),
#                 load_img(cropped_child),
#                 feature,
#                 model_name,
#                 distance_metrics,
#             )
#             simscore = distance2score(
#                 paternal_result["distance"],
#                 maternal_result["distance"],
#                 distance_metrics,
#             )
#             results.append(
#                 {
#                     "parent": "Father",
#                     "feature": feature,
#                     "similarity": simscore[0],
#                     "distance": paternal_result["distance"],
#                 }
#             )
#             results.append(
#                 {
#                     "parent": "Mother",
#                     "feature": feature,
#                     "similarity": simscore[1],
#                     "distance": maternal_result["distance"],
#                 }
#             )
#     return results


def compare_image(
    img_father: any,
    img_mother: any,
    img_child: any,
    features: list,
    distance_metrics: str = "cosine",
    model_name: str = "Facenet",
) -> list:
    """
    This function run analysis to compare the parent-child images.
    After successful run of the comparisons, this function also plot a bar chart
    to display the result visually

    Args:
        img_father (any): streamlit file uploader obj for father
        img_mother (any): streamlit file uploader obj for mother
        img_child (any):  streamlit file uploader obj for child
        features (list): list of the facial feature name for comparisons
        distance_metrics (str, optional): name of the distance metrics used for comparison. Defaults to "cosine".
        model_name (str, optional): name of the model used for facial detection. Defaults to "Facenet".

    Returns:
        paternal_result, maternal_result: analysis results
    """
    # Run the analysis
    results = []
    with st.spinner("Analyzing the images..."):
        result_wf = compare_whole_face(
            img_father,
            img_mother,
            img_child,
            distance_metrics=distance_metrics,
            model_name=model_name,
        )
        # streamlit cloud don't support this :(
        # result_ff = compare_facial_features(
        #     img_father,
        #     img_mother,
        #     img_child,
        #     features,
        #     distance_metrics=distance_metrics,
        #     model_name=model_name,
        # )

    results = result_wf  # + result_ff

    if results != {}:
        # Plot the results
        charts(results)
        # Show the whole face result
        if result_wf[0]["distance"] > result_wf[1]["distance"]:
            st.success("The child looks more like mother.")
        else:
            st.success("The child looks more like father.")

        return results
    else:
        st.error(
            "The faces are not detected properly. You can do better. Upload your best shots! 😜"
        )
