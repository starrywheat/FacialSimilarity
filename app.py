import streamlit as st

from images_utils import compare_image
from images_utils import show_img


def add_click():
    st.session_state.clicks += 1


st.set_page_config(layout="wide", page_title="Family Facial Comparison")

st.header("Whom does your child look like? 👨‍👩‍👧‍👦")
if "clicks" not in st.session_state:
    st.session_state.clicks = 0

# Display settings in sidebar
with st.sidebar:
    st.subheader("Choose the following to compare")
    eyes = st.checkbox("Eyes", value=True)
    mouth = st.checkbox("Mouth", value=True)
    nose = st.checkbox("Nose", value=True)
    st.warning("Change the following settings only if you understand what they are!")
    distance_metrics = st.radio(
        "Distance Metrics",
        ["cosine", "euclidean"],
        index=0,
        help="Cosine: Cosine distance which measure the angular distance between the images.\nEucliean: distance between 2 images in Euclidean space. ",
    )
    model = st.selectbox(
        "Image Models",
        [
            "Facenet",
            "VGG-Face",
            "ArcFace",
        ],
        help="Face recognition models provided by deepface. These are the 3 top performing models in this task.",
    )
    url_df = "https://github.com/serengil/deepface"
    url_ff = (
        "https://face-recognition.readthedocs.io/en/latest/readme.html#installation"
    )
    st.markdown(
        f"This app uses [deepface library]({url_df}) and [facial_recongition]({url_ff})."
    )

click = st.button("Try it yourself.", on_click=add_click)

# Ask for the images. Run a sample in the begining
img_father, img_mother, img_child = show_img(sample=st.session_state["clicks"] == 0)

# Compare the images and display the result
if img_father is not None and img_mother is not None and img_child is not None:
    _ = compare_image(
        img_father,
        img_mother,
        img_child,
        [x for x in ["eyes", "mouth", "nose"] if eval(x)],
        distance_metrics=distance_metrics,
        model_name=model,
    )
