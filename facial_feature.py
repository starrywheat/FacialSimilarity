import face_recognition
import numpy as np
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


def crop_img(
    image: Image.Image, face_landmarks: dict, feature: str = "right_eye"
) -> Image.Image:
    """
    This function cropped the image to contain only the specified facial feature.
    The features supported are:
    individual -
    'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
    'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip',
    combined -
    'eye', 'mouth', 'nose'

    Args:
        image (Image.Image): input image
        face_landmarks (dict): facial feature coord dict, detected by facial_recognition
        feature (str, optional): name of the facial feature. Defaults to 'right_eye'.

    Returns:
        Image.Image: _description_
    """
    # Turn the landmarks into bounding box
    # top-left x, top-left y, bottom right x, bottom right y
    landmark_coords = []
    if feature in face_landmarks:
        landmark_coords = face_landmarks[feature]
    elif feature == "eyes":
        landmark_coords = (
            face_landmarks["right_eye"]
            + face_landmarks["left_eye"]
            + face_landmarks["left_eyebrow"]
            + face_landmarks["right_eyebrow"]
        )
    elif feature == "nose":
        landmark_coords = face_landmarks["nose_bridge"] + face_landmarks["nose_tip"]
    elif feature == "mouth":
        landmark_coords = face_landmarks["top_lip"] + face_landmarks["bottom_lip"]
    else:  # No such feature
        print("No such feature.")
        return None
    try:
        left = min([coords[0] for coords in landmark_coords]) * 0.9
        top = min([coords[1] for coords in landmark_coords]) * 0.9
        right = max([coords[0] for coords in landmark_coords]) * 1.1
        bottom = max([coords[1] for coords in landmark_coords]) * 1.1
        image_cropped = image.crop((left, top, right, bottom))
        return image_cropped
    except ValueError:  # empty feature
        print(f"{feature} is not detected!")
        return None


def detect_facial_landmarks(img: any) -> list:
    """
    This function detect the facial landmarks

    Args:
        img (any): image obj. Format supported are filepath (str), numpy array

    Returns:
        list: list of the detected facial landmarks
    """
    image_array = load_img(img)
    face_landmarks_list = face_recognition.face_landmarks(image_array)
    return face_landmarks_list
