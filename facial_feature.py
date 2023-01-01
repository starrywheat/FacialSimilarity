# import face_recognition
from collections import OrderedDict

import dlib
import numpy as np
from PIL import Image

FACIAL_LANDMARKS_68_IDXS = OrderedDict(
    [
        ("mouth", (48, 68)),
        ("inner_mouth", (60, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17)),
    ]
)


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
        left = min([coords[0] for coords in landmark_coords]) * 0.95
        top = min([coords[1] for coords in landmark_coords]) * 0.95
        right = max([coords[0] for coords in landmark_coords]) * 1.05
        bottom = max([coords[1] for coords in landmark_coords]) * 1.05
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

    # face_landmarks_list = face_recognition.face_landmarks(image_array)

    shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
    # Initialize dlib's face detector, then facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # detect faces
    rects = detector(image_array, 1)

    # Facial landmarks
    facial_landmarks = []
    for rect in rects:
        landmarks = {}
        # determine the facial landmarks for the face region
        shape = predictor(image_array, rect)
        for k, v in FACIAL_LANDMARKS_68_IDXS.items():
            coords = [(shape.part(i).x, shape.part(i).y) for i in range(*v)]
            landmarks[k] = coords
        facial_landmarks.append(landmarks)
    return facial_landmarks
