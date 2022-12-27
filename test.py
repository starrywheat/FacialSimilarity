from deepface import DeepFace
import os

print("hello")
img_dir = "data"
paternal_result = DeepFace.verify(
    img1_path=os.path.join(img_dir, "father.jpeg"),
    img2_path=os.path.join(img_dir, "child.png"),
    model_name="Facenet",
    detector_backend="mtcnn",
    distance_metric="cosine",
)
maternal_result = DeepFace.verify(
    img1_path=os.path.join(img_dir, "mother.jpeg"),
    img2_path=os.path.join(img_dir, "child.png"),
    model_name="Facenet",
    detector_backend="mtcnn",
    distance_metric="cosine",
)
