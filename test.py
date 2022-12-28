from deepface import DeepFace
import os

# Settings
img_dir = "data"
distance_metrics = "euclidean"
model_name = "Facenet"

paternal_result = DeepFace.verify(
    img1_path=os.path.join(img_dir, "father.jpeg"),
    img2_path=os.path.join(img_dir, "child.png"),
    model_name=model_name,
    detector_backend="mtcnn",
    distance_metric=distance_metrics,
)
maternal_result = DeepFace.verify(
    img1_path=os.path.join(img_dir, "mother.jpeg"),
    img2_path=os.path.join(img_dir, "child.png"),
    model_name=model_name,
    detector_backend="mtcnn",
    distance_metric=distance_metrics,
)

sameperson_result = DeepFace.verify(
    img1_path=os.path.join(img_dir, "mother.jpeg"),
    img2_path=os.path.join(img_dir, "mother2.jpeg"),
    model_name=model_name,
    detector_backend="mtcnn",
    distance_metric=distance_metrics,
)

# Decision
print(f"Mother-mother: {sameperson_result['distance']}")
print(f"Father-child: {paternal_result['distance']}")
print(f"Mother-child: {maternal_result['distance']}")
if maternal_result["distance"] < paternal_result["distance"]:
    print("The child looks more like mother.")
else:
    print("The child looks more like father.")
