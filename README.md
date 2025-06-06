# FacialSimilarity

A [Streamlit](https://streamlit.io/)🎈app to compare facial similarity of family members. Whom does your child look like?

It is powered by the deep learning library [deepface](https://github.com/serengil/deepface) which detect and compare the facial images, and [dlib](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) which detect the facial landmarks (such as eyes and mouth). Simply upload the pictures of the family members (support `png`, `jpeg`, `jpg` format) and let the app analyse the images for you!

More explaination of this application can be found in this [medium post](https://medium.com/@starrywheat.dm/develop-a-streamlit-app-for-facial-similarity-dc8f3fc34c0f).

## Running locally in Docker
This app uses Streamlit (you will install this library using the previous step) and deepface library. It is best to run it with Docker so that the neccessary plugins and libraries are installed. To build and run it in Docker:
```bash
docker build -t streamlit_app .
docker run streamlit_app -p 8501:8501
```
A new window will pop up in the browser to show you the app. Otherwise, go to `http://localhost:8501` in your browser to access the app.


