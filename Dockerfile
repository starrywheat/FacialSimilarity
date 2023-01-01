FROM python:3.9-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install  build-essential cmake -y
RUN apt-get install libgtk-3-dev -y
RUN apt-get install libboost-all-dev -y
EXPOSE 8501

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
