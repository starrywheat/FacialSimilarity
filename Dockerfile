FROM python:3.9-slim

COPY . .

RUN pip install -r requirements.txt


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
