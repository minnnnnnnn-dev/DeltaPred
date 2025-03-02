FROM python:3.9.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch torchvision torchaudio

COPY dashboard/ dashboard/
COPY test/ test/

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.fileWatcherType=none", "--server.runOnSave=false", "--server.address=0.0.0.0"]