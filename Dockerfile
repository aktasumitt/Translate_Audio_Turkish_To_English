FROM python:3.12-slim
WORKDIR /app
COPY  . /app
RUN apt update -y && apt install awscli -y
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu 
RUN pip install scipy sentencepiece sacremoses soundfile ffmpeg python-box pathlib pyyaml Flask transformers

CMD [ "python", "app.py" ]
