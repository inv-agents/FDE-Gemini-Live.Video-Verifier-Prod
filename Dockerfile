FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-deu tesseract-ocr-ita \
      tesseract-ocr-por tesseract-ocr-jpn tesseract-ocr-kor tesseract-ocr-chi-sim \
      tesseract-ocr-ara tesseract-ocr-hin tesseract-ocr-ind tesseract-ocr-fra \
      tesseract-ocr-rus \
      ffmpeg \
      libgl1 libglib2.0-0 \
      libsndfile1 libasound2-dev portaudio19-dev \
      build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd -g 10001 app && useradd -m -u 10001 -g app -s /bin/bash app \
    && mkdir -p /home/app/.cache /home/app/.streamlit /data \
    && chown -R app:app /app /home/app /data

ENV HOME=/home/app
USER app

EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run main.py --server.address=0.0.0.0 --server.port=${PORT}"]
