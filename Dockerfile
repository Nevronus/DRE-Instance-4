# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers?utm_source=youtube&utm_medium=unpaidsoc&utm_campaign=CDR_ret_aiml_vrqxiinldak_PrototypeToProduction_081622&utm_content=description#tensorflow
FROM asia-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest

WORKDIR /

COPY requirements.txt .

COPY main.py .

COPY ./models ./models
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gunicorn

EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]/