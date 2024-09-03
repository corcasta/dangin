FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /dangin
COPY . .
RUN apt-get update && apt-get install sudo -y \
    && apt install libx11-dev ffmpeg libsm6 libxext6 -y \
    && pip install -r requirements.txt  \
    && pip uninstall opencv==4.7.0 -y \
    && pip install -e .
    
CMD ["uvicorn", "dangin.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
