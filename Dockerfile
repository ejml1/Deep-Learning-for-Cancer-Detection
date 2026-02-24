FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends locales tzdata libgl1-mesa-glx && \
    echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --upgrade pip && \
    pip install \
    scikit-learn \
    opencv-python \
    matplotlib \
    seaborn \
    pickle5 \
    git+https://github.com/qubvel/classification_models.git \
    keras-cv-attention-models \
    scikit-image \
    scikeras \
    keras-tuner \
    "numpy<2" \
    keras==2.12.0

CMD ["jupyter", "lab", "-p", "8888"]