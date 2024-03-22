FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends locales tzdata && \
    echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt ./

# Install libGL
RUN apt-get update --yes && \
    apt-get install -y libgl1-mesa-glx

RUN pip3 install --user --upgrade --disable-pip-version-check pip

RUN pip install scikit-learn opencv-python

RUN pip install scikit-learn matplotlib

RUN pip install seaborn

RUN pip install pickle5

RUN pip install git+https://github.com/qubvel/classification_models.git

RUN pip install keras-cv-attention-models

RUN pip install scikit-image

RUN pip install scikeras

RUN pip install keras-tuner

CMD [ "jupyter", "lab", "-p", "8888"]