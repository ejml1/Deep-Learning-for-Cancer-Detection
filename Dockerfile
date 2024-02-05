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

RUN pip3 install --user --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.txt

RUN pip install scikit-learn opencv-python

RUN pip install scikit-learn matplotlib

RUN pip install seaborn

RUN pip install pickle5

RUN pip install git+https://github.com/qubvel/classification_models.git

RUN pip install keras-cv-attention-models

RUN pip install keras-cv-attention-models

RUN pip install scikit-image

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Miniconda3-latest-Linux-x86_64.sh

# ENV PATH="/opt/conda/bin:${PATH}"

# # Create a Conda environment
# RUN conda create --name myenv python=3.8 && \
#     echo "conda activate myenv" >> ~/.bashrc

# # Activate the Conda environment
# SHELL ["/bin/bash", "--login", "-c"]

CMD [ "jupyter", "lab", "-p", "8888"]