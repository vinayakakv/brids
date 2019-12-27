FROM tensorflow/tensorflow:latest-py3
COPY dvc_0.78.1_amd64.deb /
RUN dpkg -i /dvc_0.78.1_amd64.deb
RUN apt update && apt install -y git
RUN pip install sklearn pandas seaborn joblib
