FROM continuumio/miniconda3:4.6.14
COPY ./app.py /deploy/
COPY ./conda_AutoML916c88b0f44.yml /deploy/
COPY ./model_AutoML916c88b0f44.pkl /deploy/
COPY ./scoring_AutoML916c88b0f44.py /deploy/
WORKDIR /deploy/
RUN conda update -n base -c defaults conda
RUN conda env update --name myenv --file conda_AutoML916c88b0f44.yml
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN python -c "import flask"
RUN python -c "import numpy"
RUN python -c "import pickle"
RUN python -c "import json"
RUN python -c "import pandas"
COPY app.py .
COPY model_AutoML916c88b0f44.pkl .
COPY app-rul.yaml .
EXPOSE 5000
ENTRYPOINT ["conda","run", "-n", "myenv", "python", "app.py"]
