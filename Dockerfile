FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
RUN pip install jupyter
RUN pip install matplotlib
RUN jupyter nbconvert --to python main.ipynb
COPY . /app
CMD python main.py