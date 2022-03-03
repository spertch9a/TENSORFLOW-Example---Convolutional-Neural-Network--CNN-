FROM tensorflow/tensorflow
WORKDIR /app
COPY . .
RUN pip install jupyter
RUN pip install matplotlib
RUN jupyter nbconvert --to python main.ipynb
CMD python main.py