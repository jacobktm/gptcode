FROM tensorflow/tensorflow:2.12.0-gpu

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY train_tf.py /app/

WORKDIR /app
