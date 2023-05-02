# gptcode
GPT model based code analysis

## Setup

```
sudo apt install docker.io nvidia-container-toolkit
sudo usermod -aG docker $USER
mkdir ../model
mkdir ../data
reboot
```

place source code repositories for training data set into ../data then build the docker container (also run this command to rebuild container after making any changes to train_tf.py:

`docker build -t gpt2_training .`

then run with:

`docker run ---gpus all -it --rm -v /home/$USER/data:/app/training -v /home/$USER/model:/app/mlcoding -w /app gpt2_training:latest python train_tf.py`
