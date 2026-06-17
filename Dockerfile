FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

RUN apt update -y
RUN apt install -y gcc
RUN apt install -y g++ 

RUN pip install numpy
RUN pip install timm
RUN pip install wandb
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install gdown tensorboard
RUN pip install torchprofile
RUN pip install torch_geometric
RUN pip install einops
RUN pip install compressai==1.2.4
RUN pip install torch-pruning
RUN pip install seaborn



COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

ENTRYPOINT ["python"]
