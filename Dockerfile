FROM continuumio/miniconda3

RUN apt update
RUN apt install -y nginx vim git

WORKDIR /LAH

COPY . /LAH/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
COPY environment.yml .
RUN conda env create -f environment.yml

# EXPOSE 8000

SHELL ["conda", "run", "-n", "lah", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "lah"]

# CMD ["conda", "activate", "lah"]