FROM python:3.12-slim

RUN apt update
RUN apt install -y nginx vim git

WORKDIR /LAH

COPY . /LAH/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda env create -f environment.yml

ENTRYPOINT ["/bin/bash"]

# EXPOSE 8000

# CMD ["sh", "train.sh"]
