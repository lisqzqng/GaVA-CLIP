FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# add non-root user in docker
RUN addgroup --gid 1001 dw && useradd -ms /bin/bash dw -u 1001 -g 1001

# install python 3.8 in the docker apt-get install openssl* -y && 
RUN apt-get update && apt-get install -y python3.8 python3-pip

# install wget, ssh, git
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y vim wget ssh git

# install pytorch=1.13.0lazy docker
RUN pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# make directory
RUN mkdir /home/dw/vlm

# set the working directory as xxx
WORKDIR /home/dw/vlm

# install the requirements
# will directly copy the file to the working directory
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# set the user
USER dw