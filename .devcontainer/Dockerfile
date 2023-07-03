FROM python:3.10.12-bullseye

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake

# Copy the current directory contents into the container at /home/RL_Connect4 (do not use home alone)
COPY . /home/RL_Connect4

# Set the working directory to /home/RL_Connect4
WORKDIR /home/RL_Connect4

# Install any additional dependencies you need 
RUN pip install -r requirements.txt 

# Install local folder as Python module (will use setup.py from local folder, here WORKDIR or /home/RL_Connect4)
RUN pip install -e .