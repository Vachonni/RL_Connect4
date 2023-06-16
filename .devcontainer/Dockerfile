FROM python:3.10

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake

# Set the working directory to /home
WORKDIR /home

# Copy the current directory contents into the container at /home
COPY . /home

# Install any additional dependencies you need
RUN pip install -r requirements.txt 

# Install local folder as Python module
RUN pip install -e .