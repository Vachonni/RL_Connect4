FROM python:3.10.12-bullseye

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake

# Copy the current directory contents into the container at /app (do not use home)
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any additional dependencies you need 
RUN pip install -r requirements.txt 

# Install local folder as Python module (will use setup.py from local folder, here WORDIR or /app)
RUN pip install -e .