FROM python:3.9

# Set the working directory to
WORKDIR /Machine_Learning

#install dependencies
COPY ./requirements.txt /Machine_Learning
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#copy the current directory contents into the container at /Machine_Learning
COPY . /Machine_Learning
