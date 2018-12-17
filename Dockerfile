FROM python:3.6

RUN apt-get -yqq update
# RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -yqq python
RUN apt-get -yqq install python-pip python-dev

RUN apt-get install -yqq hunspell libhunspell-dev
# RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
RUN pip install --upgrade pip
# Set the working directory to /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install .

RUN python setup.py install

# Download data files
# RUN chmod +x *.sh
# RUN ./init-dl-data.sh

# Run app.py when the container launches
# CMD ["run.sh"]