FROM continuumio/miniconda3
WORKDIR /home/app
RUN apt-get update -y 
RUN apt-get install nano unzip
RUN apt install curl -y
RUN curl -fsSL https://get.deta.dev/cli.sh | sh
COPY . /home/app
RUN pip install -r requirements.txt
CMD uvicorn app:app --host 0.0.0.0 --port 80
CMD gunicorn app:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker
