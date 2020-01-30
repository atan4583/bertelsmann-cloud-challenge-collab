FROM python:3.7-slim

ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION=us-west-2

COPY . /root/
RUN cd /root/ && pip3 install -r requirements.txt && pip3 install awscli

ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN aws s3 cp s3://udacity-ai-backend/ . --recursive
ENV FLASK_APP=process.py
WORKDIR /root/
ENTRYPOINT [ "flask", "run", "--host=0.0.0.0" ]
EXPOSE 5000