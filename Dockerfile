FROM python:slim

# to avoid installation error with chroma-hnswlib
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN python -m pip install -r requirements.txt

COPY . /app/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]