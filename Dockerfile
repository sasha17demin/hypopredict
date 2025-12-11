FROM python:3.12.9-slim

COPY requirements.txt requirements.txt
COPY api api
COPY hypopredict hypopredict
COPY models models

RUN pip install -r requirements.txt
# RUN pyenv install 3.12.9 this should update the python version, but I'll check with TA.


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
