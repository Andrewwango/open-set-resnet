FROM python:3.8

ENV APP_HOME /code
WORKDIR $APP_HOME
COPY . /code

RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org

CMD ["uvicorn", "main:app"]
