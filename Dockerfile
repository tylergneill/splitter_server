FROM python:3.5-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    zlib1g-dev \
    libjpeg-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean
COPY tensorflow-1.15.0-cp35-cp35m-manylinux2010_x86_64.whl .
RUN pip install --no-cache-dir ./tensorflow-1.15.0-cp35-cp35m-manylinux2010_x86_64.whl
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd appgroup && \
    useradd -r -M -G appgroup sanskrit
COPY --chown=sanskrit:appgroup data /app/data
COPY --chown=sanskrit:appgroup templates /app/templates
COPY --chown=sanskrit:appgroup ./*.py /app/
USER sanskrit
ENV PORT=5060
CMD gunicorn --bind 0.0.0.0:$PORT --log-level info --timeout 240 --error-logfile - flask_app:app
EXPOSE 5060