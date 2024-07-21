FROM python:3.10-slim
LABEL author="Jaime Céspedes Sisniega <jaime.cespedes@alumnos.unican.es>"

ARG DATASET
ARG PARENT_DIR

WORKDIR /app

COPY ${PARENT_DIR}/requirements/requirements.txt .

RUN apt-get -y update && \
    addgroup --gid 1001 --system app && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app && \
    pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm requirements.txt

COPY ${PARENT_DIR}/app ./
COPY ./ml/objects/${DATASET}/encoder.pt ./objects/encoder.pt
COPY ./ml/objects/${DATASET}/transform.pt ./objects/transform.pt

USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]