FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for psycopg binary are not required; keep minimal.
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml /app/pyproject.toml
COPY mesh_router /app/mesh_router

RUN pip install --no-cache-dir .

EXPOSE 4010

CMD ["mesh-router", "--host", "0.0.0.0", "--port", "4010"]

