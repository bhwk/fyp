FROM python:3.12

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && apt-get clean
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to path
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace
COPY ./src/poetry.lock ./src/poetry.lock
COPY ./src/pyproject.toml ./src/pyproject.toml
WORKDIR /workspace/src
RUN poetry install

CMD ["bash"]
