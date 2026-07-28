FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS environment

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y openssl
RUN apt-get install -y git
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

RUN uv venv /openhands-env
RUN uv pip install --python /openhands-env/bin/python 'openhands[sdk]==1.12.0'

RUN uv venv /mini-env
RUN uv pip install --python /mini-env/bin/python 'mini-swe-agent==1.17.5'

FROM environment as repository

RUN git config --global user.name "evaluate"
RUN git config --global user.email "evaluate@mail.com"

RUN git clone https://github.com/mhegelheimer/realworld-flask.git repository
WORKDIR /repository
RUN git reset --hard c7d865f89464a1770171492b036b56c87cc76c8a

COPY misc/.env .

RUN uv pip install --upgrade --system wheel setuptools
RUN uv tool install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-root

ENTRYPOINT [ "/bin/sh" ]
