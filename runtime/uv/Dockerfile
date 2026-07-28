FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS environment

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y ca-certificates
RUN apt-get install -y openssl
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y lsof net-tools procps postgresql-client
RUN rm -rf /var/lib/apt/lists/*

RUN uv venv /openhands-env
RUN uv pip install --python /openhands-env/bin/python 'openhands[sdk]==1.12.0'

RUN uv venv /mini-env
RUN uv pip install --python /mini-env/bin/python 'mini-swe-agent==1.17.5'

FROM environment AS repository

RUN git config --global user.name "evaluate"
RUN git config --global user.email "evaluate@mail.com"

WORKDIR /repository
RUN git init

RUN uv pip install --upgrade --system wheel setuptools

ENTRYPOINT [ "/bin/sh" ]
