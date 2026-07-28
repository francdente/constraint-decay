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

RUN git clone https://github.com/borys25ol/fastapi-realworld-backend.git repository
WORKDIR /repository
RUN git reset --hard 55111c6b335455734c139a2245d98be8288be528

COPY misc/.env .

RUN uv pip install --upgrade --system wheel setuptools
RUN uv pip install --system -r requirements.txt

ENTRYPOINT [ "/bin/sh" ]
