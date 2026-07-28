FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

FROM node:20-bookworm-slim as environment

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y openssl
RUN apt-get install -y git
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

COPY --from=uv /usr/local /opt/uv
ENV PATH="/opt/uv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/uv/lib:${LD_LIBRARY_PATH}"

ENV UV_TOOL_BIN_DIR=/opt/uv/bin
ENV UV_TOOL_DIR=/opt/uv/tool

RUN uv venv /openhands-env
RUN uv pip install --python /openhands-env/bin/python 'openhands[sdk]==1.12.0'

RUN uv venv /mini-env
RUN uv pip install --python /mini-env/bin/python 'mini-swe-agent==1.17.5'

FROM environment as repository

RUN git config --global user.name "evaluate"
RUN git config --global user.email "evaluate@mail.com"

RUN git clone https://github.com/SeuRonao/realworld-express-prisma repository
WORKDIR /repository
RUN git reset --hard 1f5a21ed6eec656410655cab9d5d3c7148b57a93

COPY misc/.env.development .

RUN npm install

ENTRYPOINT [ "/bin/sh" ]