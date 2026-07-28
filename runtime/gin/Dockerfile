FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

FROM golang:bookworm AS environment

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y ca-certificates
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

FROM environment AS repository

RUN git config --global user.name "evaluate"
RUN git config --global user.email "evaluate@mail.com"

RUN git clone https://github.com/gothinkster/golang-gin-realworld-example-app.git /repository
WORKDIR /repository
RUN git reset --hard 626c372d259472148d93303f74aa9b9a1cdcef24

ENV PORT=8080
ENV GIN_MODE=debug
ENV DB_PATH=./data/gorm.db
ENV TEST_DB_PATH=./data/gorm_test.db

RUN go build ./...
# RUN go test ./...
RUN go mod tidy

ENTRYPOINT [ "/bin/sh" ]
