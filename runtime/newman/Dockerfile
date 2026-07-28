FROM node:20-bookworm-slim as environment

RUN npm install -g newman
RUN npm install -g newman-reporter-html
RUN npm install -g newman-reporter-csv

COPY Conduit.postman_collection.json /newman/Conduit.postman_collection.json

ENTRYPOINT [ "/bin/sh" ]