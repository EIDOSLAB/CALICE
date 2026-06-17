#!/bin/sh

# Script to build & deploy your docker image
docker build -t url_server/CALICE:tag . -f Dockerfile
docker push url_server/CALICE:tag
