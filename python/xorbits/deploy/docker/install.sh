#!/bin/bash
set -e

source=$1
content=$2

if [ "$source" == "pip" ]; then
  pip install --upgrade pip
  mkdir -p /srv/pip
  touch /srv/pip/requirements.txt
  echo "$content" > /srv/pip/requirements.txt
  pip install -r /srv/pip/requirements.txt
elif [ "$source" == "conda_default" ]; then
  mkdir -p /srv/conda
  touch /srv/conda/conda.txt
  echo "$content" > /srv/conda/conda.txt
  conda install --yes --file /srv/conda/conda.txt
elif [ "$source" == "conda_yaml" ]; then
  mkdir -p /srv/conda
  touch /srv/conda/env.yaml
  echo "$content" > /srv/conda/env.yaml
  conda env update --name base --file /srv/conda/env.yaml
else
  echo "There are no extra packages to install."
fi
