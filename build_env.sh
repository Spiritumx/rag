#!/bin/bash
# git clone https://github.com/Spiritum-coder/graduateRAG.git
# token ***REMOVED***


curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo apt update && sudo apt install p7zip-full unzip -y

cd download

chmod +x extract_datasets.sh
./extract_datasets.sh

cd ..

conda env update --file pixi_env.yaml

docker compose up -d