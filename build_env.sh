#!/bin/bash
# git config --global credential.helper store
# source /etc/network_turbo

# git clone https://github.com/Spiritum-coder/graduateRAG.git
# token ***REMOVED***



apt update && apt install p7zip-full unzip -y

cd download

chmod +x extract_datasets.sh
./extract_datasets.sh

cd ..

conda env update --file pixi_env.yaml

chmod +x retriever_server/es.sh

./retriever_server/es.sh install

./retriever_server/es.sh start