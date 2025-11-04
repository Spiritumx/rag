#!/bin/bash
# git clone https://github.com/Spiritum-coder/graduateRAG.git
# token ***REMOVED***



sudo apt update && sudo apt install p7zip-full unzip -y

cd download

chmod +x extract_datasets.sh
./extract_datasets.sh

cd ..

conda env update --file pixi_env.yaml

chmod +x retriever_serve/es.sh

./es.sh install

./es.sh start