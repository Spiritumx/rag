# -----------------------------
# 注意：
# 1. 需要先安装 aria2： https://aria2.github.io/
# 2. aria2c 命令行支持多线程 HTTP/HTTPS 下载
# 3. 如需使用代理，请修改下方的代理配置
# -----------------------------

# 代理配置（如不需要代理，将 $USE_PROXY 设置为 $false）
$USE_PROXY = $true
$PROXY_SERVER = "http://127.0.0.1:7890"  # 修改为你的代理地址和端口

# 构建 aria2c 代理参数
if ($USE_PROXY) {
    $PROXY_ARGS = "--all-proxy=$PROXY_SERVER"
    Write-Host "使用代理: $PROXY_SERVER"
} else {
    $PROXY_ARGS = ""
    Write-Host "不使用代理"
}

# 创建目录
New-Item -ItemType Directory -Path ".temp" -Force
New-Item -ItemType Directory -Path "raw_data" -Force

# -----------------------------
# 下载 HotpotQA 数据
# -----------------------------
# Write-Host "`nDownloading raw hotpotqa data"
# New-Item -ItemType Directory -Path "raw_data\hotpotqa" -Force

# aria2c -x 16 -s 16 -o "raw_data\hotpotqa\hotpot_train_v1.1.json" "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
# aria2c -x 16 -s 16 -o "raw_data\hotpotqa\hotpot_dev_distractor_v1.json" "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

# -----------------------------
# 下载 2WikiMultihopQA 数据
# -----------------------------
# Write-Host "`nDownloading raw 2wikimultihopqa data"
# New-Item -ItemType Directory -Path "raw_data\2wikimultihopqa" -Force

# aria2c -x 16 -s 16 -o ".temp\2wikimultihopqa.zip" "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1"
# Expand-Archive -Path ".temp\2wikimultihopqa.zip" -DestinationPath "raw_data\2wikimultihopqa" -Force

# -----------------------------
# 下载 Musique 数据（需手动下载）
# -----------------------------
# Write-Host "`nDownloading raw musique data"
# New-Item -ItemType Directory -Path "raw_data\musique" -Force
# Write-Host "请先手动访问以下链接下载 musique_v1.0.zip 并放入 .temp 目录:"
# Write-Host "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing"
# Write-Host "下载后按任意键继续..."
# Read-Host
# Expand-Archive -Path ".temp\musique_data_v1.0.zip" -DestinationPath "raw_data\musique" -Force

# -----------------------------
# 下载 IIRC 数据
# -----------------------------
<# Write-Host "`nDownloading raw iirc data"
New-Item -ItemType Directory -Path "raw_data\iirc" -Force

# aria2c -x 16 -s 16 -o ".temp\iirc_train_dev.tgz" "https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz"
tar -xzf ".temp\iirc_train_dev.tgz" -C ".temp"
Move-Item ".temp\iirc_train_dev\train.json" "raw_data\iirc\train.json"
Move-Item ".temp\iirc_train_dev\dev.json" "raw_data\iirc\dev.json"

# 下载 IIRC Wikipedia 语料
Write-Host "`nDownloading iirc wikipedia corpus"
aria2c -x 16 -s 16 -o ".temp\context_articles.tar.gz" "https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz"
tar -xzf ".temp\context_articles.tar.gz" -C "raw_data\iirc" #>

# -----------------------------
# 下载 HotpotQA Wikipedia 语料
# -----------------------------
# Write-Host "`nDownloading hotpotqa wikipedia corpus"
# aria2c -x 16 -s 16 -o ".temp\wikipedia-paragraphs.tar.bz2" "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
# tar -xvf ".temp\wikipedia-paragraphs.tar.bz2" -C "raw_data\hotpotqa"
# Move-Item "raw_data\hotpotqa\enwiki-20171001-pages-meta-current-withlinks-abstracts" "raw_data\hotpotqa\wikipedia-paragraphs"

# -----------------------------
# 下载 Natural Questions 数据
# -----------------------------
Write-Host "`nDownloading Natural Questions data"
New-Item -ItemType Directory -Path "raw_data\nq" -Force

if ($USE_PROXY) {
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\nq\biencoder-nq-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"
    aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\nq\biencoder-nq-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
} else {
    # aria2c -x 16 -s 16 -o "raw_data\nq\biencoder-nq-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"
    aria2c -x 16 -s 16 -o "raw_data\nq\biencoder-nq-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
}

# -----------------------------
# 下载 TriviaQA 数据
# -----------------------------
Write-Host "`nDownloading TriviaQA data"
New-Item -ItemType Directory -Path "raw_data\trivia" -Force

if ($USE_PROXY) {
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\trivia\biencoder-trivia-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz"
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\trivia\biencoder-trivia-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz"
} else {
    # aria2c -x 16 -s 16 -o "raw_data\trivia\biencoder-trivia-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz"
    # aria2c -x 16 -s 16 -o "raw_data\trivia\biencoder-trivia-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz"
}

# -----------------------------
# 下载 SQuAD 数据
# -----------------------------
Write-Host "`nDownloading SQuAD data"
New-Item -ItemType Directory -Path "raw_data\squad" -Force

if ($USE_PROXY) {
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\squad\biencoder-squad1-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz"
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\squad\biencoder-squad1-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz"
} else {
    # aria2c -x 16 -s 16 -o "raw_data\squad\biencoder-squad1-dev.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz"
    # aria2c -x 16 -s 16 -o "raw_data\squad\biencoder-squad1-train.json.gz" "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz"
}

# -----------------------------
# 下载 Wikipedia passages
# -----------------------------
Write-Host "`nDownloading Wikipedia passages"
New-Item -ItemType Directory -Path "raw_data\wiki" -Force

if ($USE_PROXY) {
    # aria2c -x 16 -s 16 $PROXY_ARGS -o "raw_data\wiki\psgs_w100.tsv.gz" "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
} else {
    # aria2c -x 16 -s 16 -o "raw_data\wiki\psgs_w100.tsv.gz" "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
}

# -----------------------------
# 清理临时文件
# -----------------------------

Write-Host "`nAll downloads complete!"
