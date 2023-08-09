# Opensource LLM 기반 Langchain 활용하기 tutorial

## Installation
```
# 0. env and python setting
git clone https://github.com/okin1234/opensource_langchain.git
conda create -n seminar python=3.10  # python=3.10 권장
conda activate seminar

# 1. Install system dependencies by conda
conda install -c anaconda gcc_linux-64 gxx_linux-64
conda install -c conda-forge gcc gxx

# 2. If you not installed poetry, install poetry
pip install poetry

# 3. Install python dependencies by poetry (만약 에러뜨면 계속 poetry install 해주면 됨)
poetry install
```
