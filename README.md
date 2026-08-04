# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### 国内镜像源（可选）

如果在国内网络环境下 `uv sync` / `uv run` 卡在下载依赖，可以配置以下镜像。

**PyPI 包镜像**（清华源，任选一种方式）：

```sh
# 方式一：临时，当前 shell 生效
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二：单次命令生效
uv run --default-index https://pypi.tuna.tsinghua.edu.cn/simple pytest
```

也可以写进 `~/.config/uv/uv.toml` 全局生效（**推荐**，不污染仓库）：

```toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

> 注意：本仓库的 `pyproject.toml` 是作业模板的一部分，尽量不要把镜像配置写进去，避免提交时带上环境相关的改动。

**Python 解释器下载镜像**：本项目 `python-preference = "managed"`，`uv` 会自行下载 CPython。若这一步很慢：

```sh
export UV_PYTHON_INSTALL_MIRROR=https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download
```

**HuggingFace 镜像**（下载数据集和 tokenizer 文件时用）：

```sh
export HF_ENDPOINT=https://hf-mirror.com
```

`HF_ENDPOINT` 只对 `huggingface_hub` / `datasets` 这类 Python 库生效。用 `wget`/`curl` 直接下载时，需要手动把 URL 里的 `huggingface.co` 换成 `hf-mirror.com`（见下方 Download data）。

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

#### 国内镜像版本

把域名换成 `hf-mirror.com` 即可，路径完全一致。macOS 默认不自带 `wget`，用 `curl -L -O` 替代（`-L` 跟随重定向，`-O` 保留远程文件名，两者都不能省）：

``` sh
mkdir -p data
cd data

curl -L -O https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -O https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

curl -L -O https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
curl -L -O https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

下完检查一下文件大小（`ls -lh`）：如果得到的是几 KB 的小文件，说明拿到的是重定向页面或错误页，而不是真正的数据。

