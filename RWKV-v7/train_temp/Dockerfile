# 使用官方 PyTorch 镜像作为基础镜像
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# 设置工作目录
WORKDIR /workspace/RWKV-LM

# 设置 GCC 环境变量
ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# 安装项目依赖
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y --no-install-recommends libgl1 libsm6 && \
    pip install --no-cache-dir numpy tqdm pytorch-lightning deepspeed && \
    gcc --version && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 设置默认命令（可选）
CMD ["python", "train.py"]