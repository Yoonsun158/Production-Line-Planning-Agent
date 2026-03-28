
设计了一个AI agent，用于为不同类型的废旧电池生成回收计划报告，并可视化搭建的产线。

# Docker 部署

1. 安装docker
2. 安装 NVIDIA Container Toolkit
    ```bash
    # 添加 NVIDIA 仓库
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # 配置 Docker 运行时
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
3. 验证 GPU 在 Docker 中可用

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
4. 使用 Docker Compose（推荐）

    在项目根目录创建 `docker-compose.yml`，一键管理 Ollama + 后端服务：

    ```yaml
    services:
    ollama:
        image: ollama/ollama
        container_name: ollama
        ports:
        - "11434:11434"
        volumes:
        - ollama_data:/root/.ollama
        deploy:
        resources:
            reservations:
            devices:
                - driver: nvidia
                count: all
                capabilities: [gpu]
        restart: unless-stopped

    backend:
        build: ./backend
        container_name: brd-backend
        ports:
        - "8000:8000"
        environment:
        - OLLAMA_HOST=http://ollama:11434
        depends_on:
        - ollama
        restart: unless-stopped

    volumes:
    ollama_data:
    ```

    配套的 `backend/Dockerfile`：

    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    RUN python init_kb.py || true
    CMD ["python", "server.py"]
    ```

    启动命令：

    ```bash
    docker compose up -d

    # 首次启动后下载模型
    docker exec ollama ollama pull qwen2.5:14b
    docker exec ollama ollama pull nomic-embed-text

    # 初始化知识库（等 Ollama 拉完模型后）
    docker exec brd-backend python init_kb.py
    ```
5. 验证服务

```
curl http://localhost:11434/api/tags
curl http://localhost:8000/health
```

6. 安装依赖

```bash
# 创建环境（如果已存在会提示）
conda env create -f environment.yml
conda activate ollama-qwen
```

# 启动

```bash
# 服务器(linux)
# 直接启动 Docker 容器即可

docker start ollama

docker start brd-backend


# 客户端(windows)
# 首次初始化知识库（也需要调用 Ollama 做 embedding）
cd .../BatteryRecovery_V1/backend
python init_kb.py   
python server.py

# 浏览器打开前端
start ../interface.html
```

# 测试

1. 18650圆柱三元锂电池
2. 磷酸铁锂软包电池
3. 丰田普锐斯镍氢电池

