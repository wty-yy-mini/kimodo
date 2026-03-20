# 安装
1. 在133服务器上用uv配置环境，创建`uv.lock`
2. 安装方法`uv sync --extra all`，进容器`source .venv/bin/activate`

启动方法，先`hf auth login`，在开个账号申请下[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)模型的权重下载权限，随便添英文名和非中国地区就行，再开个镜像`export HF_ENDPOINT=https://hf-mirror.com`，直接启动程序就能自动下载模型权重了

一共有这几个模型[quick_start](docs/source/getting_started/quick_start.md)和G1相关的就是
- `Kimodo-G1-RP-v1`：用的训的数据更多，应该更好，下面都用这个
- `Kimodo-G1-SEED-v1`：用的数据少，主要用于比benchmark

# 启动
- kimodo_textencoder 是“文本编码服务”，[默认端口9550](localhost:9550)，加载LLM2Vec + Llama，常驻后台
- kimodo_demo 是“完整的可视化交互界面”，[默认端口7860](localhost:7860)，加载Kimodo动作生成模型
- kimodo_gen 也是文本->生成轨迹，只不过是cli版的

先进环境
```bash
source .venv/bin/activate
```

## text encoder启动
```bash
CUDA_VISIBLE_DEVICES=7 kimodo_textencoder
# 或者指定端口和监听地址
CUDA_VISIBLE_DEVICES=7 GRADIO_SERVER_PORT=9550 GRADIO_SERVER_NAME=0.0.0.0 kimodo_textencoder
```
- `GRADIO_SERVER_PORT`：指定端口，默认9550
- `GRADIO_SERVER_NAME`：指定监听地址，默认localhost，可以改为全局监听

显示如下信息就说明创建成功了
```bash
warnings.warn(
* Running on local URL:  http://0.0.0.0:9550
* To create a public link, set `share=True` in `launch()`.
```

## Demo启动
```bash
CUDA_VISIBLE_DEVICES=7 kimodo_demo
# 或者指定text encoder的地址和端口，以及Demo的地址和端口
CUDA_VISIBLE_DEVICES=7 TEXT_ENCODER_URL=http://127.0.0.1:9550 SERVER_PORT=7860 SERVER_NAME=0.0.0.0 kimodo_demo --model Kimodo-G1-RP-v1
```
- `SERVER_PORT`：指定端口，默认7860
- `SERVER_NAME`：指定监听地址，默认localhost，可以改为全局监听

出现如下信息说明启动成功了，可以访问界面了
```bash
╭────── viser (listening *:7860) ───────╮
│             ╷                         │
│   HTTP      │ http://localhost:7860   │
│   Websocket │ ws://localhost:7860     │
│             ╵                         │
╰───────────────────────────────────────╯
```

## cli生成轨迹
```bash
CUDA_VISIBLE_DEVICES=7 kimodo_gen "A person walks forward." \
  --model Kimodo-G1-RP-v1 \
  --duration 5.0 \
  --output output_g1 \
  --num_samples 3
```
- `--duration`：生成轨迹的时长，单位秒，可以是多个数值，空格分割，对应prompt的多个句子
- `--output`：输出路径，生成的轨迹会保存在output_g1.npz和output_g1.csv
- `--num_samples`：生成样本数量，默认为1

# 一些问题

1. 生成最大时长好像是10s
2. 同时有文字和限制时候可能不准
