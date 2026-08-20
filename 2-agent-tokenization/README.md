# 2-agent-tokenization - LLM Tokenization 示例

本示例演示如何使用 HuggingFace Transformers 库对大型语言模型进行 Tokenization（分词）处理，并生成文本回复。
参考 [3.2.3 调用开源大语言模型 教程](https://hello-agents.datawhale.cc/#/./chapter3/%E7%AC%AC%E4%B8%89%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80?id=_323-%e8%b0%83%e7%94%a8%e5%bc%80%e6%ba%90%e5%a4%a7%e8%af%ad%e8%a8%80%e6%a8%a1%e5%9e%8b) 实现。

## 功能

- 加载本地 Qwen1.5-0.5B-Chat 模型
- 使用分词器对对话模板进行编码
- 生成模型回复并解码输出

## 项目结构

```
2-agent-tokenization/
├── LlmTokenization.py          # 主程序：Tokenization 与文本生成示例
└── Qwen1.5-0.5B-Chat/          # 本地模型文件（已加入 .gitignore）
    └── models/qwen--Qwen1.5-0.5B-Chat/snapshots/master/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers
```

### 2. 下载模型（可选）

如果尚未下载模型，可以运行以下命令从 ModelScope 下载：

```python
from modelscope import snapshot_download
snapshot_download(model_id='qwen/Qwen1.5-0.5B-Chat', cache_dir='./Qwen1.5-0.5B-Chat')
```

### 3. 运行

```bash
python LlmTokenization.py
```

## 工作原理

1. **加载模型和分词器** - 从本地路径加载 Qwen1.5-0.5B-Chat 模型及其分词器
2. **构建对话模板** - 使用 `apply_chat_template` 将对话历史格式化为模型可接受的输入
3. **Tokenization** - 将文本转换为 Token ID 序列
4. **文本生成** - 调用 `model.generate()` 生成新的 Token
5. **解码输出** - 将生成的 Token ID 转换回可读文本

## 注意事项

- 模型文件较大（约 1.8GB），已加入 `.gitignore`，不会提交到 Git
- 首次运行需要加载模型到内存，可能需要几秒钟
- 默认使用 CPU 推理，如有 GPU 会自动切换
