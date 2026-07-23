# Florence-2-base — OCR 模型目录

默认 OCR 引擎使用 Microsoft Florence-2-base 模型。

## 下载模型权重

`model.safetensors` 文件约 442 MB，需自行下载放到此目录：

```powershell
huggingface-cli download microsoft/Florence-2-base model.safetensors --local-dir Florence-2-base
```

或下载完整仓库：

```powershell
huggingface-cli download microsoft/Florence-2-base --local-dir Florence-2-base
```

## 此目录应包含的文件

```
Florence-2-base/
├── model.safetensors          # 模型权重（约 442 MB，需自行下载）
├── config.json                # 模型配置
├── preprocessor_config.json   # 预处理配置
├── tokenizer.json             # 分词器
├── tokenizer_config.json      # 分词器配置
├── vocab.json                 # 词表
├── configuration_florence2.py # 自定义模型配置类
├── modeling_florence2.py      # 自定义模型类
├── processing_florence2.py    # 自定义处理器类
├── LICENSE                    # MIT 许可证
└── README.md                  # Florence-2 官方说明（HuggingFace）
```

## 其他 OCR 引擎

如果你在 `config.py` 中将 `OCR_BACKEND` 改为了其他 OCR 引擎，则不需要此目录。
