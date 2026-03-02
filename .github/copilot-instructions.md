## 快速导读（给 AI 编码代理）

这是 F5-TTS 项目的核心要点与可操作指南，帮助 AI 代理在本仓库中立刻开展开发工作。

- 入口与主命令：`f5-tts_infer-gradio`（Gradio 界面），`f5-tts_infer-cli`（CLI），`f5-tts_finetune-cli` / `-gradio`（训练/微调）。这些都是在 `pyproject.toml` 的 `project.scripts` 中注册的控制台脚本。
- 关键目录：`src/f5_tts/`（主代码），`src/f5_tts/infer/`（推理相关代码与示例），`src/f5_tts/train/`（训练/微调逻辑），`src/f5_tts/model/`（模型骨架与后端实现），`F5-TTS/`（镜像复制的资料/备份）。

## 大体架构（快速理解数据流与边界）

- 推理主流程：`infer_cli.py` / `infer_gradio.py` → 调用 `utils_infer.py` 中的工具函数（`preprocess_ref_audio_text`, `load_model`, `load_vocoder`, `infer_process` 等）→ 使用 `Models.CFM`（位于 `src/f5_tts/Models`）生成 mel，再由 vocoder（`vocos` 或 `bigvgan`）解码为波形。
- 模型/配置：模型配置由 `src/f5_tts/configs/*.yaml` 与 OmegaConf 管理（`f5_tts.api.F5TTS` 使用 `OmegaConf.load` 与 `hydra.utils.get_class`）。检查点通常从 Hugging Face 自动下载（例如 `hf://SWivid/F5-TTS`），也可通过 `--ckpt_file` 指定本地路径。
- 文本/词表：tokenizer 使用项目内 `vocab.txt`（默认路径 `src/f5_tts/infer/examples/vocab.txt`），可通过 `--vocab_file` 覆盖。

## 项目约定与重要实现细节（不要随意更改）

- 引用音频裁剪：参考音频会被裁短到 ~12s（`preprocess_ref_audio_text` 中的逻辑），若参考音频过长，代码会尝试基于静音段裁剪或直接截断。改动必须保留相同的静音处理策略以避免生成质量差异。
- 分段生成（chunking）：长生成文本会被 `chunk_text` 拆分（基于标点和字节长度），再并发执行批次推理（`infer_batch_process` 使用 ThreadPoolExecutor），最后进行 cross-fade 拼接（可通过 `cross_fade_duration` 控制）。保留 yield/streaming 接口时请注意 `streaming` 与非流的返回差异。
- vocoder 类型：两个主要 mel->wav 后端：`vocos`（默认）与 `bigvgan`。`bigvgan` 可能依赖第三方子模块（`third_party/BigVGAN`），若更改需保证子模块路径与加载方式一致。
- ASR/转录：当 `ref_text` 为空时，会初始化 HuggingFace 的 `pipeline('automatic-speech-recognition', model='openai/whisper-large-v3-turbo')` 做转录（见 `utils_infer.transcribe`），并有本地内存缓存以避免重复转录。
- 设备选择：代码根据环境检测 `cuda`/`xpu`/`mps`/`cpu`（见多个文件中的 device 选择逻辑）。修改时注意 dtype 与设备兼容（例如半精度选择规则在 load_checkpoint/initialize_asr_pipeline 中出现）。

## 开发 / 运行 / 调试 常用命令（来自仓库）

- 本地 editable 安装（开发时推荐）：

  pip install -e .

- 预提交钩子：

  pip install pre-commit
  pre-commit install
  pre-commit run --all-files

- 运行 Gradio 推理界面（默认会自动下载权重）：

  f5-tts_infer-gradio

- CLI 推理示例（参见 `src/f5_tts/infer/examples/basic/basic.toml`）：

  f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio infer/examples/basic/basic_ref_en.wav --ref_text "..." --gen_text "..."

- 使用自定义 checkpoint 或 vocab：

  f5-tts_infer-cli --ckpt_file ckpts/your_file.safetensors --vocab_file path/to/vocab.txt

- Docker：仓库顶层提供 Dockerfile，可直接构建镜像并运行服务（README 中有示例 `docker run`/compose）。

## 代码风格 / 模式提示（便于 AI 迁移补全）

- 资源访问常用模式：`importlib.resources.files('f5_tts').joinpath(...)` 用于在 package 内定位示例/配置文件，写修改类接口时请沿用该路径获取方式，保证 package 模式与脚本模式都能工作。
- 并发与流：`infer_batch_process` 支持两种工作模式：非流（返回最终拼接的 wav/mel）和流（yield 固定大小的 wav chunk）。修改时注意保留两种行为签名。
- 配置优先级：命令行/`.toml` > 环境变量 > 代码中的默认值（常见在 `utils_infer`、`api.py` 中）。当新增参数时也应支持 `.toml` 和 CLI 标志。

## 常见故障与快速检查点（调试捷径）

- 生成为空或纯静音：检查 ffmpeg 是否可用（pydub 依赖），也可尝试关闭 `use_ema`（对早期微调 checkpoint 常见问题）。
- 音色/发音不对：确认 `vocab.txt` 与 tokenizer 一致（`utils_infer.get_tokenizer`），以及 `mel_spec_type`（`vocos` vs `bigvgan`）是否与 checkpoint 匹配。
- 显存/设备问题：检查 device 检测逻辑与 dtype 推断（半精度只有当 GPU 支持时启用）。本地调试可通过 `--device` 或在 `F5TTS` API 中传入 `device` 来覆盖。

## 关键文件（快速链接）

- `pyproject.toml` — 命令行脚本入口与依赖。
- `src/f5_tts/infer/utils_infer.py` — 推理流程、vocoder、ASR、chunking、拼接等实现（首选阅读以理解推理细节）。
- `src/f5_tts/infer/infer_cli.py` & `infer_gradio.py` — CLI / web 接口示例与参数用法。
- `src/f5_tts/api.py` — 程序化 API 封装（可用于服务化或嵌入式调用）。
- `src/f5_tts/configs/*.yaml` — 模型配置与 backbone 指定（被 `api.py`/OmegaConf 读取）。

---
请先审阅上述内容，有哪里需要补充（例如更具体的函数签名示例、额外的调试命令或 CI 流程说明）我会据此迭代更新。
