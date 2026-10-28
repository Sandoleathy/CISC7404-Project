- 问题： vLLM 初始化时发现 GPU 可用显存不足，报错 `Free memory on device cuda:0 (6.95/8.0 GiB) ... desired GPU memory utilization (0.92, 7.36 GiB)`。原因是此前使用 Transformers 加载的模型占用了部分显存，导致剩余显存低于 vLLM 的预算要求。
- 解决方法： 释放不再使用的模型及其显存，例如使用 `del model`、`gc.collect()`、`torch.cuda.empty_cache()`；必要时重启 Notebook Kernel，并检查是否有其他进程占用显存。
- 问题： vLLM 初始化时提示 KV Cache 显存不足：模型最大序列长度为 `40960`，所需 KV Cache 为 `4.38 GiB`，但可用 KV Cache 只有 `1.89 GiB`。
- 解决方法： 将 `LLM()` 的 `max_model_len` 从模型默认的较大上下文长度降低到 `8192`，减少 KV Cache 的显存需求。
- 问题： Triton 即时编译 CUDA 相关组件时，报错 `fatal error: Python.h: No such file or directory`。原因是 Ubuntu 环境缺少 Python 3.14 的开发头文件。
- 解决方法： 在 WSL Ubuntu 中安装对应的 Python 开发包：
  ```
  sudo apt update
  sudo apt install -y python3.14-dev
  ```
- 问题： FlashInfer 在 vLLM 初始化阶段尝试即时编译采样扩展时，报错 `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。原因是当前环境找不到 CUDA Toolkit 中的 `nvcc` 编译器。虽然 PyTorch 可以使用 GPU，但这不代表系统安装了完整的 CUDA Toolkit。
- 解决方法： 尝试关闭 FlashInfer 采样器，避免触发对应的 JIT 编译路径：
  ```
  import osos.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
  ```
  必须在创建 `LLM` 实例之前设置。若在 Notebook 中已经导入或初始化过 vLLM，建议重启 Kernel 后再运行。当前记录没有完整的后续日志，因此无法单独确认该设置是否就是最终解决此问题的原因。
- 问题： 最后一次启动时，vLLM 再次报错 `Free memory on device cuda:0 (6.95/8.0 GiB) ... desired GPU memory utilization (0.9, 7.2 GiB)`。原因是显存利用率设置为 `0.90`，对应约 `7.2 GiB` 的显存预算，高于当时可用的 `6.95 GiB`。
- 解决方法： 将 `gpu_memory_utilization` 从 `0.90` 降至 `0.80`，使预算降至约 `6.4 GiB`：
  ```
  vllm_model = LLM(    model=model_path,    gpu_memory_utilization=0.80,    max_model_len=8192,)
  ```