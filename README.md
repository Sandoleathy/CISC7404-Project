# Install packages
## Using mirror source
Cuda reference is very larg. Using default source to download takes very long time.

put file `uv.toml` inside uv config folder to use mirror source.

## Install
Install requirements
```
uv sync
```

Add new packages
```
uv add xxx
```

Lock packages
```
uv lock
```

# Enter virtual environment
It is not necessary to enter virtual environment when using uv. But sometimes we still need to enter a virtual environment to install some package or use some local package.

uv will automatically generate virtual environment named `venv` using command `uv sync`.

Windows cmd
```
.venv\Script\activate
```
Linux
```
source .venv\bin\activate
```


## PyTorch install
Because of the difference between CPU and GPU platform. You need to manully install PyTorch depend on what platform you are using.

**make sure you are already enter the virtual environment before you execute commands below**
### CPU Platform
```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GPU Platform
firstly check your cuda version
- Linux
```
nvcc --version
```

go to https://pytorch.org/ to get the exact command

for example cuda 13.2

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Add `UV_HTTP_TIMEOUT=300` if it takes too much time to download.