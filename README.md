# CSR-Research

Graduate thesis research, paper, notes, and experiments.

---

## Notes on Jupyter Notebooks

Many, of not all of the Jupyter notebooks in this project use Python libraries located elsewhere in
this project. Rather than edit `sys.path` relative to each of the notebooks whenever importing one
of those libraries is necessary, run the Jupyter server with the command

```shell
PYTHONPATH=$(pwd) jupyter lab
```

*in the root directory of this repository*.

Since I use the notebooks for development, including the development of those libraries, it is helpful
to automatically reload the libraries when they change. Do this by adding the Jupyter magics

```python
%load_ext autoreload
%autoreload 2
%aimport data # library name
```

It's also nice to use better plot formats

```python
%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

## Dependency Installation Notes

Some of the dependencies required a bit more work to install, so I'll make a note here so I don't
forget how.

### CUDA 9.0

First, you need a recent version of `nvidia-driver` (I have `nvidia-driver-415`). Since my machines
need to Nvidia drivers before they will boot successfully, this was already done.

Then download the CUDA Toolkit 9.0 from [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).
I chose the runfile because I wanted to do a local installation. Download the base installer and any
patches.

I had to install these dependencies before the installer would run

```shell
sudo apt install libglu1-mesa libglu1-mesa-dev libxmu-dev
```

Then run the installers. For each, I used `~/.local/cuda-9.0` as the installation path.

```shell
chmod +x *.run
./cuda_9.0.176_384.81_linux.run --override
./cuda_9.0.176.1_linux.run
./cuda_9.0.176.2_linux.run
./cuda_9.0.176.3_linux.run
./cuda_9.0.176.4_linux.run
```

Then I added the following to my `~/.bashrc`

```shell
# export PATH="/usr/local/cuda/bin:${PATH:+:${PATH}}"
# export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export PATH="$HOME/.local/cuda/bin${PATH:+:${PATH}}"
export LIBRARY_PATH="$HOME/.local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
```

and symlinked `~/.local/cuda` to `~/.local/cuda-9.0`.

```shell
ln -s ~/.local/cuda-9.0 ~/.local/cuda
```

This way, I can symlink `/usr/local/cuda` to `~/.local/cuda` on the Opp Lab machines and have
everything work.

### CUDNN 7.4.2 for CUDA 9.0

Then download the cuDNN v7.4.2 Library for Linux tarball from [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
and extract

```shell
tar -xvf cudnn-9.0-linux-x64-v7.4.1.5.tgz
cp cuda/lib64/libcudnn* ~/.local/cuda-9.0/lib64/
cp cuda/include/cudnn.h ~/.local/cuda-9.0/include/
```

I also installed libcupti

```shell
sudo apt install libcupti-dev
```

### Tensorflow-GPU 1.12.0

Versions of Tensorflow greater than 1.12.0 require CUDA 10.
Install Tensorflow 1.12.0 with the following

```shell
pip install --user --upgrade tensorflow-gpu==1.12.0
```

and test by running the following in a python interpreter

```python
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
```

on success, it should list your gpu

```json
device_type: "GPU",
memory_limit: 6733519258,
locality: {
    bus_id: 1,
    links: {}
},
incarnation: 240771873777457974,
physical_device_desc: "device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1"
```

### spaCy 2.1.3

```shell
pip install --user --upgrade spacy[cuda90]
python3 -m spacy download en --user
```

## TensorFlow GPU Usage

TensorFlow, by default allocates your entire damn GPU, even when working with a small model. This
is frustrating, so do the following to disable this.

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

set_session(tf.Session(config=config))
```
