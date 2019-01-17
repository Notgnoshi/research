# CSR-Research

Graduate thesis research, paper, notes, and experiments.

---

## Dependency Installation Notes

Some of the dependencies required a bit more work to install, so I'll make a note here so I don't forget how.

### CUDA 9.0

First, you need a recent version of `nvidia-driver` (I have `nvidia-driver-415`). Since my machines need to Nvidia drivers before they will boot successfully, this was already done.

Then download the CUDA Toolkit 9.0 from [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). I chose the runfile because I wanted to do a local installation. Download the base installer and any patches.

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

This way, I can symlink `/usr/local/cuda` to `~/.local/cuda` on the Opp Lab machines and have everything work.

### CUDNN 7.4.2 for CUDA 9.0

Then download the cuDNN v7.4.2 Library for Linux tarball from [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) and extract

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

Install Tensorflow with the following

```shell
pip install --user --upgrade tensorflow-gpu
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

### spaCy 2.0.18

Installing spaCy with GPU support requires GCC-6, but I don't want to use GCC-6 as the default, so we have to jump through some hoops.

```shell
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++

sudo apt install gcc-6 g++-6

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 20

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
```

Then temporarily pick GCC-6 as the default by running

```shell
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

and pick GCC-6 from the list of options.

Then install spaCy with GPU support with

```shell
pip install --user --upgrade spacy[cuda90]
python3 -m spacy download en --user
```

Then revert GCC back to its default version by running

```shell
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

again.

## Keras GPU Usage

Keras, by default allocates your entire damn GPU, even when working with a small model. This is frustrating, so do the following to disable this.

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

set_session(tf.Session(config=config))
```

## TODO

* Experiment with different Python NLP [libraries](https://kleiber.me/blog/2018/02/25/top-10-python-nlp-libraries-2018/)
* Read Stanford [book](https://web.stanford.edu/~jurafsky/slp3/) on NLP
* Watch Stanford [lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) oriented towards deep learning
