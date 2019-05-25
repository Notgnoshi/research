# CSR-Research

Graduate thesis research, paper, notes, and experiments.

---

## Notes on Jupyter Notebooks

Since Jupyter and related libraries are used by *many* local projects, it's useful to install Jupyter site-wide and create a new kernel specific to this project.
Source the project virtualenv and create a new kernel as follows

```shell
source ~/.virtualenvs/research/bin/activate
python3 -m ipykernel install --user --name=research
```

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

### CUDA 10.0

First, you need a recent version of `nvidia-driver` (I have `nvidia-driver-418`). Since my machines
need to Nvidia drivers before they will boot successfully, this was already done.

Then download the CUDA Toolkit 10.0 from [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).
I chose the runfile because I wanted to do a local installation. Download the base installer and any
patches.

Then run the installers. For each, I used `~/.local/cuda-10.0` as the installation path.

```shell
chmod +x *.run
./cuda_10.0.130_410.48_linux.run --override
```

Then I added the following to my `~/.bashrc`

```shell
export PATH="$HOME/.local/cuda/bin${PATH:+:${PATH}}"
export LIBRARY_PATH="$HOME/.local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
```

and symlinked `~/.local/cuda` to `~/.local/cuda-10.0`.

```shell
ln -s ~/.local/cuda-10.0 ~/.local/cuda
```

This way, I can symlink `/usr/local/cuda` to `~/.local/cuda` on the Opp Lab machines and have
everything work.

### Tensorflow-GPU 2.0.0-rc0

Versions of Tensorflow greater than 1.12.0 require CUDA 10.
Install Tensorflow 1.12.0 with the following

```shell
pip install --user --upgrade tensorflow-gpu==2.0.0rc0
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

Installing spaCy with GPU support requires GCC-6, but I don't want to use GCC-6 as the default, so
we have to jump through some hoops.

```shell
sudo update-alternatives --remove gcc /usr/bin/gcc
sudo update-alternatives --remove g++ /usr/bin/g++

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
pip install --user --upgrade spacy[cuda100]
python3 -m spacy download en --user
```

Then revert GCC back to its default version by running

```shell
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

again.

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

### NLTK Datasets

Run the following to download the NLTK datasets I use.

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
