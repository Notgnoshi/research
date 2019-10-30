# CSR-Research

Graduate research, notes, and experiments on the generation of haiku poetry with machine learning.

<!-- Autogenerating the TOC insists on adding the title. Too bad. -->
- [CSR-Research](#csr-research)
  - [Installing the Dependencies](#installing-the-dependencies)
    - [Clone the Repository and Create a Virtualenv](#clone-the-repository-and-create-a-virtualenv)
    - [CUDA 10.0](#cuda-100)
    - [Cudnn](#cudnn)
    - [GCC Compiler Version](#gcc-compiler-version)
    - [Installing the Actual Dependencies](#installing-the-actual-dependencies)
    - [Run the Setup Script](#run-the-setup-script)
  - [Miscellaneous Tips and Tricks](#miscellaneous-tips-and-tricks)
    - [Using Jupyter with a Virtualenv](#using-jupyter-with-a-virtualenv)
    - [Common Jupyter Notebook Configuration](#common-jupyter-notebook-configuration)
    - [Checking Tensorflow-GPU for Proper Installation](#checking-tensorflow-gpu-for-proper-installation)
    - [Tensorflow GPU Allocation Trick](#tensorflow-gpu-allocation-trick)

## Installing the Dependencies

Some of the dependencies required a bit more work to install, so I'll make a note here so I don't forget how.
It's necessary to install CUDA 10.0 (not 10.1), and switch your GCC compiler to GCC-7 or earlier before installing the dependencies.

### Clone the Repository and Create a Virtualenv

```shell
git clone https://github.com/Notgnoshi/research.git
cd research
pip install --user virtualenv
virtualenv .venv --prompt="(research) "
source .venv/bin/activate
```

### CUDA 10.0

The only machine in the Opp Lab with CUDA 10.0 is `linux06`.
The rest have CUDA 10.1.

First, you need a recent version of `nvidia-driver` (I have `nvidia-driver-418`).
Since my machines need to Nvidia drivers before they will boot successfully, this was already done.

Then download the CUDA Toolkit 10.0 from [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).
I chose the runfile because I wanted to do a local installation.
Download the base installer and any patches.

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

This way, I can symlink `/usr/local/cuda` to `~/.local/cuda` on the Opp Lab machines and have everything work.

### Cudnn

**TODO**

### GCC Compiler Version

Installing compiling CUDA 10.0 kernels with `nvcc` requires GCC-7 or earlier, but I don't want to use GCC-7 as the default, so we have to jump through some hoops.
Install GCC-7 and temporarily set it as the default before installing the project dependencies.

First, check the GCC version before continuing.

```shell
gcc --version
```

Proceed if you have a version later than GCC-7.

```shell
sudo update-alternatives --remove gcc /usr/bin/gcc
sudo update-alternatives --remove g++ /usr/bin/g++

sudo apt install gcc-7 g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 20

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
```

Then temporarily pick GCC-7 as the default by running

```shell
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

and pick GCC-7 from the list of options.

**Warning:** This is a potentially dangerous action, and I can personally testify that I've horribly broken the C++ standard library installation by doing so.
Perhaps the above instructions are flawed, or perhaps I did something else to break my installation.
In either case, I found the flaw some eight months later while working on an unrelated project, and spent several days working on a solution before finding that reinstalling my compilers fixed things.
Your mileage may vary.

### Installing the Actual Dependencies

After the above prerequisite steps have been performed, installing the actual Python libraries is rather simple.
Note that the libraries are substantial, and that the virtualenv will take some 6+ gigabytes of space.
Sorry.

```shell
source .venv/bin/activate
pip install --upgrade --requirement requirements.txt
python3 -m spacy download en --user
```

Be sure to revert GCC back to its default version if you changed it temporarily.

```shell
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

### Run the Setup Script

In order to save space in the repository, the cleaned dataset is not tracked under version control.
Run

```shell
haikulib/scripts/initialize.py
```

to setup the `data/` directory.
In particular, create the `data/haiku.csv` file containing the cleaned haiku and several initial data analysis steps.

## Miscellaneous Tips and Tricks

### Using Jupyter with a Virtualenv

Since Jupyter and related libraries are used by *many* local projects, it's useful to install Jupyter site-wide and create a new kernel specific to this project.
Source the project virtualenv and create a new kernel as follows

```shell
source .venv/bin/activate
python3 -m ipykernel install --user --name=research
```

Many, of not all of the Jupyter notebooks in this project use Python libraries located elsewhere in
this project. Rather than edit `sys.path` relative to each of the notebooks whenever importing one
of those libraries is necessary, run the Jupyter server with the command

```shell
PYTHONPATH=$(pwd) jupyter lab
```

*in the root directory of this repository*.

### Common Jupyter Notebook Configuration

Since I use the notebooks for development, including the development of those libraries, it is helpful
to automatically reload the libraries when they change. Do this by adding the Jupyter magics

```python
%load_ext autoreload
%autoreload 2
%aimport haikulib
```

It's also nice to use better plot formats

```python
%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Print pandas.DataFrame's nicely.
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
# Use the 16x9 aspect ratio, with a decent size.
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)
# Use a better default matplotlib theme.
sns.set()
```

### Checking Tensorflow-GPU for Proper Installation

Test if `tensorflow-gpu` was installed correctly by running the following.

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

### Tensorflow GPU Allocation Trick

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
