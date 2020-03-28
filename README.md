# CSR-Research

Graduate research, notes, and experiments on the generation of haiku poetry with machine learning.

## Installing Docker

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# This doesn't work on Ubuntu 19.10. If on 19.10, replace '$(lsb_release -cs)' with 'disco'
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
reboot
```

<!-- ```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker run --gpus all nvidia/cuda:10.1-base nvidia-smi
``` -->

## Set up Docker for this Project

Use the provided `Makefile`.

```bash
make help
# Now's a good time for a cup of coffee.
make docker-build
make init-data
make jupyter
```

## Miscellaneous Tips and Tricks

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

<!-- ### Checking Tensorflow-GPU for Proper Installation

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
``` -->
