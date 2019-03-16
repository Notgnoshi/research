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

## TensorFlow GPU Usage

TensorFlow, by default allocates your entire damn GPU, even when working with a small model. This is frustrating, so do the following to disable this.

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

set_session(tf.Session(config=config))
```

## TODO

Overall TODOs

* Experiment with different Python NLP [libraries](https://kleiber.me/blog/2018/02/25/top-10-python-nlp-libraries-2018/)
* Read Stanford [book](https://web.stanford.edu/~jurafsky/slp3/) on NLP
* Watch Stanford [lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) oriented towards deep learning

Data Sources

* Document sources for dataset, and cite in paper.
* There are more haiku in a not-easily-parseable format [here](http://startag.tripod.com/HLpg1sep01.html) and [here](http://www.haikupoet.com/search.php)
* Look through the links [here](http://www.theheronsnest.com/archived_issues/connections/) and [here](https://www.ahapoetry.com/h_links.html) for more haikus.

Exploratory data analysis

* Analyze word frequencies before and after removing stop words. Consider the rarest words, and potentially remove the containing haikus from the dataset.
* Analyze common lines
* "grey" vs "gray" and other British spellings.
* Build a word cloud. This could be interesting to get a sense of what kind of language is in the dataset.
  * Build word cloud after removing stop words.
  * Identify haiku-specific stop words.
  * Build word cloud after stemming/lemmatization.
  * Do the same thing for flowers, colors, season bigrams, wind bigrams, etc.
* Interesting bi and tri-grams. '`*` hour', '`*` blossom(s)', '`*` season `*`', '`*` wind', `<flower name>` etc.
* What birds are mentioned. What flowers? What animals?
* There was a PyCon talk about the colors in works of literature. Can I do the same with haikus?
* Check whether Zipf's law holds for the haiku dataset.
* Find haikus that are very similar to each other (small edit distance)
* Find outliers
* There's an essay on haiku FAQ [here](http://haiku.ru/frog/alexey_def.htm). There are educational links [here](https://www.ahapoetry.com/Bare%20Bones/bbtoc%20intro.html) and [here](https://www.ahapoetry.com/all%20haiku%20info.html) on how to write haikus.

  Qualify each point with real data.

  ```text
  asshole questioning
  doesn't know about haiku
  5-7-5 bitch
  ```

* Attempt NLP/ML topic modeling.
* Attempt ML sentiment analysis.
* Can the haikus be clustered by their topics, sentiments, or something else?
  * The topics will be a vector of words? So can the vectors be clustered?
* Can an RNN find the appropriate line breaks?
  * Variable length inputs and outputs will be tricky
  * Look into seq2seq where there's an encoder and decoder at both ends of the RNN.
* How many haikus actually follow the 5-7-5 common pattern?
  * Need to find a good way to count syllables
  * Also look at number of lines
* Naive haiku generation with Markov chains and LSTM networks
* Think about grammatical evolution with some kind of encoding and production rules
