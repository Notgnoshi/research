FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS prerequisites

# Necessary for jupyter extensions
RUN apt-get update -qq && apt-get install -yqq \
    gcc-7 \
    nodejs \
    npm \
    python3 \
    python3-pip

# Add a user to do everything inside of.
RUN useradd -ms /bin/bash nots
USER nots
WORKDIR /home/nots/
ENV HOME=/home/nots/
ENV PATH="$HOME/.local/bin${PATH:+:${PATH}}"

# Install Python dependencies
# Don't use requirements, because I want Docker to be able to cache layers.

# Install intensive libraries
RUN pip3 install --no-cache-dir --user --upgrade 'tensorflow-gpu>=2.0.0b1'
RUN pip3 install --no-cache-dir --user --upgrade spacy[cuda]

# Install development tools
RUN pip3 install --no-cache-dir --user --upgrade \
    black \
    ipython \
    isort \
    jedi \
    jupyter \
    jupyterlab \
    jupytext \
    jupyterlab_code_formatter \
    nb-pdf-template \
    pydocstyle \
    pylint \
    pytest

# Install General scientific libraries
RUN pip3 install --no-cache-dir --user --upgrade \
    h5py \
    matplotlib \
    numpy \
    pandas \
    Pygments \
    scipy \
    seaborn \
    sympy

RUN pip3 install --no-cache-dir --user --upgrade \
    gensim \
    grakel-dev[lovasz] \
    langdetect \
    langid \
    networkx \
    nltk \
    requests \
    requests-html \
    scikit-learn \
    syllables \
    tensorflow-hub \
    textblob \
    webcolors \
    wordcloud

RUN python3 -m nltk.downloader stopwords wordnet averaged_perceptron_tagger punkt
RUN pip3 install --no-cache-dir --user --upgrade https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5

RUN python3 -m nb_pdf_template.install
RUN jupyter nbextension install --user --py jupytext
RUN jupyter nbextension enable jupytext --user --py
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter
RUN jupyter serverextension enable --user --py jupyterlab_code_formatter
RUN jupyter lab build

# Fix https://github.com/psf/black/issues/1223
RUN mkdir -p "$HOME/.cache/black/19.10b0"

RUN mkdir -p "$HOME/.jupyter" \
    && echo "c.LatexExporter.template_file = 'classicm'" >> "$HOME/.jupyter/jupyter_nbconvert_config.py" \
    && echo "c.LatexExporter.template_file = 'classicm'" >> "$HOME/.jupyter/jupyter_notebook_config.py" \
    && echo "c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'" >> "$HOME/.jupyter/jupyter_notebook_config.py" \
    && echo 'c.ContentsManager.default_jupytext_formats = "ipynb,md"' >> "$HOME/.jupyter/jupyter_notebook_config.py"

# TODO: Install latex stuff?
# TODO: This is a *massive* image. Try to cut down its size by adding another layer?
# TODO: Add a volume for the pip cache dir so it doesn't have to redownload everything every time.
