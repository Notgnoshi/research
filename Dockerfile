FROM ubuntu:18.04

# Necessary for jupyter extensions
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -yqq \
    wget \
    python3 \
    python3-pip \
    texlive \
    texlive-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    cm-super \
    git

# Add a user to do everything inside of.
RUN useradd --create-home --shell /bin/bash container
USER container
WORKDIR /home/container/
ENV HOME=/home/container/
ENV PATH="$HOME/.local/bin${PATH:+:${PATH}}"

# Install Python dependencies
# Don't use requirements, because I want Docker to be able to cache layers.

RUN pip3 install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir --user --upgrade  \
    Pygments \
    annoy \
    black \
    commentjson \
    gensim \
    grakel-dev[lovasz] \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 \
    ipython \
    isort \
    jedi \
    jsonschema \
    jupyter \
    jupyterlab \
    jupyterlab_code_formatter \
    jupytext \
    langdetect \
    langid \
    matplotlib \
    nb-pdf-template \
    networkx \
    nltk \
    numpy \
    pandas \
    pydocstyle \
    pylint \
    pytest \
    requests \
    requests-html \
    scikit-learn \
    scipy \
    seaborn \
    spacy \
    syllables \
    sympy \
    textblob \
    transformers \
    webcolors \
    wordcloud

RUN python3 -m nltk.downloader stopwords wordnet averaged_perceptron_tagger punkt
RUN wget -qO- https://nodejs.org/dist/v12.16.1/node-v12.16.1-linux-x64.tar.xz | tar -xJ
ENV PATH="$HOME/node-v12.16.1-linux-x64/bin${PATH:+:${PATH}}"

RUN python3 -m nb_pdf_template.install \
    && jupyter nbextension install --user --py jupytext \
    && jupyter nbextension enable jupytext --user --py \
    && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
    && jupyter serverextension enable --user --py jupyterlab_code_formatter \
    && jupyter lab build

# Fix https://github.com/psf/black/issues/1223
RUN mkdir -p "$HOME/.cache/black/19.10b0" \
    && mkdir -p "$HOME/.jupyter" \
    && echo "c.LatexExporter.template_file = 'classicm'" >> "$HOME/.jupyter/jupyter_nbconvert_config.py" \
    && echo "c.LatexExporter.template_file = 'classicm'" >> "$HOME/.jupyter/jupyter_notebook_config.py" \
    && echo "c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'" >> "$HOME/.jupyter/jupyter_notebook_config.py" \
    && echo 'c.ContentsManager.default_jupytext_formats = "ipynb,md"' >> "$HOME/.jupyter/jupyter_notebook_config.py"

ENV PYTHONPATH=/workspaces/research

# TODO: Add a volume for the pip cache dir so it doesn't have to redownload everything every time.
