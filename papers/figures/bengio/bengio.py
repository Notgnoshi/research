#!/usr/bin/env python3
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
PNN_PATH = REPO_ROOT / "depends" / "PlotNeuralNet"
sys.path.append(str(PNN_PATH))

from pycore.tikzeng import *
from pycore.blocks import *

# TODO: I *really* don't like using PlotNeuralNet.
architecture = [
    to_head(PNN_PATH),
    to_cor(),
    to_begin(),

    to_Pool(name="word1", offset="(0, 0, 0)", to="(0, 0, 0)", width=2, height=8, depth=2, caption=""),
    to_Pool(name="word2", offset="(0, -1.6, 0)", to="(0, 0, 0)", width=2, height=8, depth=2, caption=""),
    to_Pool(name="word3", offset="(0, -3.2, 0)", to="(0, 0, 0)", width=2, height=8, depth=2, caption=""),
    to_Pool(name="wordn", offset="(0, -6.4, 0)", to="(0, 0, 0)", width=2, height=8, depth=2, caption="$x$"),

    to_Pool(name="hidden1", offset="(2, -3.2, 0)", to="(0, 0, 0)", width=2, height=32, depth=2, caption="$H$"),

    to_SoftMax(name="softmax", s_filer="", offset="(4, -3.2, 0)", to="(0, 0, 0)", width=2, height=48, depth=2, caption="softmax"),

    to_connection(of="word3", to="hidden1"),
    to_connection(of="hidden1", to="softmax"),
    to_end(),
]

to_generate(architecture, "bengio.tex")
