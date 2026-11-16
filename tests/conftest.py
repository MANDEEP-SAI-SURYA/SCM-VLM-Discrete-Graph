"""The tests are CPU tests (stubbed DETR / BERT, models built on the CPU).  On a GPU machine
(e.g. Colab) configs.config.DEVICE would be "cuda" and train.py would move the inputs to the GPU
while the test models stay on the CPU, so hide the GPU from the test process."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
