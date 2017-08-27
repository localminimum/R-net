# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

Tensorflow implementation of https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

Work in progress.

## Requirements
  * NumPy
  * librosa
  * tqdm
  * TensorFlow == 1.2

# Setup
Once you clone this repo, run the following lines from bash.
```shell
$ pip install -r requirements.txt
$ bash setup.sh
$ python process.py --process True
```
# Training
```shell
$ python model.py
```
# Tensorboard
Run tensorboard.
```shell
$ tensorboard --logdir=r-net:r_net/
```
