# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

Tensorflow implementation of https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

Currently haven't trained with full dataset.
Trained using 3000 independent randomly sampled question-answering pairs.
![Alt text](/../dev/screenshots/figure.png?raw=true "Training error")

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
To train the model, run the following lines.
```shell
$ python model.py
```
# Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=r-net:r_net/
```
