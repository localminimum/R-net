# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

Tensorflow implementation of https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
![Alt text](/../demo/screenshots/demo.gif?)

The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/). Pretrained GloVe embeddings are used for both words (https://nlp.stanford.edu/projects/glove/) and characters (https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt).

**As of 26 Feb 2018, thanks to @theSage21 we have a working demo of R-net!**

## Requirements
  * Python2.7
  * NumPy
  * tqdm
  * spacy
  * TensorFlow==1.2

# Downloads and Setup
Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ pipenv install
$ bash setup.sh
$ pipenv shell
$ python process.py --reduce_glove True --process True
```

# Training / Testing / Debugging / Interactive Demo
You can change the hyperparameters from params.py file to fit the model in your GPU. To train the model, run the following line.
```shell
$ python model.py
```
To test or debug your model after training, change mode="train" to debug or test from params.py file and run the model.

**To use demo, put batch size = 1**

# Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=r-net:train/
```
![Alt text](/../dev/screenshots/graph.png?raw=true "Tensorboard Graph")

# Log
**26/02/18**
As of 26th Feb 2018, thanks to @theSage21 we have an html demo that can easily be launched to user's local host and try out R-net on custom paragraphs and questions.

**18/10/17**
After some hyperparameter searching, our model quickly reaches EM/F1 score of 50/60 in 4 hours with the hyperparameters suggested in params.py file. However, it quickly overfits after that. **Current best model reaches EM/F1 of 55/67 on dev set**.

**05/09/17**
After rewriting the architectures, the model converges with full dataset and it takes about 20 hours to reach F1/EM=67/60 on training set and 40/30 on dev set. with batch size of 54. Reproducing the results obtained by R-Net in the original paper is a new work in progress.

**02/09/17**
One of the challenges I faced while training was to fit a minibatch of size 32 or larger into my GTX 1080. Since SQuAD dataset displayed high variance in data, higher batch size was essential in training (otherwise the model doesn't converge). Reducing GPU memory usage significantly to fit batch size of 32 and higher is a work in progress. If you have any suggestions on reducing the GPU memory usage, please put forward a pr.

**27/08/17**
As a sanity check I trained the network with 3000 independent randomly sampled question-answering pairs. With my GTX 1080, it took about 4 hours and a half for the model to get the gist of what's going on with the data. With full dataset (90,000+ pairs) we are expecting longer time for convergence. Some sort of normalization method might help speed up convergence (though the authors of the original paper didn't mention anything about the normalization).
