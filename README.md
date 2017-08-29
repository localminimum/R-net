# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

Tensorflow implementation of https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
![Alt text](/../dev/screenshots/architecture.png?raw=true "R-NET")

Training with full SQuAD dataset is currently a work in progress.

The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/).

## Requirements
  * NumPy
  * tqdm
  * TensorFlow == 1.2

# Downloads and Setup
Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ pip install -r requirements.txt
$ bash setup.sh
$ python process.py --process True
```

# Training
You can change the hyperparameters from params.py file.
To train the model, run the following line.
```shell
$ python model.py
```

# Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=r-net:r_net/
```
![Alt text](/../dev/screenshots/graph.png?raw=true "Tensorboard Graph")

# Note
As a sanity check I trained the network with 3000 independent randomly sampled question-answering pairs. With my GTX 1080, it took about 4 hours and a half for the model to get the gist of what's going on with the data. With full dataset (90,000+ pairs) we are expecting longer time for convergence.

Some sort of normalization method might help speed up convergence (though the authors of the original paper didn't mention anything about the normalization).

![Alt text](/../dev/screenshots/figure.png?raw=true "Training error")
