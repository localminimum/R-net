class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 100
    train_prop = 0.9 # Not implemented atm
    data_dir = "./data/"
    train_dir = data_dir + "trainset/"
    dev_dir = data_dir + "devset/"
    logdir = "./train/train"
    glove_dir = "./glove.840B.300d.txt" # Glove file name (If you want to use your own glove, replace the file name here)
    glove_char = "./glove.840B.300d.char.txt" # Character Glove file name
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09" # Directory to Stanford coreNLP tool

    # Data dir
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    # Training
    mode = "train" # case-insensitive options: ["train", "test", "debug"]
    max_q_len = 37
    dropout = None # dropout probability if None, don't use dropout
    optimizer = "adadelta" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 50 if mode is not "test" else 100# Size of the mini-batch for training
    save_steps = 50 # Save the model at every 50 steps
    clip = False # clip gradient norm
    norm = 5.0 # global norm
    # Change the hyperparameters of your learning algorithms here
    opt_arg = {'adadelta':{'learning_rate':1, 'rho': 0.95, 'epsilon':1e-6},
                'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8},
                'gradientdescent':{'learning_rate':1},
                'adagrad':{'learning_rate':1}}

    # Architecture
    max_len = 200 # Maximum number of words in each passage context
    vocab_size = 2196018 # Number of vocabs in glove.840B.300d.txt + 1 for an unknown token
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an unknown character
    emb_size = 300 # Embeddings size for both words and characters
    attn_size = 75 # RNN celland attention module size
    num_layers = 3 # Number of layers at question-passage matching and self matching network
    bias = True # Use bias term in attention
