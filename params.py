class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 10
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
    mode = "demo" # case-insensitive options: ["train", "test", "debug"]
    dropout = 0.2 # dropout probability, if None, don't use dropout
    zoneout = None # zoneout probability, if None, don't use zoneout
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 1 if mode is not "test" else 100# Size of the mini-batch for training
    save_steps = 50 # Save the model at every 50 steps
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'learning_rate':1, 'rho': 0.95, 'epsilon':1e-6},
                'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8},
                'gradientdescent':{'learning_rate':1},
                'adagrad':{'learning_rate':1}}

    # Architecture
    SRU = True # Use SRU cell, if False, use standard GRU cell
    max_p_len = 300 # Maximum number of words in each passage context
    max_q_len = 30 # Maximum number of words in each question context
    max_char_len = 16 # Maximum number of characters in a word
    vocab_size = 91604 # Number of vocabs in glove.840B.300d.txt + 1 for an unknown token
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an unknown character
    emb_size = 300 # Embeddings size for words
    char_emb_size = 8 # Embeddings size for characters
    attn_size = 75 # RNN cell and attention module size
    num_layers = 3 # Number of layers at question-passage matching
    bias = True # Use bias term in attention
