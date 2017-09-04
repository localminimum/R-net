class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 100
    train_prop = 0.9 # Not implemented atm
    data_dir = "./data/"
    train_dir = data_dir + "trainset/"
    dev_dir = data_dir + "devset/"
    logdir = "./train/adadelta"
    glove_dir = "glove.840B.300d.txt"
    glove_char = "glove.840B.300d.char.txt"
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09"

    # Data dir
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    # Training
    debug = True # Set it to True to debug the computation graph
    test = False
    learning_rate = 1 # Adadelta doesn't require initial learning rate
    dropout = 0.2 # dropout probability
    optimizer = "adadelta" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 48
    save_steps = 50 # Save the model at every 50 steps
    clip = False # clip gradient norm
    norm = 5.0 # global norm

    # Architecture
    max_len = 200 # Maximum number of words in each passage context
    vocab_size = 2196018 # Number of vocabs in glove.840B.300d.txt + 1 for an unknown token
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an unknown character
    emb_size = 300 # Embeddings size for both words and characters
    attn_size = 64 # RNN celland attention module size
    num_layers = 3 # Number of layers at question-passage matching and self matching network
