class Params():

    # data
    data_size = 80000
    num_epochs = 100
    data_dir = "./data/"
    logdir = "./train/fix_char"
    glove_dir = "glove.840B.300d.txt"
    glove_char = "glove.840B.300d.char.txt"
    target_dir = data_dir + "/indices.txt"
    q_word_dir = data_dir + "/words_questions.txt"
    q_chars_dir = data_dir + "/chars_questions.txt"
    p_word_dir = data_dir + "/words_context.txt"
    p_chars_dir = data_dir + "/chars_context.txt"
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09"

    # model
    debug = False
    max_len = 200
    save_steps = 50
    learning_rate = 1
    vocab_size = 2196018
    char_vocab_size = 95
    batch_size = 32
    train_prop = 0.9
    emb_size = 300
    attn_size = 75
    num_layers = 3
