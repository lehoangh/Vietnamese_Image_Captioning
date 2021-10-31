import logging


class Config(object):
    # Paths
    img_path = 'data/resized2017'
    train_path = 'data/annotations/uitviic_captions_train2017.json'
    val_path = 'data/annotations/uitviic_captions_val2017.json'
    test_path = ''
    vocab_path = 'data/vocab.pkl'
    model_path = 'models/'
    machine_output_path = 'data/machine_output.json'
    tokenizer = 'pyvi'

    optimizer = "adam"

    threshold = 1
    learning_rate = 0.001
    # num_epochs = 15
    num_epochs = 5
    new_epochs = 10
    crop_size = 224
    batch_size = 128
    embed_size = 256 # Dimension of word embedding vectors
    hidden_size = 512 # Dimension of lstm hidden states
    num_layers = 1 # NUmber of layers in lstm   
    num_workers = 2

    save_step = 1
    # save_step = 10

    log_step = 10

    # Các mô hình encoder và decoder sử dụng để test ảnh mới
    encoder_path = 'models/encoder-10-100.ckpt'
    decoder_path = 'models/decoder-10-100.ckpt'

    """Setting file to logging"""
    def setup_logging():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) # process everything, even if everything isn't printed
        fh = logging.FileHandler(__name__)
        fh.setLevel(logging.INFO) # or any level you want
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(filename)s: %(asctime)-8s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger


    # logger = setup_logging()

