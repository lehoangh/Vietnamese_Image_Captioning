import logging


class Config(object):
    # Paths
    img_path = 'data/resized2017'
    img_path_val = 'data/resized_val2017'
    train_path = 'data/annotations/uitviic_captions_train2017.json'
    val_path = 'data/annotations/uitviic_captions_val2017.json'
    test_path = ''
    vocab_path = 'data/vocab.pkl'
    model_path = 'models/'
    results_path = 'data/results/'
    # results: trên tập test, eval: trên tập eval
    machine_output_path = results_path + 'captions_val2017_machineoutput_eval.json'
    tokenizer = 'pyvi'

    optimizer = "adam"

    threshold = 1
    learning_rate = 0.001
    # num_epochs = 15
    # Số lần epochs để huấn luyện lần đầu tiên
    num_epochs = 5
    # Số lần epochs để huấn luyện các lần tiếp theo
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

    # Các mô hình encoder và decoder sử dụng để test ảnh mới, mô hình có loss thấp nhất
    encoder_path = 'models/encoder-11-40-20211031_155008.ckpt'
    decoder_path = 'models/decoder-11-40-20211031_155008.ckpt'

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

