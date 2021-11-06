
data_dir = './data'

data_type = 'val2017'

val_path = f"{data_dir}/uitviic_captions_{data_type}.json"

# results: trên tập test, eval: trên tập eval
subtypes = ['results', 'eval']

alg_name = 'machineoutput'

[res_file, eval_imgs_file, eval_file]= \
    [f'{data_dir}/results/captions_{data_type}_{alg_name}_{subtype}.json' for subtype in subtypes]

# tokenizer = 'nltk'
tokenizer = 'pyvi'



    
