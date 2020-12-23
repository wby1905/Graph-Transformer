import torch as t


class Config:
        
    K = 2
    m = 3
    W = 'all'
    top_k = 30

    conv_num = K - 1
    gcn_num = 1
    conv_channel = 256
    conv1d_channel = 32
    fc_channel = 32
    fc_layers_num = 5

    num_workers = 0
    batch_size = 16
    lr = 0.0001
    lr_decay = 0.5
    weight_decay = 0
    drop_out = 0
    eps = 0.05
    
    max_epoch = 100
    print_freq = 500
    val_split = 0.2
    k_fold = 10

    dataset = 'MUTAG'
    classes = 2
    input_features = 1433

    seed = 777

    load_model_path = './checkpoints'
    dataset_dir = './datasets/'
    save_processed_dir = './datasets/processed/'
    result_file = 'result.csv'

    use_gpu = True
    device = t.device('cuda') if use_gpu else t.device('cpu')


opt = Config()
