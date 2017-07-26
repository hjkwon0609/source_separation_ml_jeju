class EmbeddingConfig:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    use_vpnn = False
    use_gru = False

    num_freq_bins = 513
    num_time_frames = 1876

    batch_size = 1000 if use_vpnn else 7
    output_size = num_freq_bins * 2
    num_hidden = 1024 # 1024

    num_layers = 2

    num_epochs = 5000
    l2_lambda = 0
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999

    file_reader_min_after_dequeue = 10
    file_reader_capacity = batch_size * 10

    n_fft = 1024
    hop_length = 256

    embedding_dim = 50  # 50 for general training

    keep_prob = 0.5

    num_segments = 4

    # overfit dims


