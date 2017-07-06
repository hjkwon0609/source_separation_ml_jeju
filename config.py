class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    num_freq_bins = 514
    num_time_frames = 5169

    batch_size = 20
    output_size = num_freq_bins * 2
    num_hidden = 512

    num_layers = 3

    num_epochs = 50
    l2_lambda = 0.01
    lr = 1e-2