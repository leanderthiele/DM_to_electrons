this_config = {
    'box_sidelength': 2048,

    # these numbers are for the 2048 box,
    # they get rescaled for other box sizes
    # so that the physical cubes are always of
    # equal size
    'DM_sidelength': 64,
    'gas_sidelength': 32,

    'eval_period': 128,
    'loss_function': 'MSELoss',
    'optimizer': 'Adam',
    'learning_rate': 1e-4,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'num_workers': 1,
    'Nsamples': {
        'training': 8192,
        'validation': 256,
        },

    'train_time': 350.0, # minutes

    'output_path': '/home/lthiele/DM_to_electrons/Outputs/',
}
