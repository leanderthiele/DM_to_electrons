this_config = {
    'box_sidelength': 2048,

    # these numbers are for the 2048 box,
    # they get rescaled for other box sizes
    # so that the physical cubes are always of
    # equal size
    'DM_sidelength': 64,
    'gas_sidelength': 32,

    # loss function
    'loss_function': 'MSELoss',
    'loss_function_MSELoss_kw': {
                                },
    'loss_function_L1Loss_kw' : {
                                    'reduction': 'mean', # keep 'mean' to compare different batch sizes
                                },

    # optimizer
    'optimizer': 'Adam',
    'optimizer_Adam_kw': {
                            'lr': 1e-4,
                            'betas': (0.9, 0.999),
                            'eps': 1e-8,
                            'weight_decay': 0.0,
                         },

    # data loader
    'data_loader_kw': {
                        'batch_size': 8,
                        'num_workers': 1,
                        'shuffle': False,
                      },

    # describes individual epochs
    'Nsamples': {
                    'training': 8192,
                    'validation': 256,
                },

    # learning rate scheduler
    'lr_scheduler': 'None',
    'lr_scheduler_ReduceLROnPlateau_kw': {
                                            'factor': 0.2,
                                            'patience': 10,
                                            'verbose': True,
                                         },


    'train_time': 2870.0, # minutes

    'output_path': '/home/lthiele/DM_to_electrons/Outputs/',
}
