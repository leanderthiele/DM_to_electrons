this_config = {
    'box_sidelength': 2048,

    # these numbers are for the 2048 box,
    # they get rescaled for other box sizes
    # so that the physical cubes are always of
    # equal size
    'DM_sidelength' : 64,
    'gas_sidelength': 32,

    # loss function
    'loss_function': 'L1Loss',
    'loss_function_MSELoss_kw': {
                                    'reduction': 'mean',
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
                        'batch_size': 16,
                        'num_workers': 4,
                        'shuffle': False,
                        'drop_last': True,
                      },

    # artificial noise in target
    'gas_noise': 0.0,
    'DM_noise': 0.0,

    # sample selector
    'sample_selector_kw': {
                            'empty_fraction': 0.5,
                            'halo_weight_fct': 'lambda logM, dlogM : 1.0',
                            'pos_mass_file': '/tigress/lthiele/Illustris_300-1_Dark/important_halos.hdf5',
                          },

    # when to switch to the full network (after only seeing the model)
    'pretraining_epochs': 0,

    # target transformation
    'target_transformation': True,
    'target_transformation_kw': {
                                    # time constant for decay of the log term (in epochs)
                                    # alpha = delta + (1-delta) * exp( - epoch / tau)
                                    'tau': 300,
                                    # the transformation is given by
                                    # f(x) = kappa * alpha * log(1 + x/gamma) + (1-alpha)*x/gamma
                                    'gamma': 0.1,
                                    'kappa': 1.0,
                                    'delta': 0.0,
                                },

    # describes individual epochs
    'Nsamples': {
                    'training': 8192,
                    'validation': 512,
                },

    # learning rate scheduler
    'lr_scheduler': 'None',
    'lr_scheduler_ReduceLROnPlateau_kw': {
                                            'factor': 0.2,
                                            'patience': 10,
                                            'verbose': True,
                                         },

    'input_path' : '/tigress/lthiele/boxes/hdf5_files/',
    'output_path': '/scratch/gpfs/lthiele/Outputs/',
    'summary_path':'/scratch/gpfs/lthiele/summaries/'
}
