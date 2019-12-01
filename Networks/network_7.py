"""
5 Levels, all possible concatenations
model fed in in Level_0 out
"""

this_network = {#{{{
    'NLevels': 5,
    'feed_model': True,

    'model_block': [
                {
                    'inplane': 1,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
    ],

    'Level_0': {
        'concat': False,
        'in': [
                {
                    'inplane': 1,
                    'outplane': 32,
                },
        ],
        'out': [ # gets 32^3 input
                { # Final layer, collapse to single channel
                    'inplane': 64,
                    'outplane': 1,
                    'conv_kw': {
                                    'stride': 1,
                                    'kernel_size': 1,
                                    'padding': 0,
                               },
                    'batch_norm': None,
                },
        ],
    },

    'Level_1': {
        'concat': True,
        'resize_to_gas': True,
        'in': [
                {
                    'inplane': 32,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
        ],
        'out': [
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32, # remove one channel to make space for model
                },
        ],
    },

    'Level_2': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 32,
                    'outplane': 64,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
        ],
        'out': [
                {
                    'inplane': 128,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 32,
                },
        ],
    },

    'Level_3': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 64,
                    'outplane': 128,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                },
        ],
        'out': [
                {
                    'inplane':  256,
                    'outplane': 128
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 64,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                },
        ],
    },

    'Level_4': {
        'through': [
                {
                    'inplane': 128,
                    'outplane': 256,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 256,
                    'outplane': 256,
                },
                {
                    'inplane': 256,
                    'outplane': 256,
                },
                {
                    'inplane': 256,
                    'outplane': 128,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                },
        ],
    },
}#}}}
