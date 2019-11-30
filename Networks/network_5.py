"""
4 Levels, all possible concatenations,
a bit deeper than network_2
and twice as many feature channels than network 3

can run this network with batch_size 16 on 4 GPUs
"""

this_network = {#{{{
    'NLevels': 4,

    'Level_0': {
        'concat': True,
        'in': [
                {
                    'inplane': 1,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
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
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 128,
                    'outplane': 1,
                    'conv_kw': {
                                    'stride': 1,
                                    'kernel_size': 1,
                                    'padding': 0,
                               },
                    'batch_norm': None,
                    'activation': None,
                },
        ],
    },

    'Level_1': {
        'concat': True,
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
                {
                    'inplane': 128,
                    'outplane': 128,
                },
        ],
        'out': [
                {
                    'inplane': 256,
                    'outplane': 128,
                },
                {
                    'inplane': 128,
                    'outplane': 128,
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
                    'crop_output': True,
                },
        ],
    },

    'Level_2': {
        'concat': True,
        'in': [
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
                    'outplane': 256,
                },
        ],
        'out': [
                {
                    'inplane':  512,
                    'outplane': 256,
                },
                {
                    'inplane':  256,
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
                    'crop_output': True,
                },
        ],
    },

    'Level_3': {
        'through': [
                {
                    'inplane': 256,
                    'outplane': 512,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                },
                {
                    'inplane': 512,
                    'outplane': 256,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': True,
                },
        ],
    },
}#}}}
