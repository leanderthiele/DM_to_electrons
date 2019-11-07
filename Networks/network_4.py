"""
5 Levels, all possible concatenations
"""

this_network = {#{{{
    'NLevels': 5,

    'Level_0': {
        'concat': True,
        'in': [
                {
                    'inplane': 1,
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
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 64,
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
                    'crop_output': True,
                },
        ],
    },

    'Level_3': {
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
        ],
        'out': [
                {
                    'inplane': 512,
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

    'Level_4': {
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
                    'outplane': 256,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': True,
                }
        ],
    },
}#}}}
