this_network = {#{{{
    'NLevels': 3,

    'Level_0': {
        'concat': True,
        'in': [
                {
                    'inplane': 1,
                    'outplane': 16,
                },
                {
                    'inplane': 16,
                    'outplane': 16,
                },
        ],
        'out': [
                {
                    'inplane': 32,
                    'outplane': 8,
                },
                {
                    'inplane': 8,
                    'outplane': 8,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 8,
                    'outplane': 1,
                },
        ],
    },

    'Level_1': {
        'concat': True,
        'in': [
                {
                    'inplane': 16,
                    'outplane': 32,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
        ],
        'out': [
                {
                    'inplane': 64,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 16,
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
        'through': [
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
}#}}}
