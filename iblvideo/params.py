SIDE_FEATURES = {
    'roi_detect': {
        'label': 'roi_detect',
        'features': None,
        'weights': 'roi_detect-*',
        'crop': lambda x, y, s: None,
        'sequence_length': 16,  # batch size for inference; 16 works for 8GB GPU
        'eks_params': {},
     },
    'nose_tip': {
        'label': 'nose_tip',
        'features': ['nose_tip'],  # window anchor from roi network
        'weights': 'nose_tip-*',
        'crop': lambda x, y, s: [100 / s, 100 / s, x - 50 / s, y - 50 / s],
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU (smaller network)
        'eks_params': {},
    },
    'eye': {
        'label': 'eye',
        'features': ['pupil_top_r'],  # window anchor from roi network
        'weights': 'eye-mic-*',
        'crop': lambda x, y, s: [100 / s, 100 / s, x - 50 / s, y - 50 / s],
        'sequence_length': 64,  # batch size for inference; 64 works for 8GB GPU
        'eks_params': {  # smoothing params; closer to 1 = more smoothing
            'diameter': 0.9999,
            'com': 0.999,
        },
    },
    'paws': {
        'label': 'paws',
        'features': ['nose_tip'],  # dummy entry to force run with other specialized networks
        'weights': 'paw2-mic-*',
        'crop': lambda x, y, s: None,
        'sequence_length': 24,  # batch size for inference; 24 works for 8GB GPU
        'eks_params': {  # smooth params; ranges from .01-20; smaller values = more smoothing
            's': 10,
        },
    },
    'tongue': {
        'label': 'tongue',
        'features': ['tube_top', 'tube_bottom'],  # window anchor from roi network
        'weights': 'tongue-mic-*',
        'crop': lambda x, y, s: [160 / s, 160 / s, x - 60 / s, y - 100 / s],
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU (smaller network)
        'eks_params': {},
    },
}

BODY_FEATURES = {
    'roi_detect': {
        'label': 'roi_detect',
        'features': None,
        'weights': 'tail-mic-*',
        'crop': lambda x, y, s: None,
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
        'eks_params': {},
    },
    'tail_start': {
        'label': 'tail_start',
        'features': ['tail_start'],
        'weights': 'tail-mic-*',
        'crop': lambda x, y, s: None,  # [220, 220, x - 110, y - 110],
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
        'eks_params': {},
    }
}

LEFT_VIDEO = {
    'original_size': [1024, 1280],  # height, width
    'flip': False,
    'features': SIDE_FEATURES,
    'scale': 1,  # spatial sampling, if > 1 means smaller
}

RIGHT_VIDEO = {
    'original_size': [1024 // 2, 1280 // 2],  # height, width
    'flip': True,
    'features': SIDE_FEATURES,
    'scale': 2,  # spatial sampling, if > 1 means smaller
}

BODY_VIDEO = {
    'original_size': [1024 // 2, 1280 // 2],  # height, width
    'flip': False,
    'features': BODY_FEATURES,
    'scale': 1,  # spatial sampling, if > 1 means smaller
}
