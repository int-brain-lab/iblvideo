SIDE_FEATURES = {
    'roi_detect': {
        'label': 'roi_detect',
        'features': None,
        'weights': 'roi_detect-*',
        'crop': lambda x, y: None,
        'postcrop_downsampling': 1,
        'resize_dims': (512, 512),  # frame size for training
        'sequence_length': 16,  # batch size for inference; 16 works for 8GB GPU
     },
    'nose_tip': {
        'label': 'nose_tip',
        'features': ['nose_tip'],  # window anchor from roi network
        'weights': 'nose_tip-*',
        'crop': lambda x, y: [100, 100, x - 50, y - 50],
        'postcrop_downsampling': 1,
        'resize_dims': (128, 128),  # frame size for training
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
    },
    'eye': {
        'label': 'eye',
        'features': ['pupil_top_r'],  # window anchor from roi network
        'weights': 'eye-mic-*',
        'crop': lambda x, y: [100, 100, x - 50, y - 50],
        'postcrop_downsampling': 1,
        'resize_dims': (128, 128),  # frame size for training
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
    },
    'paws': {
        'label': 'paws',
        'features': ['nose_tip'],  # dummy entry to force run with other specialized networks
        'weights': 'paw2-mic-*',
        'crop': lambda x, y: None,
        'postcrop_downsampling': 10,
        'resize_dims': (128, 128),  # frame size for training
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
    },
    'tongue': {
        'label': 'tongue',
        'features': ['tube_top', 'tube_bottom'],  # window anchor from roi network
        'weights': 'tongue-mic-*',
        'crop': lambda x, y: [160, 160, x - 60, y - 100],
        'postcrop_downsampling': 1,
        'resize_dims': (128, 128),  # frame size for training
        'sequence_length': 96,  # batch size for inference; 96 works for 8GB GPU
    },
}

BODY_FEATURES = {
    'roi_detect': {
        'label': 'roi_detect',
        'features': None,
        'weights': 'tail-mic-*',
        'crop': lambda x, y: None,
        'postcrop_downsampling': 1,
    },
    'tail_start': {
        'label': 'tail_start',
        'features': ['tail_start'],
        'weights': 'tail-mic-*',
        'crop': lambda x, y: [220, 220, x - 110, y - 110],
        'postcrop_downsampling': 1,
    }
}

LEFT_VIDEO = {
    'original_size': [1280, 1024],
    'flip': False,
    'features': SIDE_FEATURES,
    'sampling': 1,  # sampling factor applied before cropping, if > 1 means upsampling
}

RIGHT_VIDEO = {
    'original_size': [1280 // 2, 1024 // 2],
    'flip': True,
    'features': SIDE_FEATURES,
    'sampling': 2,  # sampling factor applied before cropping, if > 1 means upsampling
}

BODY_VIDEO = {
    'original_size': [1280 // 2, 1024 // 2],
    'flip': False,
    'features': BODY_FEATURES,
    'sampling': 1,  # sampling factor applied before cropping, if > 1 means upsampling
}
