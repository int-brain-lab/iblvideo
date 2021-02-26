SIDE_FEATURES = {
    'roi_detect':
        {'label': 'roi_detect',
         'features': None,
         'weights': 'roi_detect-2019-12-11',
         'crop': None,
         'postcrop_downsampling': 1},
    'nose_tip':
        {'label': 'nose_tip',
         'features': ['nose_tip'],
         'weights': 'nose_tip-2019-12-23',
         'crop': lambda x, y: [100, 100, x - 50, y - 50],
         'postcrop_downsampling': 1},
    'eye':
        {'label': 'eye',
         'features': ['pupil_top_r'],
         'weights': 'eye-mic-2020-01-24',
         'crop': lambda x, y: [100, 100, x - 50, y - 50],
         'postcrop_downsampling': 1},
    'paws':
        {'label': 'paws',
         'features': ['nose_tip'],
         'weights': 'paw2-mic-2020-03-23',
         'crop': None,  # lambda x, y: [900, 800, x, y - 100],
         'postcrop_downsampling': 10},
    'tongue':
        {'label': 'tongue',
         'features': ['tube_top', 'tube_bottom'],
         'weights': 'tongue-mic-2019-04-26',
         'crop': lambda x, y: [160, 160, x - 60, y - 100],
         'postcrop_downsampling': 1},
}

BODY_FEATURES = {
    'roi_detect':
        {'label': 'roi_detect',
         'features': None,
         'weights': 'tail-mic-2019-12-16',
         'crop': None,
         'postcrop_downsampling': 1},
    'tail_start':
        {'label': 'tail_start',
         'features': ['tail_start'],
         'weights': 'tail-mic-2019-12-16',
         'crop': lambda x, y: [220, 220, x - 110, y - 110],
         'postcrop_downsampling': 1}
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
