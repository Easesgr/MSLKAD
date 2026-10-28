"""Factory for the author's MSLKAD architecture."""

from .MSLKAD import MSLKAD


def get_models(args):
    if set(args['model']) != {'MSLKAD'}:
        raise ValueError('This release supports exactly one model: MSLKAD.')
    config = args['model']['MSLKAD']
    if config.get('model_name', 'MSLKAD') != 'MSLKAD':
        raise ValueError('model_name must be MSLKAD.')
    options = config['model_args']
    for name in ('n_resblocks', 'n_resgroups', 'n_feats'):
        if type(options[name]) is not int or options[name] <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    if options['n_feats'] % 3:
        raise ValueError('n_feats must be divisible by 3 for the three MLA branches.')
    return {'MSLKAD': MSLKAD(args)}
