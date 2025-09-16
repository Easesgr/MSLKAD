import torch
from util.sys_util import AttrDict,no_of_params
from pprint import pformat

import logging

logger = logging.getLogger(__name__)


def _get_model(args, model_name, optim_args):

    #JI
    if  model_name == 'MFADR':
        from model.MSLKAD import MFADR
        m = MFADR
    else:
        raise NotImplementedError()
    if torch.cuda.is_available():
        selected_gpus = args.train.gpus
        device = torch.device(f"cuda:{selected_gpus[0]}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA and MPS not available. Using CPU.")


    model = m(args).to(device)
    optim_args = { **optim_args}
    optim_args = AttrDict(optim_args)
    model.optim_args = optim_args
    model.model_name = model_name
    logger.info(
        f'{model_name} created. No. of parameters: {no_of_params(model)}. Optimization Args: {pformat(optim_args)}.')
    logger.debug(model)

    return model


def get_models(args):
    '''
        Models are grouped into two components: downsampling (generator), and discriminator
        All model related parameters are stored in the model itself: namely, the LR, LR decay, and steps

        Returns a dict of models with the associated keys
    '''
    model_components = ['MFADR']
    models = {}
    for mc in model_components:
        if hasattr(args.model, mc):
            md = getattr(args.model, mc)
            models[mc] = _get_model(args, md.model_name, md.optim_args)

    return models