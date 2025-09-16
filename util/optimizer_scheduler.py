import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR
def initialize_optimizer(args,model,model_name):

    optim_g_params = args.model[model_name].optim_args

    # 根据优化器类型选择不同的优化器
    if optim_g_params['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            betas=tuple(optim_g_params['betas'])
        )
    elif optim_g_params['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            momentum=optim_g_params['momentum']
        )
    elif optim_g_params['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optim_g_params['lr'],
            weight_decay=optim_g_params['weight_decay'],
            betas=tuple(optim_g_params['betas'])
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_g_params['type']}")

    return optimizer

def initialize_scheduler(args,optimizer,model_name):

    scheduler_params = args.model[model_name].scheduler

    # 根据调度器类型选择不同的调度器
    if scheduler_params['type'] == 'MultiStepLR':
        scheduler = MultiStepLR(
            optimizer,
            milestones=scheduler_params['milestones'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_params['type'] == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_params['type'] == 'ExponentialLR':
        scheduler = ExponentialLR(
            optimizer,
            gamma=scheduler_params['gamma']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_params['type']}")

    return scheduler


