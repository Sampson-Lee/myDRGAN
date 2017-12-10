from torch.optim import lr_scheduler


def get_scheduler(optimizer, args):
    if args.lr_policy == 'constant':
        scheduler = optimizer
    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs//4, gamma=0.1)
    elif args.lr_policy == 'multistep':
        milestones = [int(args.epochs*(1-0.5**power)) for power in range(4)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler