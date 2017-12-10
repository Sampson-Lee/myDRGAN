from train_single_DRGAN import train_single_DRGAN
from train_multiple_DRGAN import train_multiple_DRGAN
from generate_image import generate_image
from representation_learning import representation_learning

def initrun(dataloader, netD, netG, args):
    # model train or inference
    if args.mode in ['trainsingle', 'trainrandom']:
        mes = train_single_DRGAN(dataloader, netD, netG, args)
    if args.mode=='trainmulti':
        mes = train_multiple_DRGAN(dataloader, netD, netG, args)
    if args.mode in ['gensingle', 'genmulti']:
        mes = generate_image(dataloader, netG, args)
    if args.mode in ['idensingle', 'idenmulti']:
        mes = representation_learning(dataloader, netG, args)

    return mes