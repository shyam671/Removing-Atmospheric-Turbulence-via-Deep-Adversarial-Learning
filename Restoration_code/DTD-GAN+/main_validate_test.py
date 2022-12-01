import argparse
import os
import time
import torch
import torchvision
from datasets import ImageFolder, PairedImageFolder
import torchvision.transforms as transforms
import pairedtransforms
import networks_CA_SUB as networks
from PIL import Image
import synthdata
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def mse(x, y):
    return np.linalg.norm(x - y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/mnt/Data1', help='path to images')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
    parser.add_argument('--outroot', default='./results', help='path to save the results')
    parser.add_argument('--exp-name', default='test', help='name of expirement')
    parser.add_argument('--load', default='', help='name of pth to load weights from')
    parser.add_argument('--freeze-cc-net', dest='freeze_cc_net', action='store_true', help='dont train the color corrector net')
    parser.add_argument('--freeze-warp-net', dest='freeze_warp_net', action='store_true', help='dont train the warp net')
    parser.add_argument('--test', dest='test', action='store_true', help='only test the network')
    parser.add_argument('--synth-data', dest='synth_data', action='store_true', help='use synthetic data instead of tank data for training')
    parser.add_argument('--epochs', default=3, type=int, help='number of epochs to train for')
    parser.add_argument('--no-warp-net', dest='warp_net', action='store_false', help='do not include warp net in the model')
    parser.add_argument('--warp-net-downsample', default=3, type=int, help='number of downsampling layers in warp net')
    parser.add_argument('--no-color-net', dest='color_net', action='store_false', help='do not include color net in the model')
    parser.add_argument('--color-net-downsample', default=3, type=int, help='number of downsampling layers in color net')
    parser.add_argument('--no-color-net-skip', dest='color_net_skip', action='store_false', help='dont use u-net skip connections in the color net')
    parser.add_argument('--dim', default=32, type=int, help='initial feature dimension (doubled at each downsampling layer)')
    parser.add_argument('--n-res', default=8, type=int, help='number of residual blocks')
    parser.add_argument('--norm', default='gn', type=str, help='type of normalization layer')
    parser.add_argument('--denormalize', dest='denormalize', action='store_true', help='denormalize output image by input image mean/var')
    parser.add_argument('--weight-X-L1', default=1., type=float, help='weight of L1 reconstruction loss after color corrector net')
    parser.add_argument('--weight-Y-L1', default=1., type=float, help='weight of L1 reconstruction loss after warp net')
    parser.add_argument('--weight-Y-VGG', default=1., type=float, help='weight of perceptual loss after warp net')
    parser.add_argument('--weight-Z-L1', default=1., type=float, help='weight of L1 reconstruction loss after color net')
    parser.add_argument('--weight-Z-VGG', default=.5, type=float, help='weight of perceptual loss after color net')
    parser.add_argument('--weight-Z-Adv', default=0.2, type=float, help='weight of adversarial loss after color net')
    args = parser.parse_args()

    # set random seed for consistent fixed batch
    torch.manual_seed(8)

    # set weights of losses of intermediate outputs to zero if not necessary
    if not args.warp_net:
        args.weight_Y_L1=0
        args.weight_Y_VGG=0
    if not args.color_net:
        args.weight_Z_L1=0
        args.weight_Z_VGG=0
        args.weight_Z_Adv=0


    # datasets
    # train_dir_1 = os.path.join(args.dataroot, 'Imagenet_turb', 'train');
    # train_dir_2 = os.path.join(args.dataroot, 'Imagenet', 'train');
#    val_dir_1 = os.path.join(args.dataroot,'Imagenet_turb', 'val');
#    val_dir_2 = os.path.join(args.dataroot,'Imagenet', 'val');
    test_dir = os.path.join(args.dataroot,'test')

    # if args.synth_data:
    #     train_data = ImageFolder(train_dir_2, transform=
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #             synthdata.SynthData(224, n=args.batch_size),
    #         ]))
    # else:
    #     train_data = PairedImageFolder(train_dir_1, train_dir_2, transform=
    #         transforms.Compose([
    #             pairedtransforms.RandomResizedCrop(224),
    #             pairedtransforms.RandomHorizontalFlip(),
    #             pairedtransforms.ToTensor(),
    #             pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ]))
    #val_data = PairedImageFolder(val_dir_1, val_dir_2, transform=
    #    transforms.Compose([
    #        pairedtransforms.Resize(256),
            #pairedtransforms.CenterCrop(224),
    #        pairedtransforms.ToTensor(),
    #        pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #    ]))
    test_data = ImageFolder(test_dir, transform=
        transforms.Compose([
#            transforms.Resize(256),
#            #transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

#    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False, shuffle=True)
#    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers= 1, pin_memory= True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)

    # fixed test batch for visualization during training
    #fixed_batch = iter(val_loader).next()[0]

    # model
    model=networks.Model(args)
    print(model)
    model.cuda()
    model.load_state_dict(torch.load('/home/shyam.nandan/WGAN/LearningToSeeThroughTurbulrntWater/TurbulentWater-Enssemble/results/best.pth'))
    # load weights from checkpoint
    #if args.test and not args.load:
    #    args.load = args.exp_name
    #if args.load:
    #    model.load_state_dict(torch.load(os.path.join(args.outroot, '%s_net.pth'%args.load)), strict=args.test)

    # create outroot if necessary
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)

    # if args.test only run test script
    if args.test:
    	train(test_loader, model, 1, args)
    	return

    #train(val_loader, model, 1, args)
    #return
#def train(loader, model, fixed_batch, epoch, args):
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

def train(loader, model, epoch, args):
    model.eval()

    total_psnr_r = 0
    total_psnr_t = 0

    total_mse_r = 0
    total_mse_t = 0

    total_ssim_r = 0

    for i, data in enumerate(loader):
        input = data[0].cuda()

        
        ############################## Ensemble method ###############################################
	## Original       
        x, warp, y, z_0 = model(input)

	## 90D + flip 
        input90f = input.transpose(2,3)
	x, warp, y, z_90f = model(input90f)        
        z_90f = z_90f.transpose(2,3)
	
	## 90D 
        input90 = input.transpose(2,3)
	input90 = flip(input90,2)
	x, warp, y, z_90 = model(input90)
	z_90 = flip(z_90, 2)        
        z_90 = z_90.transpose(2,3)
	## 270D 
        input270 = flip(input,2)
	input270 = input270.transpose(2,3)
	x, warp, y, z_270 = model(input270)
	z_270 = z_270.transpose(2,3)
	z_270 = flip(z_270,2)

	## 270D +f
        input270f = flip(input,2)
	input270f = input270f.transpose(2,3)
	input270f = flip(input270f,2)
	x, warp, y, z_270f = model(input270f)
	z_270f = flip(z_270f,2)
	z_270f = z_270f.transpose(2,3)
	z_270f = flip(z_270f,2)
        
	## 180 +f 
        input180f = flip(input,3)
	x, warp, y, z_180f = model(input180f)
	z_180f = flip(z_180f,3)

	## 180 
        input180 = flip(input,3)
	input180 = flip(input180,2)
	x, warp, y, z_180 = model(input180)
	z_180 = flip(z_180,2)
	z_180 = flip(z_180,3)

        ## flip 
	#input_f = flip(input,2)
	#x, warp, y, z_f = model(input_f)
	#z_f = flip(z_f,2)

	## Average

        z = (z_0 + z_90f + z_90 + z_270 + z_270f + z_180f + z_180 )/7
        #z = z_0
	##############################################################################################
        ##PSNR and MSE for restored images
	#t = target.data.cpu().numpy().reshape((256, 256, 3))
	p = z.data.cpu().numpy()
	print(os.path.join(args.outroot,str(i),'.jpg'))
	torchvision.utils.save_image(z.cpu(), os.path.join(args.outroot,str(i)+'.jpg'), nrow=1, normalize=True, range=(-1,1))
	#total_psnr_r += psnr(t, p)
        #total_mse_r  += mse(t, p)
	#total_ssim_r += ssim(t, p, data_range = t.max() - t.min(), multichannel=True)
	#torchvision.utils.save_image(z.cpu(), os.path.join(args.outroot,'TR_RDGANEn',str(i) + '_TR_RDGANEn.jpg' ), nrow=1, normalize=True, range=(-1,1), pad_value=1)
	print(i)
    #print(total_psnr_r/i,total_mse_r/i, total_ssim_r/i)

if __name__ == '__main__':
    main()
