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

import csv

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
            #pairedtransforms.Resize(256),
            #pairedtransforms.CenterCrop(256),
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
#    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=args.workers, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False, shuffle=False)

    # fixed test batch for visualization during training
    #fixed_batch = iter(val_loader).next()[0]

    # model
    model=networks.Model(args)
    print(model)
    model.cuda()
    model.load_state_dict(torch.load('/home/shyam.nandan/WGAN/LearningToSeeThroughTurbulrntWater/TurbulentWater-master_v2/results/best.pth'))
    # load weights from checkpoint
    if args.test and not args.load:
        args.load = args.exp_name
    if args.load:
        model.load_state_dict(torch.load(os.path.join(args.outroot, '%s_net.pth'%args.load)), strict=args.test)

    # create outroot if necessary
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)

    # if args.test only run test script
    if args.test:
        test(test_loader, model, args)
        return

    train(val_loader, model, 1, args)
    return
#def train(loader, model, fixed_batch, epoch, args):
def train(loader, model, epoch, args):
    model.eval()

    total_psnr_r = 0
    total_psnr_t = 0

    total_mse_r = 0
    total_mse_t = 0

    total_ssim_r = 0

    for i, ((input, target), ipath) in enumerate(loader):
        input = input.cuda()
        x, warp, y, z = model(input)

	t = target.data.cpu().numpy().reshape((224, 224, 3))
	p = z.data.cpu().numpy().reshape((224, 224, 3))
	
	#total_psnr_r += psnr(t, p)
        #total_mse_r  += mse(t, p)
	#total_ssim_r += ssim(t, p, data_range = t.max() - t.min(), multichannel=True)
	torchvision.utils.save_image(z.cpu(), os.path.join(args.outroot,'10',ipath[0][5:]), nrow=1, normalize=True, range=(-1,1)) 
	print(i)
    #print(total_psnr_r/i,total_mse_r/i, total_ssim_r/i)

def visualize(input, target, model, name):
    model.eval()
    with torch.no_grad():
        x, warp, y, z = model(input)
        warp = (warp+5)/10
        warp = torch.cat((warp, torch.ones_like(y[:,:1,:,:])), dim=1)
        visuals = torch.cat([input.cpu(), x.cpu(), warp.cpu(), y.cpu(), z.cpu(), target.cpu()], dim=2)
        torchvision.utils.save_image(visuals, name, nrow=16, normalize=True, range=(-1,1), pad_value=1)
        ##PSNR
        max_val = np.amax(input.data.cpu().numpy())
        mse = np.mean(np.square(input.data.cpu().numpy() - z.data.cpu().numpy()))
        psnr = np.clip(np.multiply(np.log10(max_val *max_val / mse[mse > 0.]), 10.), 0., 99.99)[0]
        with open('./results/psnr.csv', 'a') as f:
        	writer = csv.writer(f)
    		writer.writerow([psnr])


def test(loader, model, args):
    model.eval()
    with torch.no_grad():
        end_time = time.time()
        for i, data in enumerate(loader):
            input = data[0].cuda()
            data_time = time.time() - end_time
            x, warp, y, z = model(input, cc=False)

            # save the output for each image by name
            for out, name in zip(z,data[-1]):
                if not os.path.exists(os.path.join(args.outroot, '%s_test'%args.exp_name, os.path.dirname(name))):
                    os.makedirs(os.path.join(args.outroot, '%s_test'%args.exp_name, os.path.dirname(name)))

                im = Image.fromarray( (out*.5+.5).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() )
                im.save(os.path.join(args.outroot, '%s_test'%args.exp_name, name))

            # save warpnet output Image
            for out, name in zip(y,data[-1]):
                if not os.path.exists(os.path.join(args.outroot, '%s_warpout'%args.exp_name, os.path.dirname(name))):
                    os.makedirs(os.path.join(args.outroot, '%s_warpout'%args.exp_name, os.path.dirname(name)))

                imy = Image.fromarray( (out*.5+.5).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() )
                imy = imy.convert('RGB')
                imy.save(os.path.join(args.outroot, '%s_warpout'%args.exp_name, name))

            # save warp field
            for out, name in zip(warp,data[-1]):
                if not os.path.exists(os.path.join(args.outroot, '%s_warp'%args.exp_name, os.path.dirname(name))):
                    os.makedirs(os.path.join(args.outroot, '%s_warp'%args.exp_name, os.path.dirname(name)))

                imw = Image.fromarray( (out*.5+.5).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() )
                imw = imw.convert('RGB')
                imw.save(os.path.join(args.outroot, '%s_warp'%args.exp_name, name))



            batch_time = time.time() - end_time
            print '%s Test: %04d/%04d time: %.3f %.3f '%(args.exp_name, i, len(loader), data_time, batch_time)
            end_time = time.time()

            #if i==10:
            #    break



if __name__ == '__main__':
    main()
