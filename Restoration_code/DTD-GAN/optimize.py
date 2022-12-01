import torch
import torch.nn.functional as F
from torch import nn
import itertools
from collections import OrderedDict
import Interp
from torchvision.models import vgg16
import numpy as np
import csv
from common import *

class Optimize_Model(nn.Module):
    def __init__(self):
        super(Optimize_Model, self).__init__()

        # weights of losses
        #self.weights = {'X_L1':args.weight_X_L1, 'Y_L1':args.weight_Y_L1, 'Z_L1':args.weight_Z_L1, 'Y_VGG':args.weight_Y_VGG, 'Z_VGG':args.weight_Z_VGG, 'Z_Adv':args.weight_Z_Adv}
        self.weights = {'X_L1':1, 'Y_L1':1, 'Z_L1':1, 'Y_VGG':1, 'Z_VGG':1, 'Z_Adv':1}

	    self.optimizer_G = torch.optim.Adam(params_G, lr=learning_rate, betas=(.5, 0.999))

        # for reconstruction loss
        self.recon_criterion = nn.L1Loss()

        # discriminator for adversarial loss
        if self.weights['Z_Adv']>0:
            #self.discriminator = Discriminator(dim=args.dim)
            self.discriminator = Discriminator(dim=224)
            
	    self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(.5, 0.999))

    def compute_loss_G(self, x, y, z, target):
        losses = OrderedDict()
        vggf = vgg_features()
        # Reconstruction loss
        if self.weights['X_L1']>0:
            losses['X_L1'] = self.recon_criterion(x, target)
        if self.weights['Y_L1']>0:
            losses['Y_L1'] = self.recon_criterion(y, target)
        if self.weights['Z_L1']>0:
            losses['Z_L1'] = self.recon_criterion(z, target)

        # Perceptual loss
        if self.weights['Y_VGG']>0:
            losses['Y_VGG'] =  self.recon_criterion(vggf.forward(y), vggf.forward(target))
        if self.weights['Z_VGG']>0:
            losses['Z_VGG'] =  self.recon_criterion(vggf.forward(z), vggf.forward(target))

        # Adversarial loss
        if self.weights['Z_Adv']>0:
            losses['Z_Adv'] = self.discriminator.calc_gen_loss(z)

        return losses


    def optimize_parameters(self, x, y, z, target,epoch):
        #x, warp, y, z = self.forward(input)

        # update discriminator
        if epoch == 2:
	   for g in self.optimizer_G.param_groups:
   	       g['lr']  = 2e-5
           for g in self.optimizer_D.param_groups:
   	       g['lr']  = 2e-5

        if self.weights['Z_Adv']>0:
            self.optimizer_D.zero_grad()
            loss_d = self.discriminator.calc_dis_loss(z, target)
            loss_d.backward()
            self.optimizer_D.step()

        # update generators
        self.optimizer_G.zero_grad()
        losses  = self.compute_loss_G(x, y, z, target)
        loss = sum([losses[key]*self.weights[key] for key in losses.keys()])
        loss.backward()
        self.optimizer_G.step()

        if self.weights['Z_Adv']>0:
            losses['Dis'] = loss_d
        return losses


    def print_losses(self, losses):
        ##Save_Loss
        with open('./results/loss.csv', 'a') as f:
        	writer = csv.writer(f)
    		writer.writerow([losses['Y_L1'].data.cpu().numpy(),losses['Y_VGG'].data.cpu().numpy()])
        return ' '.join(['Loss %s: %.4f'%(key, val.item()) for key,val in losses.items()])
    
    def calc_dis_loss(self, input_fake, input_real):
        input_fake = input_fake.detach()
        input_real = input_real.detach()
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(out0, requires_grad=False).cuda()
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                              F.binary_cross_entropy(F.sigmoid(out1), all1))
        elif self.gan_type == 'wgan':
            loss = out0.mean()-out1.mean()
            # grad penalty
            BatchSize = input_fake.size(0)
            alpha = torch.rand(BatchSize,1,1,1, requires_grad=False).cuda()
            interpolates = (alpha * input_real) + (( 1 - alpha ) * input_fake)
            interpolates.requires_grad=True
            outi = self.forward(interpolates)
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            gradients = torch.autograd.grad(outi, interpolates, grad_outputs=all1, create_graph=True)[0]
            #gradient_penalty = ((gradients.view(BatchSize,-1).norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty = ((gradients.view(BatchSize,-1).norm(1, dim=1) - 1).clamp(0) ** 2).mean()
            loss += 10*gradient_penalty
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        out0 = self.forward(input_fake)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 1)**2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(out0.data, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        elif self.gan_type == 'wgan':
            loss = -out0.mean()
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

