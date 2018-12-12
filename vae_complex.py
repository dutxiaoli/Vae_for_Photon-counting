import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import time
from torchvision.utils import save_image
import argparse
import os


configurations = {
        # same configuration as original work
        # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
                max_iteration=400000,
                lr=1.0e-10,
                momentum=0.99,
                weight_decay=0.0005,
            
        )

}


parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--train_dataroot', type=str, default='/media/iiau/UbuntuData/wtt/train_data', help='path to train data')
parser.add_argument('--test_dataroot', type=str, default='/media/iiau/UbuntuData/wtt/test_data', help='path to test data')
parser.add_argument('--snapshot_root', type=str, default='./snapshot', help='path to snapshot')
parser.add_argument('--reconstruct_root', type=str, default='./reconstruct_map', help='path to reconstruct map')
parser.add_argument('--generate_root', type=str, default='./generate_map', help='path to generate map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
parser.add_argument('--resume', help='Checkpoint path')
parser.add_argument('--snapshot_opt_root', type=str, default='./snapshot_opt', help='path to snapshot optimize')
args = parser.parse_args()
cfg = configurations[args.config]

cuda = torch.cuda.is_available


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3,padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.bn(out1)
        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out2 = self.bn(out2)
        out3 = self.conv3(out2)
        out3 = self.relu(out3)

        return out3


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(D_in, 64, 3,padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1,3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.bn(out1)
        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out2 = self.bn(out2)
        out3 = self.conv3(out2)
        out3 = self.relu(out3)
        return out3


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Conv2d(32, 8, 3,padding=1)
        self._enc_log_sigma = torch.nn.Conv2d(32, 8, 3,padding=1)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        latent_size = z.size()
        return self.decoder(z), latent_size


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

def save_reconstruct_img(dst_data, recon_data, epoch, batch_size):
    dst_data = dst_data.cpu()
    recon_data = recon_data.cpu()
    n = min(dst_data.size(0), 8)
    comparison = torch.cat( [(dst_data[:n].view(n,1,28, 28)),
                            (recon_data[:n].view(n,1,28, 28))])
    save_image(comparison.data,
               'results/vae_conv/reconstruction_' + str(epoch) + '.png', nrow=n)

def sample_and_construct(epoch,decoder,latent_size):
    sample = Variable(torch.randn(latent_size).cuda())
    sample = decoder(sample)
    sample =  sample.cpu()
    save_image(sample.data,
               'results/sample_vae_conv/sample_' + str(epoch) + '.png')

if __name__ == '__main__':


    input_dim = 28 * 28
    batch_size = 32
    start = time.clock()

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(3, 100, 100)
    decoder = Decoder(8, 100, 3)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    # add cuda
    vae.cuda()
    criterion.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(100):
        
        if(epoch!=0):
            print("\nloading parameters")
            vae.load_state_dict(torch.load(args.snapshot_root + '/feature-current.pth'))
            optimizer.load_state_dict(torch.load(args.snapshot_opt_root + '/opti-current.pth'))
            #
        title = 'Training Epoch {}'.format(epoch)
        
        
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs).cuda(), Variable(classes).cuda()
            
            optimizer.zero_grad()
            dec, latent_size = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data[0]

        elapsed = (time.clock() - start)
        print( epoch, l,'time:',elapsed)
        save_reconstruct_img(inputs, dec, epoch,batch_size)
        sample_and_construct(epoch, decoder,latent_size)
    
        filename = ('%s/feature_%d.pth' % (args.snapshot_root, epoch))
        filename_opti = ('%s/opti-best_%d.pth' % (args.snapshot_opt_root, epoch))
        filename_current = ('%s/feature-current.pth' % (args.snapshot_root))
        filename_opti_current = ('%s/opti-current.pth' % (args.snapshot_opt_root))
        torch.save(vae.state_dict(), filename)
        torch.save(optimizer.state_dict(), filename_opti)
        torch.save(vae.state_dict(), filename_current)
        torch.save(optimizer.state_dict(), filename_opti_current)
