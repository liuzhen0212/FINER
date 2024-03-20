import os
import sys
import time
import numpy as np
import torch
from torch import nn
import matplotlib
from models import Finer, Siren, Gauss, PEMLP, Wire
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import trange
from utils import setup_seed, mse_fn, psnr_fn, read_image, write_image
import random
import imageio.v2 as imageio
import configargparse


setup_seed(3407)
device = torch.device('cuda:0')


class Logger:
    filename = 'experiment_scripts_finer/logs/calc_time/logs_time/logtime.txt'
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')
    

def get_train_data(img_path, zero_mean=True):
    img = np.array(imageio.imread(img_path), dtype=np.float32) / 255.
    # normalize
    if zero_mean:
        img = (img - 0.5) / 0.5 # [-1, 1]
    H, W, C = img.shape
    gt = torch.tensor(img).view(-1, C)
    coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)], indexing='ij'), dim=-1).view(-1, 2)
    return coords, gt, [H, W, C]


def get_model(opts):
    if opts.model_type == 'finer':
        model = Finer(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, first_bias_scale=opts.first_bias_scale, scale_req_grad=opts.scale_req_grad)
    elif opts.model_type == 'siren':
        model = Siren(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega)
    elif opts.model_type == 'wire':
        model = Wire(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                     first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, scale=opts.scale)
    elif opts.model_type == 'gauss':
        model = Gauss(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      scale=opts.scale)
    elif opts.model_type == 'pemlp':
        model = PEMLP(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      N_freqs=opts.N_freqs)
    return model


def get_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs', help='logdir')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    
    # dataset
    parser.add_argument('--dataset_dir', type=str, default= 'data/div2k/test_data', help='dataset')
    parser.add_argument('--img_id', type=int, default=0, help='id of image')
    parser.add_argument('--not_zero_mean', action='store_true') 
    
    # training options
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--steps_til_summary', type=int, default=10, help='steps_til_summary')
    
    # network 
    parser.add_argument('--model_type', type=str, default='siren', required=['siren', 'finer', 'wire', 'gauss', 'pemlp'])
    parser.add_argument('--hidden_layers', type=int, default=3, help='hidden_layers') 
    parser.add_argument('--hidden_features', type=int, default=256, help='hidden_features')
    
    #
    parser.add_argument('--first_omega', type=float, default=30, help='(siren, wire, finer)')    
    parser.add_argument('--hidden_omega', type=float, default=30, help='(siren, wire, finer)')    
    parser.add_argument('--scale', type=float, default=30, help='simga (wire, guass)')    
    parser.add_argument('--N_freqs', type=int, default=10, help='(PEMLP)')    

    # finer
    parser.add_argument('--first_bias_scale', type=float, default=None, help='bias_scale of the first layer')    
    parser.add_argument('--scale_req_grad', action='store_true') 
    return parser.parse_args()


def train(model, coords, gt, size, zero_mean=True, loss_fn=mse_fn, lr=1e-3, num_epochs=2000, steps_til_summary = 10):

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))
        
    train_iter = []
    train_psnr = []
    train_lr = []
    
    total_time = 0
    for epoch in trange(1, num_epochs + 1):      
        time_start = time.time()

        pred = model(coords)
        loss = loss_fn(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        total_time += time.time() - time_start
   

        if not epoch % steps_til_summary:
            with torch.no_grad():
                train_iter.append(epoch)
                if zero_mean:
                    train_psnr.append((psnr_fn(model(coords)/2+0.5, gt/2+0.5)).item())
                else:
                    train_psnr.append((psnr_fn(model(coords), gt)).item())
        
    with torch.no_grad():
        if zero_mean:
            pred_img = model(coords).reshape(size)/2+0.5
        else:
            pred_img = model(coords).reshape(size)
        
    ret_dict = {
        'train_iter': train_iter,
        'train_psnr': train_psnr,
        'pred_img': pred_img,
        'model_state': model.state_dict(),
        'total_time': total_time
    }
    return ret_dict



if __name__ == '__main__':
    opts = get_opts()
    
    print('--- Run Configuration ---')
    for k, v in vars(opts).items():
        print(k, '=', v)
    print('--- Run Configuration ---')
    
    ## logdir 
    logdir = os.path.join(opts.logdir, opts.exp_name)
    os.makedirs(logdir, exist_ok=True)
    
    # image path
    coords, gt, size = get_train_data(os.path.join(opts.dataset_dir, f'%02d.png'%(opts.img_id)), not opts.not_zero_mean)
    coords = coords.to(device)
    gt = gt.to(device)
    print(coords.shape, gt.shape, gt.max(), gt.min())
    
    # model
    model = get_model(opts).to(device)
    print(model)
    
    # train
    ret_dict = train(model, coords, gt, size, not opts.not_zero_mean, mse_fn, lr=opts.lr, num_epochs=opts.num_epochs, steps_til_summary=opts.steps_til_summary)
    
    # save 
    torch.save(ret_dict, os.path.join(logdir, 'outputs_%02d.pt'%((opts.img_id))))
    
    print('Train PSNR: %.4f'%(ret_dict['train_psnr'][-1]))
    
    # [option]
    # Logger.write(f'{opts.exp_name}      %.4f'%(ret_dict['total_time']))