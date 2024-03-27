import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
import cv2
from loss import *
import time
import random
from glob import glob
import pdb
import scipy.io as scio
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='./input/SUIM', help='path to images folder')
parser.add_argument('--data_suffix', default='jpg', help='suffix of the testing image')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                    default='./pretrain_ckpt/model_best.tar')
parser.add_argument('--output', metavar='DIR', default='./result/SUIM/', help='path to output folder')

parser.add_argument('--downsize', default=16, type=float, help='superpixel grid cell, must be same as training setting')
parser.add_argument('--scale', default=1, type=float,
                    help='the multiple of image scaling, which can adjust the number of superpixels')
parser.add_argument('-nw', '--num_threads', default=4, type=int, help='num_threads')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

args = parser.parse_args()

random.seed(100)


@torch.no_grad()
def test(args, model, img_paths, save_path, idx):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]
    # may get 4 channel (alpha channel) for some format
    print(imgId)
    color = True
    img_ = cv2.imread(load_path)
    if len(img_.shape) == 2:
        img_ = np.tile(np.expand_dims(img_, 2), (1, 1, 3))
        mask = np.where(img_ > 0, np.ones_like(img_), np.zeros_like(img_))
        color = False
    img_ = img_[:, :, :3]
    ori_img = input_transform(img_)
    H, W, _ = img_.shape
    H_, W_ = int(np.ceil(H * args.scale / 16.) * 16), int(np.ceil(W * args.scale / 16.) * 16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel = int(n_spixl_h * n_spixl_w)

    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)

    # compute output
    tic = time.time()
    output, _ = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H, W), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)

    # *********************************** Save result ***********************************
    # Save superpixel segmentation map as mat
    example = scio.loadmat('example.mat')
    spix_index = spixel_label_map.reshape(H, W) + 1
    example['segs'][0][0] = spix_index
    mat_dir = os.path.join(save_path, 'mat', str(n_spixel))
    if not os.path.isdir(mat_dir):
        os.makedirs(mat_dir)
    scio.savemat(os.path.join(mat_dir, imgId + '.mat'), {'segs': example['segs']})

    # Save superpixel segmentation image
    # if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
    #     os.makedirs(os.path.join(save_path, 'spixel_viz'))
    # spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    # imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))
    # *********************************** Save result ***********************************

    if idx % 10 == 0:
        print("Processing %d" % idx)

    return toc


def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> Fetching img pairs in '{}'".format(data_dir))

    save_path = args.output
    print('=> Will save everything to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tst_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    tst_lst.sort()

    if len(tst_lst) == 0:
        print('Wrong data dir or suffix!')
        exit(1)

    print('{} samples found'.format(len(tst_lst)))

    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.pretrained))
    model = models.__dict__[network_data['arch']](data=network_data).cuda()
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    mean_time = 0
    for n in range(len(tst_lst)):
        time = test(args, model, tst_lst, save_path, n)
        mean_time += time
    print("Avg_time per img: %.3f" % (mean_time / len(tst_lst)))


if __name__ == '__main__':
    main()
