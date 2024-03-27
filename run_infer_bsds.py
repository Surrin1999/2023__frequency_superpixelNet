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
import pdb
import scipy.io as scio
from torchvision.transforms.transforms import RandomRotation

import sys

sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='BSDS500/',
                    help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                    default='./pretrain_ckpt/model_best.tar')
parser.add_argument('--output', metavar='DIR', default='output/BSDS500/', help='path to output folder')

parser.add_argument('--downsize', default=16, type=float, help='superpixel grid cell, must be same as training setting')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

# the BSDS500 has two types of image, horizontal and veritical one, here I use train_img and input_img to presents them respectively
parser.add_argument('--train_img_height', '-t_imgH', default=320, type=int, help='img height must be 16*n')
parser.add_argument('--train_img_width', '-t_imgW', default=480, type=int, help='img width must be 16*n')
parser.add_argument('--input_img_height', '-v_imgH', default=480, type=int, help='img height_must be 16*n')  #
parser.add_argument('--input_img_width', '-v_imgW', default=320, type=int, help='img width must be 16*n')

args = parser.parse_args()
args.test_list = args.data_dir + '/test.txt'

random.seed(100)


@torch.no_grad()
def test(model, img_paths, save_path, spixeIds, idx, scale):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # origin size 481*321 or 321*481
    img_ = cv2.imread(load_path)
    H_, W_, _ = img_.shape

    # choose the right spixelIndx
    if H_ == 321 and W_ == 481:
        spixl_map_idx_tensor = spixeIds[0]
        img = cv2.resize(img_, (int(480 * scale), int(320 * scale)), interpolation=cv2.INTER_CUBIC)
    elif H_ == 481 and W_ == 321:
        spixl_map_idx_tensor = spixeIds[1]
        img = cv2.resize(img_, (int(320 * scale), int(480 * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        print('The image size is wrong!')
        return

    img1 = input_transform(img)

    # compute output
    tic = time.time()
    output, _ = model(img1.cuda().unsqueeze(0))

    # assign the spixel map and  resize to the original size
    curr_spixl_map = update_spixl_map(spixl_map_idx_tensor, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1)
    spix_index_np = spix_index_np.astype(np.int64)
    segment_size = (spix_index_np.shape[0] * spix_index_np.shape[1]) / (int(600 * scale * scale) * 1.0)
    min_size = int(0.06 * segment_size)
    max_size = int(3 * segment_size)
    spixel_label_map = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]

    torch.cuda.synchronize()
    toc = time.time() - tic

    n_spixel = len(np.unique(spixel_label_map))

    example = scio.loadmat('example.mat')
    example['segs'][0][0] = spixel_label_map + 1
    scio.savemat(os.path.join(save_path, imgId + ".mat"), {'segs': example['segs']})

    if idx % 10 == 0:
        print("%d superpixels, processing %d" % (n_spixel, idx))

    return toc, n_spixel


def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    train_img_height = args.train_img_height
    train_img_width = args.train_img_width
    input_img_height = args.input_img_height
    input_img_width = args.input_img_width

    mean_time_list = []
    # The spixel number we test
    for scale in [0.4, 0.6, 0.8, 0.9, 1.0]:
        assert (320 * scale % 16 == 0 and 480 * scale % 16 == 0)
        save_path = args.output + '/test_multiscale_enforce_connect/SPixelNet_nSpixel_{0}'.format(
            int(20 * scale * 30 * scale))

        args.train_img_height, args.train_img_width = train_img_height * scale, train_img_width * scale
        args.input_img_height, args.input_img_width = input_img_height * scale, input_img_width * scale

        print('=> will save everything to {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        tst_lst = os.listdir('input/BSDS500/test')
        tst_lst = [os.path.join('input/BSDS500/test', elem) for elem in tst_lst]
        # tst_lst = []
        # with open(args.test_list, 'r') as tf:
        #     img_path = tf.readlines()
        #     for path in img_path:
        #         tst_lst.append(path[:-1])

        print('{} samples found'.format(len(tst_lst)))

        # create model
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained model '{}'".format(args.pretrained))
        model = models.__dict__[network_data['arch']](data=network_data).cuda()
        model.eval()
        args.arch = network_data['arch']
        cudnn.benchmark = True

        # for vertical and horizontal input seperately
        spixlId_1, _ = init_spixel_grid(args, b_train=True)
        spixlId_2, _ = init_spixel_grid(args, b_train=False)
        mean_time = 0
        # the following code is for debug
        for n in range(len(tst_lst)):
            time, n_spixel = test(model, tst_lst, save_path, [spixlId_1, spixlId_2], n, scale)
            mean_time += time
        mean_time /= len(tst_lst)
        mean_time_list.append((n_spixel, mean_time))

        print("for spixel number {}: with mean_time {} , generate {} spixels".format(int(20 * scale * 30 * scale),
                                                                                     mean_time, n_spixel))
        for item in mean_time_list:
            print('SP Num: {}, Time: {}'.format(item[0], item[1]))

    # with open(args.output + 'test_multiscale_enforce_connect/mean_time.txt', 'w+') as f:
    #     for item in mean_time_list:
    #         tmp = "{}: {}\n".format(item[0], item[1])
    #         f.write(tmp)


if __name__ == '__main__':
    main()
