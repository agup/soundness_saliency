
'''
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

python saliency_certificate_Imagenet_desktop.py  --label_list 526  527 664 673 782 851  --image_input_path input_data/sample_images.npy --label_input_path input_data/sample_labels.npy  --steps 2000 --lr 0.05 \
--bs 10 --noise_mode rand_image --noise_bs 10 --K 1 \
--scale_list 4 --reg_l1_list 0.00002 --reg_tv_list 0.01 \
--fit_label correct  \
--start_end 0 300 --debug
'''

import torch
import os
import pickle as pkl
import numpy as np
import torchvision
import itertools

from certificate_methods import *
from utils import ReducedModel, parse
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import cv2

def main():
    args = parse()
    image_name = args.image_input_path
    label_name = args.label_input_path
    channel, height, width = 3, 224, 224
    num_classes = 10
    tot_pixels = height * width
    label_list = args.label_list 
    label_map = {v : k for k, v in enumerate(label_list)}
        
    
    print('Loading selected data from ', image_name, ' and ', label_name)
    all_images = torch.from_numpy(np.load(os.path.join(image_name)))
    all_labels = torch.from_numpy(np.load(os.path.join(label_name)))

    ######### Load the (image, label) data and check model accuracy #########
    print('Using ResNet50 pretrained on imagenet')
    model = torchvision.models.resnet50(pretrained=False).cuda()
    model.load_state_dict(torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    model = ReducedModel(model, label_list)
    model.eval()

  
    ###Create directory to store results if need be #######
    if not os.path.exists('results'):
        os.makedirs('results')


    ####### Start training masks #######
    steps = args.steps
    lr = args.lr
    bs = args.bs
    noise_mode = args.noise_mode
    noise_bs = args.noise_bs
    K = args.K

    reg_l1_list = args.reg_l1_list
    scale_list = args.scale_list
    reg_tv_list = args.reg_tv_list
    reg_ent_list = args.reg_ent_list
    
    fit_label = args.fit_label
    obj = args.obj
    
    start_idx, end_idx = args.start_end
    debug = args.debug
    
    model.eval()
    dataset_size = all_images.shape[0]
    
    def evaluate():
        c = 0
        correct, total = 0, 0
        for idx in range(start_idx, min(end_idx, all_images.shape[0]), bs):
            images = all_images[idx : idx+bs].cuda()
            labels = all_labels[idx : idx+bs].numpy()  
            with torch.no_grad():
                outputs = model(images)
                output_probs = torch.nn.Softmax(dim = 1)(outputs).detach().cpu().numpy()
            output_labels = np.argmax(output_probs, axis = 1)
            correct += np.sum(output_labels == labels)
            total += len(labels)
            np.sort(output_probs, axis=-1)
            c += np.sum(output_probs[-2] > 0.1)
            print(f"{correct}/{total}: {correct / total}")
            print(f"{c}/{total}: {c / total}")
    if noise_mode == 'gray':
        noise_images = None
    elif noise_mode == 'rand_image':
        noise_images = torch.from_numpy(np.load('./noise_images.npy'))
    
    dump_dir =  "results"
    try: 
        os.makedirs(dump_dir) 
    except OSError as error: 
        print(error)

    print('Start training', flush=True)
    for idx in range(start_idx, min(end_idx, all_images.shape[0]), bs):
        images = all_images[idx : idx+bs]
        model.eval()
        # select probabilities to fit the saliency masks to

        for target_label in range(len(label_list)):
            dump_file = os.path.join(dump_dir, f'batch_{idx}label_{label_list[target_label]}.pkl')
            if os.path.exists(dump_file):
                continue
            probs = torch.zeros(images.shape[0], len(label_list))
            probs[:, target_label] = 1
            masks_dict = {}
            for scale, reg_l1, reg_tv, reg_ent in \
                itertools.product(scale_list, reg_l1_list, reg_tv_list, reg_ent_list):

                config = (noise_mode, K, scale, reg_l1, reg_tv, reg_ent)
                print(idx, config, flush=True)
                batch_masked_model = learn_masks_for_batch_Kcert(
                    model, images, probs, K=K, scale=scale,
                    opt=optim.Adam, lr=lr, steps=steps, obj=obj,
                    noise_mean=None, noise_batch=noise_images, noise_bs=noise_bs,
                    reg_l1=reg_l1, reg_tv=reg_tv, reg_ent=reg_ent, old_mask=None, debug=debug)
                masks = batch_masked_model.mask().detach().cpu()[:-1]
                masks_dict[config] = masks.numpy()
            pkl.dump(masks_dict, open(dump_file, 'wb'))
            torch.cuda.empty_cache()


if __name__ == '__main__': main()
