'''
python mask_like_certifier_imagenet_desktop.py --label_list 526  527 664 673 782 851   --image_input_path input_data/sample_images.npy --label_input_path input_data/sample_labels.npy  --steps 200 --lr 0.05 --bs 10 --noise_mode rand_image --noise_bs 10 --K 1 \
--scale_list 4 --reg_l1_list 0.00002 --reg_tv_list 0.01 --fit_label correct \
--start_end 0 10 --debug 

'''

cheating = False
curve_start = 0
curve_end = 1
epsilon = 1e-5

import torch
import os
import pickle as pkl
import numpy as np
import torchvision

from certificate_methods import *
from utils import ReducedModel, parse, cifar10_resnet164, cifar10_loaders, completeness, soundness
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageNet
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import itertools
import math

cifar10_classes_name = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    args = parse()
    image_name = args.image_input_path
    label_name = args.label_input_path
    channel, height, width = 3, 224, 224
    num_classes = 10
    tot_pixels = height * width
    dump_dir = 'results'
    label_list = args.label_list
    label_map = {v : k for k, v in enumerate(label_list)}
   
    print('Loading selected data from ', image_name, '  ', label_name)
    all_images = torch.from_numpy(np.load(os.path.join(image_name)))
    all_labels = torch.from_numpy(np.load(os.path.join(label_name)))

    print('Loading resnet50')
    model = torchvision.models.resnet50(pretrained=False).cuda()
    model.load_state_dict(torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    model = ReducedModel(model, label_list)
    model.eval()

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
    

    # Set values according to args
    if noise_mode == 'gray':
        noise_images = None
    elif noise_mode == 'rand_image':
        noise_images = torch.from_numpy(np.load('./noise_images.npy'))
    
    try: 
        os.makedirs(dump_dir) 
    except OSError as error: 
        print(error)


    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    normalize = transforms.Normalize(mean=mean, std=std)
    inv_normalize = transforms.Normalize(
        mean=[-mean[i]/std[i] for i in range(3)],
        std=[1/std[i] for i in range(3)]
    )
    fig = plt.figure()
    to_print = [0, 1, 2, 3, 4, 5]
    rows = len(to_print)
    columns = 3

    clean_correct, adv_correct, adv_correct_fix_mask, total = 0, 0, 0, 0
    agree = 0
    agree_incorrect, incorrect = 0, 0
    completeness_total, soundness_total, worst_soundness_total, sbest_soundness_total = 0, 0, 0, 0
    all_soundness_total, all_completeness_total, worst_completeness_total = 0, 0, 0
    auc_total = 0

    for idx in range(start_idx, min(end_idx, all_images.shape[0]), bs):
        images = all_images[idx : idx+bs]
        labels = all_labels[idx : idx+bs].cpu().numpy()
        
        masks_dict = {}
        for target_label in range(len(label_list)):
            dump_file = os.path.join(dump_dir, 'batch_' + str(idx) + 'label_' + str(label_list[target_label])+ '.pkl')
            masks_dict[target_label] = pkl.load(open(dump_file, 'rb'))
           
        model.eval()
        
        softmax = torch.nn.Softmax(dim = -1)
        with torch.no_grad():
            vanilla_probs = softmax(model(images.cuda()))
            vanilla_order = torch.argsort(vanilla_probs, dim=-1, descending=True).cpu().numpy()
            vanilla_labels = torch.argmax(vanilla_probs, dim=-1).cpu().numpy()
            vanilla_probs = vanilla_probs.detach().cpu().numpy()
        torch.cuda.empty_cache()

        dump_file = os.path.join(dump_dir, f'batchscore_{"cheating_" if cheating else ""}{curve_start}_{curve_end}_' + str(idx) + '.pkl')
        if not os.path.exists(dump_file):
            score_dict = {}
            for scale, reg_l1, reg_tv, reg_ent in itertools.product(
                scale_list, reg_l1_list, reg_tv_list, reg_ent_list):
                
                config = (noise_mode, K, scale, reg_l1, reg_tv, reg_ent)
                
                AUCscores_ins = []
                AUCscores_del = []
                reg_scores = []
                for target_label in range(len(label_list)):
                    masks = torch.from_numpy(masks_dict[target_label][config])
                    while len(masks.shape) < 5:
                        masks = masks.unsqueeze(0)
                    assert(masks.shape == torch.Size([1,len(labels),1,224,224]))
                    
                    if cheating:
                        for i in range(len(labels)):
                            if target_label != vanilla_labels[i]:
                                masks[0, i, 0, :, :] = torch.from_numpy(masks_dict[vanilla_labels[i]][config][0, i, 0, :, :])
                    
                    probs = torch.zeros(images.shape[0], len(label_list))
                    probs[:, target_label] = 1
                    score_ins, score_del = eval_mask_noise_mean_AUC_scores(
                        images, probs.argmax(dim=1), masks[0], model,
                        step=224*8, noise_mean=None, bs=50, start=curve_start, end=curve_end)
                    
                    reg_score = np.zeros(len(labels))
                    for i in range(len(labels)):
                        loss = 0
                        if reg_tv != 0:
                            loss += reg_tv * batch_mask_Kcert_TV(masks[:, i].unsqueeze(1).repeat(2,1,1,1,1))
                       
                        reg_score[i] = loss
                
                    AUCscores_ins.append(score_ins)
                    AUCscores_del.append(score_del)
                    reg_scores.append(reg_score)
            
                AUCscores_ins = np.array(AUCscores_ins)
                AUCscores_del = np.array(AUCscores_del)
                reg_scores = np.array(reg_scores)

                score_dict[config] = {'AUC_ins': AUCscores_ins, 'AUC_del': AUCscores_del, 'reg': reg_scores}
            
            pkl.dump(score_dict, open(dump_file, 'wb'))
            torch.cuda.empty_cache()
        else:
            score_dict = pkl.load(open(dump_file, 'rb'))

        for scale, reg_l1, reg_tv, reg_ent in itertools.product(
            scale_list, reg_l1_list, reg_tv_list, reg_ent_list):
            
            config = (noise_mode, K, scale, reg_l1, reg_tv, reg_ent)
            
            AUCscores_ins = score_dict[config]['AUC_ins']
            AUCscores_del = score_dict[config]['AUC_del']
            reg_scores = score_dict[config]['reg']


            output = np.argmax(AUCscores_ins - 0.0 * reg_scores, axis=0)
           
            clean_correct += np.sum(output == labels)
            agree += np.sum(output == vanilla_labels)
            total += len(labels)
            incorrect += np.sum(np.not_equal(labels, vanilla_labels))
            agree_incorrect += np.sum(np.not_equal(labels, vanilla_labels) * (output == vanilla_labels))
           
            AUC_probs = AUCscores_ins.transpose()
           
            completeness_vec = completeness(AUC_probs, vanilla_probs, epsilon1=0.1)
            completeness_total += np.sum(completeness_vec[range(len(labels)), vanilla_labels])
            soundness_vec = soundness(AUC_probs, vanilla_probs, 0.1)
            soundness_total += (np.sum(soundness_vec) - np.sum(soundness_vec[range(len(labels)), vanilla_labels])) / 9
            for i in range(len(labels)):
                all_soundness_total += np.min(soundness_vec[i])
                worst_soundness_total += np.min(np.append(soundness_vec[i][: vanilla_labels[i]],soundness_vec[i][vanilla_labels[i] + 1:]))
                all_completeness_total += np.min(completeness_vec[i])
                worst_completeness_total += np.min(np.append(completeness_vec[i][: vanilla_labels[i]],completeness_vec[i][vanilla_labels[i] + 1:]))
                sbest = np.argmax(np.append(vanilla_probs[i][: vanilla_labels[i]], vanilla_probs[i][vanilla_labels[i] + 1:]))
                if sbest >= vanilla_labels[i]:
                    sbest += 1
                sbest_soundness_total += soundness_vec[i][sbest]
                if vanilla_probs[i][sbest] > 0.1 and vanilla_probs[i][sbest] < 0.12 and soundness_vec[i][sbest] > 0.8 and completeness_vec[i][sbest] > 0.8:
                    print(idx + i, vanilla_probs[i][sbest], AUC_probs[i][sbest], vanilla_probs[i][vanilla_labels[i]], AUC_probs[i][vanilla_labels[i]])
            auc_total += np.sum(AUC_probs[range(len(labels)), vanilla_labels] * (labels == vanilla_labels))
            if (idx + bs) % 1 == 0:
                
                
            
                print(f'average worst completeness of wrong labels {worst_completeness_total / total}')
                print(f'average worst completeness of all labels {all_completeness_total / total}')
                
                print(f'average worst soundness of all labels {all_soundness_total / total}')
                
           
        


if __name__ == '__main__': main()
