This repository contains code for the paper New Definitions and Evaluations for Saliency Methods : Staying Intrinsic and Sound. NeuRIPS 2022.

To cite this paper, please use the following BibTex entry:

```
@inproceedings{2022soundness,
  title = {New Definitions and Evaluations for Saliency Methods : Staying Intrinsic and Sound},
  author = {Arushi Gupta, Nikunj Saunshi, Dingli Yu, Kaifeng Lyu, Sanjeev Arora},
  booktitle = {NeuRIPS},
  year = {2022}
}
```

We now describe how to run our simple masking method and compute completeness and soundness on the resulting maps.

#First, please run save noise batch to save the batch of images that will be used to create Gamma for our method
#You may not get the exact same results we did because we changed batch_size to 1000 in save_noise_batch.py  so we had 1000 noise images. Here we have set it to 100 because of github file size limitations

```
python3 save_noise_batch.py
```


#Next, we generate some random sample inputs (of course please replace with your own inputs if need be)
#We have generated these images so their labels lie in [526  527 664 673 782 851]
python3 generate_random_inputs.py

#This should save in the folder input_data 2 files. sample_images.npy and sample_labels.npy

#Next, to run our saliency method please type

```
python saliency_certificate_Imagenet_desktop.py  --label_list 526  527 664 673 782 851  --image_input_path input_data/sample_images.npy --label_input_path input_data/sample_labels.npy  --steps 2000 --lr 0.05 \
--bs 10 --noise_mode rand_image --noise_bs 10 --K 1 \
--scale_list 4 --reg_l1_list 0.00002 --reg_tv_list 0.01 \
--fit_label correct  \
--start_end 0 300 --debug
```

#Where label_list is the list of relevant labels from ImageNet (to save the computational burden of computing on 1000 labels) and image_input_path is the path to the images file and label_input_path is the path to the relevant labels.


#Finally, to compute completeness and soundness, please run

```
python mask_like_certifier_imagenet_desktop.py --label_list 526  527 664 673 782 851   --image_input_path input_data/sample_images.npy --label_input_path input_data/sample_labels.npy  --steps 200 --lr 0.05 --bs 10 --noise_mode rand_image --noise_bs 10 --K 1 \
--scale_list 4 --reg_l1_list 0.00002 --reg_tv_list 0.01 --fit_label correct \
--start_end 0 50 --debug 
```

#where start_end 0 50 means indices 0 through 50 will be processed




