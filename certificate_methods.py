import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys


def probs_to_fit(probs, fit_label, num_classes):
    if fit_label == 'correct' or fit_label == 1:
        return probs
    elif fit_label == 'sbest' or fit_label == 2:
        new_labels = probs.argsort(dim=1)[:,-2]
        new_probs = torch.zeros_like(probs)
        new_probs[np.arange(new_probs.shape[0]), new_labels] = 1
    elif fit_label == 'rand':
        labels = probs.argmax(dim=1)
        new_labels = labels + torch.from_numpy(np.random.choice(num_classes-1, labels.shape[0])) + 1
        new_probs = torch.zeros_like(probs)
        new_probs[np.arange(new_probs.shape[0]), new_labels] = 1
    else:
        labels = probs.argmax(dim=1)
        new_labels = probs.argsort(dim=1)[:,-int(fit_label)]
        new_probs = torch.zeros_like(probs)
        new_probs[np.arange(new_probs.shape[0]), new_labels] = 1
    return new_probs



######## Batch Masked Model object #########
class BatchMaskedModelKcert(nn.Module):
    def __init__(self, n, h, w, K=1, scale=1., bs=1, old_mask=None):
        super(BatchMaskedModelKcert, self).__init__()
        self.n = n
        self.h = h
        self.w = w
        self.scale = scale
        if scale > 1:
            self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.K = K
        self.bs = bs

        # Main mask parameters
        if K == 1:
            self.weights_t = torch.zeros(1, bs, 1, int(h/scale), int(w/scale))
        else:
            self.weights_t = torch.zeros(K+1, bs, 1, int(h/scale), int(w/scale))
        self.weights = nn.Parameter(self.weights_t)

        self.softmax = nn.Softmax(dim = 1)
        self.softmax0 = nn.Softmax(dim = 0)

    def mask(self):
        if self.K == 1:
            mask0 = torch.sigmoid(self.weights)
            masks = torch.cat((mask0, 1. - mask0))
        else:
            masks = self.softmax0(self.weights)
        if self.scale > 1:
            return torch.stack([self.upsample(masks[i]) for i in range(masks.shape[0])])
        else:
            return masks

    def forward(self, model, x, K_idx=0, masks=None, use_logits=False, noise_mean=None, noise_batch=None):
        assert x.shape[0] == self.bs
        if self.K == 1:
            K_idx = 0
        if masks is None:
            masks = self.mask()
        mask = masks[K_idx].repeat(1, self.n, 1, 1)
        noise_mask = masks[-1].repeat(1, self.n, 1, 1)
        masked_x = x * mask
        model.eval()

        # Compute noise. Check if other images are provided as noised samples
        # If not, just use the mean value
        if noise_batch is None:
            if not noise_mean is None:
                assert noise_mean.shape == x[0].shape
                noise = noise_mean.cuda()
            else:
                noise = torch.zeros(size=(self.bs, self.n, self.h, self.w)).cuda()
            noised_x = noise * noise_mask

            mod_x = masked_x + noised_x
            logits = model(mod_x)
            probs = self.softmax(logits)

        # If a batch is provided, compute logits for all noise choices
        else:
            assert(len(noise_batch.shape) == 5)
            # Get tensors in good shape
            if noise_batch.shape[0] == 1:
                noise_batch = noise_batch.repeat(self.bs, 1, 1, 1, 1)
            masked_x = masked_x.unsqueeze(1).repeat(1, noise_batch.shape[1], 1, 1, 1)
            noise_mask = noise_mask.unsqueeze(1).repeat(1, noise_batch.shape[1], 1, 1, 1)

            # modified input
            mod_x_batch = masked_x + noise_batch.cuda() * noise_mask
            shape = mod_x_batch.shape
            all_logits = model(mod_x_batch.view(-1, *shape[2:]))
            logits = all_logits.view(*shape[:2], -1).mean(dim=1)

            all_probs = self.softmax(all_logits)
            probs = all_probs.view(*shape[:2], -1).mean(dim=1)

        if use_logits:
            return logits

        return probs



######## Batch Masked Model training code #########
class EntLoss(nn.Module):
    def __init__(self):
        super(EntLoss, self).__init__()

    def forward(self, x):
        eps = 1e-9
        ent = x * torch.log(x + eps)
        return -1.0 * ent.sum()

def batch_mask_Kcert_ent(masks):
    ent_loss = EntLoss()
    return ent_loss(masks)


def batch_mask_Kcert_TV(masks):
    f = torch.square # abs
    shift = None #1
    masks = masks.squeeze(2)[:-1]
    if shift is not None:
        k, bs, h, w = masks.shape
        distance = 0
        for i in range(0, shift + 1):
            for j in range(-shift, shift + 1):
                if i > 0 or j > 0:
                    # print(masks[:,:,:-i,:-j].shape, masks[:,:,i:,j:].shape)
                    if j >= 0:
                        distance += f(masks[:,:,:h-i,:w-j] - masks[:,:,i:,j:]).sum()
                    else:
                        distance += f(masks[:,:,:h-i,-j:] - masks[:,:,i:,:w+j]).sum()
        return distance / (0.25 * ((shift + shift + 1) * (shift + shift + 1) - 1))
    else:
        return f(masks[:,:,:-1,:-1] - masks[:,:,1:,:-1]).sum() +\
            f(masks[:,:,:-1,:-1] - masks[:,:,:-1,1:]).sum()


def batch_mask_Kcert_l1(masks):
    return masks[:-1].sum()


def learn_masks_for_batch_Kcert(model, images, target_probs, K=1, K_bs=None, scale=1,
                                opt=optim.Adam, lr=0.05, steps=2000, obj='xent',
                                noise_mean=None, noise_batch=None, noise_bs=1,
                                reg_l1=0.001, reg_tv=0., reg_ent=0., old_mask=None, debug=True):
    print(K, scale, lr, steps, obj, noise_bs, reg_l1, reg_tv, reg_ent)
    model.eval()
    bs, n, h, w = images.shape
    images = images.cuda()
    target_labels = target_probs.argmax(dim=1).cuda()

    batch_masked_model = BatchMaskedModelKcert(n, h, w, K=K, scale=float(scale), bs=bs, old_mask=old_mask).cuda()
    optimizer = opt([batch_masked_model.weights], lr=lr)
    xent = nn.CrossEntropyLoss(reduction='sum')
    if K_bs is None:
        K_bs = K

    # Run for a few steps
    for iteration in range(steps):
        optimizer.zero_grad()

        # Strategy for "noise"
        if not noise_batch is None:
            noise_idx = np.random.choice(noise_batch.shape[0], noise_bs * bs)
            noise_batch_small = noise_batch[noise_idx]
            shape = noise_batch_small.shape
            noise_batch_small = noise_batch_small.reshape(bs, noise_bs, *shape[1:]).cuda()
        else:
            noise_batch_small = None

        # Main loss
        loss = 0.
        masks = batch_masked_model.mask()
        for K_idx in np.random.choice(K, K_bs, replace=False):
            pred_probs = batch_masked_model(model, images, K_idx=K_idx, masks=masks, use_logits=False,
                                            noise_mean=noise_mean, noise_batch=noise_batch_small)
            if obj == 'xent':
                loss += xent(pred_probs, target_labels)
            else:
                loss += torch.abs(pred_probs[np.arange(bs), target_labels] -\
                                  target_probs[np.arange(bs), target_labels]).sum()
        loss *= K / K_bs
        
        # Regularizations
        torch_zero = torch.tensor([0.])
        l1_norm, tv_term, ent_term = torch_zero, torch_zero, torch_zero
        if reg_l1 != 0:
            l1_norm = batch_mask_Kcert_l1(masks)
            loss += reg_l1 * l1_norm
        if reg_tv != 0:
            tv_term = batch_mask_Kcert_TV(masks)
            loss += reg_tv * tv_term
        if reg_ent != 0:
            ent_term = batch_mask_Kcert_ent(masks) / (h*w)
            loss += reg_ent * ent_term
        loss /= (bs * K)
        loss.backward()
        optimizer.step()
    
        if (iteration % 200 == 0 and debug) or iteration == steps-1:
            with torch.no_grad():
                l1_norm = batch_mask_Kcert_l1(batch_masked_model.mask()).item() / bs
                tv_term = batch_mask_Kcert_TV(batch_masked_model.mask()).item() / bs
                ent_term = batch_mask_Kcert_ent(batch_masked_model.mask()
                                               ).item() / (h*w) / bs
                print('\r{0}: loss: {1:.2f}, l1 norm: {2:.0f}, tv: {3:.2f}, ent: {4:.2f}, pred prob: {5:.4f}'.format(
                    iteration, loss.item(), l1_norm, tv_term, ent_term,
                    pred_probs[np.arange(bs), target_labels].mean()), end='')
                sys.stdout.flush()

    print('')

    return batch_masked_model



######## Mask evaluation code #########

def sparse_mask(mask, frac=0.1, sample=False):
    mask_np = mask.detach().cpu().numpy()
    # Sample or pick top fraction
    if sample:
        top_indices = np.random.choice(np.arange(mask_np.size), int(frac*mask_np.size), replace=False, p=np.ravel(mask_np)/mask_np.sum())
        top_indices = np.unravel_index(top_indices, mask_np.shape)
    else:
        if frac == 0:
            top_indices = np.array([])
        else:
            top_indices = np.unravel_index(np.argsort(mask_np.reshape(-1))[-int(frac*mask_np.size):], mask_np.shape)
    new_mask = torch.zeros_like(mask)
    new_mask[top_indices] = 1.

    return new_mask



def sparse_mask_mass(mask, frac=0.1):
    mask = mask.detach().cpu()
    _, h, w = mask.shape
    tp = h * w

    # get normalized mask
    mask_norm = mask / torch.sum(mask)
    mask_sorted, mask_norm_inds = torch.sort(mask_norm.view(-1, tp)[0], descending=True)
    mask_cum = torch.cumsum(mask_sorted, 0)
    mask_norm_indsx, mask_norm_indsy = np.unravel_index(mask_norm_inds.cpu().detach(), (h,w))

    flat_inds_ofsorted = np.where(mask_cum <= frac)[0]
    num_pix = min(len(flat_inds_ofsorted) + 1, tp)
    inds_of_origx = mask_norm_indsx[np.arange(num_pix)]
    inds_of_origy = mask_norm_indsy[np.arange(num_pix)]
    new_mask = torch.zeros((1,h,w))
    new_mask[0, inds_of_origx, inds_of_origy] = 1.
    return new_mask




def eval_mask_noise_mean_AUC(image, label, mask, og_model, noise_mean=None,
                             mode='ins', step=1, bs=100, mass=False, requires_grad=False, normalize=None, start=0, end=1):
    og_model.eval()
    n, h, w = image.shape
    tp = h * w
    
    if mode == 'del':
        mask = -mask
    if noise_mean is None:
        noise_mean = torch.zeros(size=(n, h, w))
    noise_mean = noise_mean.cuda()
    if requires_grad:
        image = image.clone().detach().cuda().requires_grad_(requires_grad)
    else:
        image = image.cuda()

    if normalize is not None:
        image = normalize(image)

    pred_probs = []
    mask_queue = []

    if mass:
        frac_list = np.arange(0, 1+step, step)
    else:
        frac_list = np.arange(0, tp+1, step)
    if mode == 'del':
        frac_list = frac_list[:-1]
    else:
        frac_list = frac_list[1:]
    frac_list = frac_list[int(len(frac_list) * start): int(len(frac_list) * end)]
    grad_sum = None
    for i in frac_list:
        if mass:
            mask_i = sparse_mask_mass(mask, frac=i)
        else:
            mask_i = sparse_mask(mask, frac=float(i)/tp, sample=False)

        mask_queue.append(mask_i)

        if len(mask_queue) == bs or frac_list[-1] == i:
            mask_queue = torch.stack(mask_queue).cuda()
            mod_image = image[None,:,:,:] * mask_queue.repeat(1,n,1,1) +\
                        noise_mean[None,:,:,:] * (1. - mask_queue.repeat(1,n,1,1))
            
            if requires_grad is False:
                with torch.no_grad():
                    pred_probs.append(nn.Softmax(dim=1)(og_model(mod_image))[:,label].cpu().numpy())
            else:
                pred = nn.Softmax(dim=1)(og_model(mod_image))
                loss = torch.sum(pred[:, label])
                grad, = torch.autograd.grad(loss, [image])
                grad_sum = grad if grad_sum is None else grad_sum + grad
                pred_probs.append(pred[:, label].cpu().detach().numpy())

            mask_queue = []
    
    # if len(mask_queue) > 0:
    #     mask_queue = torch.stack(mask_queue).cuda()
    #     mod_image = image[None,:,:,:] * mask_queue.repeat(1,n,1,1) +\
    #                 noise_mean[None,:,:,:] * (1. - mask_queue.repeat(1,n,1,1))
    #     with torch.no_grad():
    #         pred_probs.append(nn.Softmax(dim=1)(og_model(mod_image))[:,label].cpu().numpy())

    if requires_grad:
        return np.hstack(pred_probs), grad_sum / len(frac_list)
    else:
        return np.hstack(pred_probs)


def eval_mask_noise_mean_AUC_scores(images, labels, masks, og_model, **kwargs):
    og_model.eval()
    masks = masks.detach()
    AUC_ins_probs, AUC_del_probs = eval_mask_noise_mean_prob_values(
        images, labels, masks, og_model, **kwargs)
    return np.mean(AUC_ins_probs, axis=1), np.mean(AUC_del_probs, axis=1)



def eval_mask_noise_mean_prob_values(images, labels, masks, og_model, step=1, noise_mean=None, bs=100, mass=False, **kwargs):
    AUC_ins_list, AUC_del_list = [], []
    for i in range(images.shape[0]):
        AUC_ins_list.append(eval_mask_noise_mean_AUC(images[i], labels[i], masks[i], og_model, mode='ins',
                                                     step=step, noise_mean=noise_mean, bs=bs, mass=mass, **kwargs))
        AUC_del_list.append(eval_mask_noise_mean_AUC(images[i], labels[i], masks[i], og_model, mode='del',
                                                     step=step, noise_mean=noise_mean, bs=bs, mass=mass, **kwargs))
    return np.array(AUC_ins_list), np.array(AUC_del_list)



def eval_mask_noise_batch_AUC(image, label, mask, og_model, noise_batch=None,
                              step=1, mode='ins', bs=10, mass=False):
    og_model.eval()
    n, h, w = image.shape
    tp = h * w
    
    if mode == 'del':
        mask = -mask
    assert noise_batch is not None
    noise_batch = noise_batch.cuda()
    noise_bs = noise_batch.shape[0]
    image = image.cuda()
    
    pred_probs = []
    mask_queue = []

    if mass:
        frac_list = np.arange(0, 1+step, step)
    else:
        frac_list = np.arange(0, tp+1, step)
    for i in frac_list:
        if mass:
            mask_i = sparse_mask_mass(mask, frac=i)
        else:
            mask_i = sparse_mask(mask, frac=float(i)/tp, sample=False)

        mask_queue.append(mask_i)

        if len(mask_queue) == bs:
            mask_queue = torch.stack(mask_queue).cuda()
            image_review = image.unsqueeze(0).repeat(noise_bs, bs, 1, 1, 1)
            mask_queue_review = mask_queue.unsqueeze(0).repeat(noise_bs, 1, n, 1, 1)
            noise_batch_review = noise_batch.unsqueeze(1).repeat(1, bs, 1, 1, 1)

            mod_image = image_review * mask_queue_review + noise_batch_review * (1. - mask_queue_review)
            shape = mod_image.shape
            mod_image_review = mod_image.view(-1, *shape[2:])
            with torch.no_grad():
                probs = nn.Softmax(dim=1)(og_model(mod_image_review))[:,label].view(*shape[:2]).mean(dim=0).cpu().numpy()
                pred_probs.append(probs)
            
            mask_queue = []
    
    if len(mask_queue) > 0:
        bs = len(mask_queue)
        mask_queue = torch.stack(mask_queue).cuda()
        image_review = image.unsqueeze(0).repeat(noise_bs, bs, 1, 1, 1)
        mask_queue_review = mask_queue.unsqueeze(0).repeat(noise_bs, 1, n, 1, 1)
        noise_batch_review = noise_batch.unsqueeze(1).repeat(1, bs, 1, 1, 1)

        mod_image = image_review * mask_queue_review + noise_batch_review * (1. - mask_queue_review)
        shape = mod_image.shape
        mod_image_review = mod_image.view(-1, *shape[2:])
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(og_model(mod_image_review))[:,label].view(*shape[:2]).mean(dim=0).cpu().numpy()
            pred_probs.append(probs)
    
    return np.hstack(pred_probs)


def eval_mask_noise_batch_AUC_scores(images, labels, masks, og_model, **kwargs):
    og_model.eval()
    masks = masks.detach()
    AUC_ins_probs, AUC_del_probs = eval_mask_noise_batch_prob_values(
        images, labels, masks, og_model, **kwargs)
    return np.mean(AUC_ins_probs[1:]), np.mean(AUC_del_probs[:-1])



def eval_mask_noise_batch_prob_values(images, labels, masks, og_model, noise_batch=None,
                                      noise_bs=40, step=1, bs=100, mass=False):
    AUC_ins_list, AUC_del_list = [], []
    for i in range(images.shape[0]):
        noise_idx = np.random.choice(noise_batch.shape[0], noise_bs, replace=False)
        AUC_ins_list.append(eval_mask_noise_batch_AUC(images[i], labels[i], masks[i], og_model, mode='ins',
                                                     step=step, noise_batch=noise_batch[noise_idx], bs=bs, mass=mass))
        AUC_del_list.append(eval_mask_noise_batch_AUC(images[i], labels[i], masks[i], og_model, mode='del',
                                                     step=step, noise_batch=noise_batch[noise_idx], bs=bs, mass=mass))
    return np.array(AUC_ins_list), np.array(AUC_del_list)



# def eval_mask(image, mask, og_model, noise_mean=None, noise_batch=None):
#     og_model.eval()
#     new_mask = mask.detach()

#     # Simple masking, replacing masked pixels with some fixed pixels values
#     if noise_batch is None:
#         if noise_mean is None:
#             noise_mean = torch.zeros_like(image).cuda()
#         mod_image = image * new_mask + noise_mean * (1. - new_mask)
#         with torch.no_grad():
#             pred_probs = nn.Softmax(dim=1)(og_model(mod_image.unsqueeze(0)))[0]
#             return pred_probs
#     else:
#         masked_image = image * new_mask
#         mod_images = torch.stack([masked_image + noise.cuda() * (1. - new_mask) for noise in noise_batch])
#         with torch.no_grad():
#             pred_probs = nn.Softmax(dim=1)(og_model(mod_images)).mean(dim=0)
#         return pred_probs

