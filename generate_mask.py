import copy
import os
from collections import OrderedDict
from rich import print as rich_print

import arg_parser
import evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import unlearn
import utils
from scipy.special import erf
from trainer import validate
from imagenet import get_x_y_from_data_dict
import time

def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    gradients = {}
    forget_loader = data_loaders["forget"]
    model.eval()
    for name, param in model.named_parameters():
        gradients[name] = 0

    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

    else:
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()
            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1,0.3,0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}
        # Concatenate all tensors into a single tensor
        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])
        # Calculate the threshold index for the top i fraction of elements
        threshold_index = int(len(all_elements) * i)
        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]
            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions
            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        all_gradients = torch.cat(
            [gradient.flatten() for gradient in gradients.values()]
        )

        sigmoid_gradients = torch.abs(2 * (torch.sigmoid(all_gradients) - 0.5))
        tanh_gradients = torch.abs(torch.tanh(all_gradients))

        sigmoid_soft_dict = {}
        tanh_soft_dict = {}
        start_idx = 0
        for net_name, gradient in gradients.items():
            num_params = gradient.numel()
            end_idx = start_idx + num_params
            sigmoid_gradient = sigmoid_gradients[start_idx:end_idx]
            sigmoid_gradient = sigmoid_gradient.reshape(gradient.shape)
            sigmoid_soft_dict[net_name] = sigmoid_gradient

            tanh_gradient = tanh_gradients[start_idx:end_idx]
            tanh_gradient = tanh_gradient.reshape(gradient.shape)
            tanh_soft_dict[net_name] = tanh_gradient
            start_idx = end_idx

        if args.mem is None:
            torch.save(hard_dict, os.path.join(args.save_dir, "hard_{}_{}_{}_{}_num{}_groupid{}_seed{}.pt".format(
                i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.seed)))
        else:
            # torch.save(
            #     sigmoid_soft_dict,
            #     os.path.join(args.save_dir, "sigmoid_soft_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
            #         i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
            #     args.group_index, args.mem, args.seed)),
            # )
            # torch.save(
            #     tanh_soft_dict, os.path.join(args.save_dir, "tanh_soft_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
            #         i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
            #     args.group_index, args.mem, args.seed))
            # )
            torch.save(hard_dict, os.path.join(args.save_dir, "hard_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
                i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem, args.seed)))


def load_pth_tar_files(folder_path):
    pth_tar_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                pth_tar_files.append(file_path)

    return pth_tar_files


def compute_gradient_ratio(mask_path):
    mask = torch.load(mask_path)
    all_elements = torch.cat([tensor.flatten() for tensor in mask.values()])
    ones_tensor = torch.ones(all_elements.shape)
    ratio = torch.sum(all_elements) / torch.sum(ones_tensor)
    name = mask_path.split("/")[-1].replace(".pt", "")
    return name, ratio


def print_gradient_ratio(mask_folder, save_path):
    ratio_dict = {}
    mask_path_list = load_pth_tar_files(mask_folder)
    for i in mask_path_list:
        name, ratio = compute_gradient_ratio(i)
        print(name, ratio)
        ratio_dict[name] = ratio.item()

    ratio_df = pd.DataFrame([ratio_dict])
    ratio_df.to_csv(save_path + "ratio_df.csv", index=False)


def main():
    start_rte = time.time()
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    if "_orig_" in args.mask:
        args.save_dir = args.save_dir + '/sgd'
    elif "_sam-min_" in args.mask:
        args.save_dir = args.save_dir + '/sam'
        i = args.mask.find("rho")
        args.save_dir = args.save_dir + '-' + args.mask[i:i+6]
    elif "_sam-max_" in args.mask:
        args.save_dir = args.save_dir + '/sharpmax'
        i = args.mask.find("rho")
        args.save_dir = args.save_dir + '-' + args.mask[i:i+6]

    if "_adaptTrue_" in args.mask:
        args.save_dir = args.save_dir + '-adaptTrue'
    elif "_adaptFalse_" in args.mask:
        args.save_dir = args.save_dir + '-adaptFalse'

    os.makedirs(args.save_dir, exist_ok=True)
    print(f'Saving to directory: {args.save_dir}')
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
            train_idx
        ) = utils.setup_model_dataset(args)
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
    elif args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader_full, val_loader = utils.setup_model_dataset(args)
    elif args.dataset == "tiny_imagenet":
        args.class_to_replace = None
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    model.cuda()
    rich_print(args)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=12,
            pin_memory=True,
            shuffle=shuffle,
        )

    if args.mem is not None and args.group_index is None and args.mem_proxy is None:
        fine_overlap = False
        mem_fs_split = True
        proxy_fs_split = False
    elif args.mem is None and args.group_index is not None and args.mem_proxy is None:
        fine_overlap = True
        mem_fs_split = False
        proxy_fs_split = False
    elif args.mem_proxy is not None:
        fine_overlap = False
        mem_fs_split = False
        proxy_fs_split = True
    else:
        fine_overlap = False
        mem_fs_split = False
        proxy_fs_split = False

    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    elif mem_fs_split:
        print('[fs split]: mem_fs_split')
        train_loader = DataLoader(
            train_loader_full.dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        if args.dataset == 'imagenet':
            imagenet_indices = np.load(f'imagenet_filenames_indices_{args.num_indexes_to_replace}_{args.seed}.npz')
            # Get the indices for each set of filenames
            high_mem_indices = imagenet_indices['high_mem_indices']
            low_mem_indices = imagenet_indices['low_mem_indices']
            med_mem_indices = imagenet_indices['med_mem_indices']

            assert len(high_mem_indices) == args.num_indexes_to_replace
            assert len(low_mem_indices) == args.num_indexes_to_replace
            assert len(med_mem_indices) == args.num_indexes_to_replace
            
            # Decide which set to use as forget set based on args.mem
            print('check: args.mem: ', args.mem)
            if args.mem == 4:  # high
                forget_indices = high_mem_indices
            elif args.mem == 0:  # low
                forget_indices = low_mem_indices
            elif args.mem == 2:  # medium
                forget_indices = med_mem_indices
            elif args.mem is None:  # mix
                forget_indices  = (high_mem_indices[:args.num_indexes_to_replace // 3] + 
                                    med_mem_indices[:args.num_indexes_to_replace // 3] + 
                                    low_mem_indices[-args.num_indexes_to_replace // 3:])
            else:
                raise ValueError('Invalid mem value')
            forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_indices))
            all_indices = set(range(len(train_loader.dataset)))
            # Determine retain filenames based on sequential flag
            if args.sequential:
                # Sequential unlearning logic
                if args.mem == 0:  # low
                    retain_filenames = all_indices - set(low_mem_indices)
                elif args.mem == 2:  # medium
                    retain_filenames = all_indices - set(low_mem_indices + med_mem_indices)
                elif args.mem == 4:  # high
                    retain_filenames = all_indices - set(low_mem_indices + med_mem_indices + high_mem_indices)
            else:
                retain_indices = all_indices - set(forget_indices)
                print('check 2, retain set size: ', len(retain_indices))

            print(f'Number of forget files: {len(forget_indices)}')
            print(f'Number of retain files: {len(retain_indices)}')
            print(f'Total files of original trainloader: {len(train_loader.dataset)}')
            
            retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_indices))
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

            hm = high_mem_indices[:1000]
            lm = low_mem_indices[-1000:]
            mm = med_mem_indices[:1000]
            hm_dataset = torch.utils.data.Subset(train_loader.dataset, list(hm))
            mm_dataset = torch.utils.data.Subset(train_loader.dataset, list(mm))
            lm_dataset = torch.utils.data.Subset(train_loader.dataset, list(lm))
            hm_loader = replace_loader_dataset(hm_dataset, seed=seed, shuffle=True)
            mm_loader = replace_loader_dataset(mm_dataset, seed=seed, shuffle=True)
            lm_loader = replace_loader_dataset(lm_dataset, seed=seed, shuffle=True)
        else:
            if args.dataset == 'cifar10':
                loaded_results = np.load('estimates_results_woShuffle.npz')
                loaded_memorization = loaded_results['memorization']
            elif args.dataset == 'cifar100':
                loaded_results = np.load('cifar100_infl_matrix.npz')
                loaded_memorization = loaded_results['tr_mem']
            loaded_memorization = loaded_memorization[train_idx]

            indices = list(range(len(train_loader.dataset)))
            indices_mem = list(zip(indices, loaded_memorization))

            indices_mem.sort(key=lambda x: x[1], reverse=True)
            h_mem_list = indices_mem[:args.num_indexes_to_replace]
            l_mem_list = indices_mem[-args.num_indexes_to_replace:]
            indices_mem.sort(key=lambda x: abs(x[1] - 0.5))
            m_mem_list = indices_mem[:args.num_indexes_to_replace]

            if args.shuffle:
                indices_mem_mix = h_mem_list + l_mem_list + m_mem_list
                np.random.shuffle(indices_mem_mix)
                h_mem_list = indices_mem_mix[:args.num_indexes_to_replace]
                l_mem_list = indices_mem_mix[-args.num_indexes_to_replace:]
                m_mem_list = indices_mem_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
            else:
                pass

            h_mem_idx, h_mem = zip(*h_mem_list)
            l_mem_idx, l_mem = zip(*l_mem_list)
            m_mem_idx, m_mem = zip(*m_mem_list)
            print('check: h_mem: ', h_mem[:100])
            print('check: l_mem: ', l_mem[:100])
            print('check: m_mem: ', m_mem[:100])

            # changed from low mid high to numbers to interface
            print('check: args.mem: ', args.mem)
            if args.mem == 4:
                forget_dataset_indices = h_mem_idx
            elif args.mem == 0:
                forget_dataset_indices = l_mem_idx
            elif args.mem == 2:
                forget_dataset_indices = m_mem_idx
            elif args.mem is None:
                hm = h_mem_idx[:args.num_indexes_to_replace // 3]
                mm = m_mem_idx[:args.num_indexes_to_replace // 3]
                lm = l_mem_idx[-args.num_indexes_to_replace // 3:]
                forget_dataset_indices = hm + mm + lm
            else:
                raise ValueError('Invalid mem value')

            forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
            all_indices = set(range(len(train_loader.dataset)))
            if args.sequential: # changed from low mid high to numbers to interface
                if args.mem == 0:
                    retain_dataset_indices = all_indices - set(l_mem_idx)
                elif args.mem == 2:
                    retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx)
                elif args.mem == 4:
                    retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx + h_mem_idx)
                print('check 2, retain set size: ', len(retain_dataset_indices))
            else:
                retain_dataset_indices = all_indices - set(forget_dataset_indices)
                print('check 2, retain set size: ', len(retain_dataset_indices))

            retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    elif args.dataset == 'tiny_imagenet':
        print('tiny imagenet prepares F_rand during setup_model_dataset().')
    else:
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    if args.dataset == 'imagenet' or args.dataset == 'tiny_imagenet':
        test_loader = val_loader
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        start_unlearn = time.time()
        save_gradient_ratio(unlearn_data_loaders, model, criterion, args)

    end_rte = time.time()
    print(f'Overall time taken for unlearning & preparation: {end_rte - start_rte:.3f}s')
    print(f'Time taken for generating mask only: {end_rte - start_unlearn:.3f}s')

if __name__ == "__main__":
    main()
