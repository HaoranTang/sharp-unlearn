import copy
import os
from collections import OrderedDict
from rich import print as rich_print
import matplotlib.pyplot as plt
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import unlearn
import utils
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from imagenet import get_x_y_from_data_dict
from trainer import validate
from unlearn.impl import wandb_init, wandb_finish
from analysis import *

def select_fs(scores, indices, args, seq_last=False):
    if args.sequential and seq_last == False:
        num_indexes_to_replace = 3000
        mem = 'mix'
        print('--- sequential unlearning: step 0~(n-1) ---')
    elif args.sequential and seq_last == True:
        num_indexes_to_replace = args.num_indexes_to_replace
        mem = args.mem
        print('--- sequential unlearning: step n ---')
    else:
        num_indexes_to_replace = args.num_indexes_to_replace
        mem = args.mem

    selected_scores = scores[indices]

    # indices = list(range(len(train_loader.dataset)))
    indices_scores = list(zip(indices, selected_scores))
    indices_scores.sort(key=lambda x: x[1], reverse=True)

    h_score_list = indices_scores[:num_indexes_to_replace]
    l_score_list = indices_scores[-num_indexes_to_replace:]
    h_score_list_3000 = indices_scores[:3000]
    h_score_values = [x[1] for x in h_score_list_3000]
    median_score = np.median(selected_scores)
    medium_score = (np.min(selected_scores) + np.max(selected_scores)) / 2
    medium_custom = median_score

    # print(f'check: {np.min(heldout_retrain_scores):.3f}, {np.min(h_ret_values):.3f}')
    print(f'check heldout retrain: min: {np.min(selected_scores):.3f}, max: {np.max(selected_scores):.3f}, '
          f'medium: {medium_score:.3f}, median: {median_score:.3f}, medium_custom: {medium_custom:.3f}')

    indices_scores.sort(key=lambda x: abs(x[1] - medium_custom))
    m_score_list = indices_scores[:num_indexes_to_replace]

    if args.shuffle:
        indices_proxy_mix = h_score_list + l_score_list + m_score_list
        np.random.shuffle(indices_proxy_mix)
        h_score_list = indices_proxy_mix[:num_indexes_to_replace]
        l_score_list = indices_proxy_mix[-num_indexes_to_replace:]
        m_score_list = indices_proxy_mix[num_indexes_to_replace:-num_indexes_to_replace]
    else:
        pass

    h_score_idx, h_score = zip(*h_score_list)
    l_score_idx, l_score = zip(*l_score_list)
    m_score_idx, m_score = zip(*m_score_list)

    print(f'check: h_score [{min(h_score):.3f}, {max(h_score):.3f}], examples: {h_score[:10]}')
    print(f'check: m_score [{min(m_score):.3f}, {max(m_score):.3f}], examples: {m_score[:10]}')
    print(f'check: l_score [{min(l_score):.3f}, {max(l_score):.3f}], examples: {l_score[:10]}')

    if mem == 'high':
        forget_dataset_indices = h_score_idx
    elif mem == 'low':
        forget_dataset_indices = l_score_idx
    elif mem == 'mid':
        forget_dataset_indices = m_score_idx
    elif mem == 'mix':
        hc = h_score_idx[:num_indexes_to_replace // 3]
        mc = m_score_idx[:num_indexes_to_replace // 3]
        lc = l_score_idx[-num_indexes_to_replace // 3:]
        forget_dataset_indices = hc + mc + lc
    else:
        raise ValueError('Invalid mem value')

    return forget_dataset_indices, h_score_idx, m_score_idx, l_score_idx

def entanglement_and_visualization(unlearn_data_loaders, model, args):
    model.eval()
    # -------------------------------------- PART 1: EVALUATION -------------------------------------- #
    results = {}
    # Get embeddings and labels
    print("Extracting features...")
    retain_embeddings, retain_labels = get_features_gpu(unlearn_data_loaders["retain"], model, args)
    forget_embeddings, forget_labels = get_features_gpu(unlearn_data_loaders["forget"], model, args)
    # global
    print("Computing variance-based entanglement...")
    results["variance"] = compute_variance_entanglement(
        retain_embeddings, forget_embeddings, retain_labels, forget_labels, class_wise=False
    )
    print("Computing Wasserstein entanglement...")
    results["wasserstein"] = compute_wasserstein_entanglement(
        retain_embeddings, forget_embeddings, retain_labels, forget_labels, class_wise=False
    )
    # class-wise
    print("Computing variance-based entanglement...")
    results["variance_cls"] = compute_variance_entanglement(
        retain_embeddings, forget_embeddings, retain_labels, forget_labels, class_wise=True
    )
    print("Computing Wasserstein entanglement...")
    results["wasserstein_cls"] = compute_wasserstein_entanglement(
        retain_embeddings, forget_embeddings, retain_labels, forget_labels, class_wise=True
    )
    # Display all results
    for metric, value in results.items():
        print(f"{metric.capitalize()} Entanglement: {value:.4f}")
    # -------------------------------------- PART 2: VISUALIZATION -------------------------------------- #
    if args.novisual:
        return results
    # Set up filepath for saving visualizations
    model_name = os.path.split(args.mask)[1]
    filepath = f"visualizations/{model_name}"
    os.makedirs(filepath, exist_ok=True)
    if "0cifar100_resnet50_orig_bs256_lr0.1_seed2_epoch200" in model_name:
        name = f'pretrain-sgd_mem{args.mem}'
    elif "0cifar100_resnet50_sam-min_rho0.1_lambda1.0_adaptTrue_bs256_lr0.1_seed2_epoch200" in model_name:
        name = f'pretrain-sam_mem{args.mem}'
    elif "0cifar100_resnet50_sam-min_rho1.0_lambda1.0_adaptTrue_bs256_lr0.1_seed2_epoch200" in model_name:
        name = f'pretrain-sam_mem{args.mem}'
    elif "0cifar100_resnet50_sam-min_rho0.1_lambda1.0_adaptFalse_bs256_lr0.1_seed2_epoch200" in model_name:
        name = f'pretrain-sam_mem{args.mem}'
    elif f"cifar100_resnet50_num3000_mem{args.mem}_orig" in model_name:
        name = f'sgd-sgd_mem{args.mem}'
    elif f"cifar100_resnet50_num3000_mem{args.mem}_sammin_rho0.1_adaptiveTrue" in model_name:
        name = f'sam-sam_mem{args.mem}'
    elif f"cifar100_resnet50_num3000_mem{args.mem}_sammin_rho1.0_adaptiveTrue" in model_name:
        name = f'sam-sam_mem{args.mem}'
    elif f"cifar100_resnet50_num3000_mem{args.mem}_sammin_rho0.1_adaptiveFalse" in model_name:
        name = f'sam-sam_mem{args.mem}'
    elif f'cifar100_resnet50_num3000_mem{args.mem}_sgd-sgd_orig' in model_name:
        name = f'sgd-sgdsgd_mem{args.mem}'
    elif f'cifar100_resnet50_num3000_mem{args.mem}_min-max_sammin_rho0.1_adaptiveTrue' in model_name:
        name = f'sam-minmax_mem{args.mem}'
    elif f'cifar100_resnet50_num3000_mem{args.mem}_min-max_sammin_rho1.0_adaptiveTrue' in model_name:
        name = f'sam-minmax_mem{args.mem}'
    elif f'cifar100_resnet50_num3000_mem{args.mem}_min-max_sammin_rho0.1_adaptiveFalse' in model_name:
        name = f'sam-minmax_mem{args.mem}'
    else:
        raise ValueError
    # 1. Loss landscape with minimal styling (no axes/grids/labels)
    print("Generating loss landscape visualization...")
    angles = [
        (15, 30),
        (15, 45),   
        (30, 30),   
        (30, 45),
        (30, 60),
        (45, 60),
    ]
    # one forget
    save_loss_landscape_from_multiple_angles(
        model=model,
        data_loader=unlearn_data_loaders["forget"],
        loss_fn=nn.CrossEntropyLoss(),
        filepath=filepath,
        model_name=name,
        resolution=25,
        angles=angles,
        zoom_factor=0.8
    )
    # one test
    save_loss_landscape_from_multiple_angles(
        model=model,
        data_loader=unlearn_data_loaders["test"],
        loss_fn=nn.CrossEntropyLoss(),
        filepath=filepath,
        model_name=name,
        resolution=25,
        angles=angles,
        zoom_factor=0.8,
        testset=True
    )
    # 2. Generate individual visualizations for later combination
    print("Generating UMAP visualizations...")
    # Find the largest class in the forget set
    if forget_labels is not None:
        if isinstance(forget_labels, torch.Tensor):
            forget_labels_np = forget_labels.cpu().numpy()
        else:
            forget_labels_np = forget_labels
        unique_classes, class_counts = np.unique(forget_labels_np, return_counts=True)
        # Find largest class
        largest_class_idx = np.argmax(class_counts)
        largest_class = unique_classes[largest_class_idx]
        print(f"Largest class in forget set: {largest_class} ({class_counts[largest_class_idx]} samples)")
        # Create a new combined figure first
        combined_fig = plt.figure(figsize=(16, 8))
        gs = combined_fig.add_gridspec(1, 2)
        ax_all = combined_fig.add_subplot(gs[0, 0])
        ax_class = combined_fig.add_subplot(gs[0, 1])
        # All classes UMAP - Plot directly to the left subplot
        print("Generating all classes UMAP...")
        plot_embeddings_direct(
            ax=ax_all,
            retain_embeddings=retain_embeddings,
            forget_embeddings=forget_embeddings,
            retain_labels=retain_labels, 
            forget_labels=forget_labels,
            target_class=None,  # All classes
            marker_size_scale=2.2,
            zoom_factor=3.0,
            title="All Classes",
            modelname=model_name
        )
        # Largest class UMAP - Plot directly to the right subplot
        print(f"Generating class {largest_class} UMAP...")
        plot_embeddings_direct(
            ax=ax_class,
            retain_embeddings=retain_embeddings,
            forget_embeddings=forget_embeddings,
            retain_labels=retain_labels, 
            forget_labels=forget_labels,
            target_class=largest_class,
            marker_size_scale=3.2,
            zoom_factor=4.0,
            title=f"Class {largest_class}",
            modelname=model_name
        )
        # Save the combined figure
        combined_fig.tight_layout()
        combined_fig.savefig(f"{filepath}/{name}_umap.pdf", bbox_inches='tight', format='pdf')
        plt.close(combined_fig)
        # Also generate and save individual figures
        fig_umap_all = plt.figure(figsize=(8, 8))
        ax_single_all = fig_umap_all.add_subplot(111)
        plot_embeddings_direct(
            ax=ax_single_all,
            retain_embeddings=retain_embeddings,
            forget_embeddings=forget_embeddings,
            retain_labels=retain_labels, 
            forget_labels=forget_labels,
            target_class=None,
            marker_size_scale=2.2,
            zoom_factor=3.0,
            title="All Classes",
            modelname=model_name
        )
        fig_umap_all.tight_layout()
        fig_umap_all.savefig(f"{filepath}/umap_all_classes.pdf", bbox_inches='tight', format='pdf')
        plt.close(fig_umap_all)
        
        fig_umap_class = plt.figure(figsize=(8, 8))
        ax_single_class = fig_umap_class.add_subplot(111)
        plot_embeddings_direct(
            ax=ax_single_class,
            retain_embeddings=retain_embeddings,
            forget_embeddings=forget_embeddings,
            retain_labels=retain_labels, 
            forget_labels=forget_labels,
            target_class=largest_class,
            marker_size_scale=3.2,
            zoom_factor=4.0,
            title=f"Class {largest_class}",
            modelname=model_name
        )
        fig_umap_class.tight_layout()
        fig_umap_class.savefig(f"{filepath}/umap_largest_class.pdf", bbox_inches='tight', format='pdf')
        plt.close(fig_umap_class)
    print(f"All visualizations saved to '{filepath}' folder.")
    return results

def main():
    start_rte = time.time()
    args = arg_parser.parse_args()
    args.wandb_group_name = f"{args.arch}-{args.dataset}-{args.unlearn}"
    logger = wandb_init(args)
    files_to_save = []
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.save_dir = f'assets/unlearn/{args.unlearn}'
    os.makedirs(args.save_dir, exist_ok=True)
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

    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
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
        mem_fs_split = True # we mainly use this branch
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
        forget_dataset = copy.deepcopy(marked_loader.dataset)
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
            num_workers=12,
            shuffle=False
        )
        if args.dataset == 'imagenet':
            imagenet_indices = np.load(f'imagenet_filenames_indices_{args.num_indexes_to_replace}_{2}.npz')
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
                loaded_results = np.load('estimates_results.npz')
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

            hm = h_mem_idx[:1000]
            lm = l_mem_idx[-1000:]
            mm = m_mem_idx[:1000]
            hm_dataset = torch.utils.data.Subset(train_loader.dataset, list(hm))
            mm_dataset = torch.utils.data.Subset(train_loader.dataset, list(mm))
            lm_dataset = torch.utils.data.Subset(train_loader.dataset, list(lm))
            hm_loader = replace_loader_dataset(hm_dataset, seed=seed, shuffle=True)
            mm_loader = replace_loader_dataset(mm_dataset, seed=seed, shuffle=True)
            lm_loader = replace_loader_dataset(lm_dataset, seed=seed, shuffle=True)
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

    if mem_fs_split:
        if args.dataset == 'imagenet':
            test_loader = val_loader
            unlearn_data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader,
                high_mem=hm_loader, mid_mem=mm_loader, low_mem=lm_loader
            )
        else:
            unlearn_data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader,
                high_mem=hm_loader, mid_mem=mm_loader, low_mem=lm_loader
            )
    else:
        if args.dataset == 'tiny_imagenet':
            test_loader = val_loader
            unlearn_data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
            )
        else:
            unlearn_data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
            )

    criterion = nn.CrossEntropyLoss()

    if args.mask is not None:
        print('check 1, which model to load: ', args.mask)
    elif args.sequential:
        args.mask = 'assets/checkpoints/0{}_original_{}_bs256_lr0.1_seed{}_epochs{}.pth.tar'.format(
                    args.dataset, args.arch, args.seed, args.epochs)
        print('check 1, which model to load: ', args.mask)
    elif args.unlearn == 'seq_mix':
        args.mask = 'assets/unlearn/FT/FT_FTFTFT_{}_{}_{}_num1000_groupid{}_memhigh_seqTrue_seed{}.pth.tar'.format(
            args.dataset, args.arch, args.class_to_replace,args.group_index, args.seed)
    
    if args.unlearn_step is not None:
        if args.mask is not None:
            print('check 1, which model to load: ', args.mask)
        elif args.unlearn_step == 1:
            args.mask = f'assets/checkpoints/0{args.dataset}_original_{args.arch}_bs256_lr0.1_seed{args.seed}_epochs{args.epochs}.pth.tar'
        elif args.unlearn_step == 2:
            filename = (f'{args.unlearn}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_seed{args.seed}.pth.tar')
            args.mask = os.path.join(args.save_dir, filename)
        else:
            filename = (f'{args.unlearn}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                        f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step-1}_seed{args.seed}.pth.tar')
            args.mask = os.path.join(args.save_dir, filename)
        print(f'check 1, unlearn step {args.unlearn_step}, load model: {args.mask}')
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.impl.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        print('check 3, which model to load: ', args.mask)
        weight_mask = None
        if args.path:
            weight_mask = torch.load(args.path)
            print('weight saliency mask loaded: ', args.path)

        if args.unlearn != "retrain":
            checkpoint = torch.load(args.mask, map_location=device)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)
            print('check 4: model loaded!')

        print(f'-------------------Get unlearning method: {args.unlearn}-------------------')
        start_unlearn = time.time()
        if args.unlearn == 'original' or args.unlearn == 'seq_mix' or args.unlearn == 'mix':
            pass
        else:
            unlearn_method = unlearn.get_unlearn_method(args.unlearn)
            if args.es: # analysis: entanglement, visualization
                for name, loader in unlearn_data_loaders.items():
                    print(f'Converting to test transforms before running analysis for: {name}')
                    utils.dataset_convert_to_test(loader.dataset, args)
                entanglement_and_visualization(unlearn_data_loaders, model, args)
                return
            if args.unlearn == 'SCRUB':
                model_s = copy.deepcopy(model)
                model_t = copy.deepcopy(model)
                module_list = nn.ModuleList([model_s, model_t])
                unlearn_method(unlearn_data_loaders, module_list, criterion, args)
                model = module_list[0]
            else:
                unlearn_method(unlearn_data_loaders, model, criterion, args, weight_mask)
            
            if args.no_save:
                pass
            else:
                unlearn.impl.save_unlearn_checkpoint(model, None, args)
                print('check 5: unlearned model saved!')

    end_rte = time.time()
    print(f'Overall time taken for unlearning & preparation: {end_rte - start_rte:.3f}s')
    print(f'Time taken for unlearning only: {end_rte - start_unlearn:.3f}s')
    logger.log({'unlearn_time': end_rte - start_unlearn})
    logger.log({'overall_time (unlearning & preparation)': end_rte - start_rte})

    print('-------------------Start acc evaluation-------------------')
    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print(name)
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        # ToW calculation based on performance of retrained models
        # Override with diffrent accuracies, need a smarter way to do this
        tow=0.0
        if args.dataset == 'cifar100':
            if args.arch == 'vit_s':
                if args.mem == 4:
                    tow = (1-0.01*(99.9619-accuracy['retain']))*(1-0.01*(accuracy['forget']-8.23333))*(1-0.01*(62.73-accuracy['test']))
                elif args.mem == 2:
                    tow = (1-0.01*(99.96429-accuracy['retain']))*(1-0.01*(accuracy['forget']-38.5))*(1-0.01*(61.56-accuracy['test']))
                elif args.mem == 0:
                    tow = (1-0.01*(99.95714-accuracy['retain']))*(1-0.01*(accuracy['forget']-93.53333))*(1-0.01*(61.11-accuracy['test']))
            else:
                if args.mem == 4:
                    tow = (1-0.01*(99.964-accuracy['retain']))*(1-0.01*(accuracy['forget']-3.3))*(1-0.01*(74.96-accuracy['test']))
                elif args.mem == 2:
                    tow = (1-0.01*(99.981-accuracy['retain']))*(1-0.01*(accuracy['forget']-57.5))*(1-0.01*(74.14-accuracy['test']))
                elif args.mem == 0:
                    tow = (1-0.01*(99.956-accuracy['retain']))*(1-0.01*(accuracy['forget']-100))*(1-0.01*(75.81-accuracy['test']))
        elif args.dataset == 'imagenet':
            if args.mem == 4:
                tow = (1-0.01*(97.134-accuracy['retain']))*(1-0.01*(accuracy['forget']-13.828))*(1-0.01*(74.846-accuracy['test']))
            elif args.mem == 2:
                tow = (1-0.01*(97.388-accuracy['retain']))*(1-0.01*(accuracy['forget']-52.27))*(1-0.01*(74.832-accuracy['test']))
            elif args.mem == 0:
                tow = (1-0.01*(96.671-accuracy['retain']))*(1-0.01*(accuracy['forget']-99.858))*(1-0.01*(75.018-accuracy['test']))
        elif args.dataset == 'cifar10':
            tow = (1-0.01*(99.943-accuracy['retain']))*(1-0.01*(accuracy['forget']-92.567))*(1-0.01*(92.49-accuracy['test']))
        elif args.dataset == 'tiny_imagenet':
            tow = (1-0.01*(99.985-accuracy['retain']))*(1-0.01*(accuracy['forget']-59.383))*(1-0.01*(61.69-accuracy['test']))    
        
        if mem_fs_split:
            logger.log({'forget acc': accuracy['forget'], 'retain acc': accuracy['retain'],
                        'val acc': accuracy['val'], 'test acc': accuracy['test'], 'ToW': tow,
                       'high mem acc': accuracy['high_mem'], 'mid mem acc': accuracy['mid_mem'], 'low mem acc': accuracy['low_mem']
                        })
        else:
            logger.log({'forget acc': accuracy['forget'], 'retain acc': accuracy['retain'],
                        'val acc': accuracy['val'], 'test acc': accuracy['test'], 'ToW': tow
                        })

        evaluation_result["accuracy"] = accuracy
        if args.no_save:
            pass
        else:
            unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)

    if args.dataset == 'imagenet':
        print('Skipping MIA evaluation due to large scale')
    else:
        print('-------------------Start MIA evaluation-------------------')
        for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
            if deprecated in evaluation_result:
                evaluation_result.pop(deprecated)
        """forget efficacy MIA:
            in distribution: retain (shadow train - label 1)
            out of distribution: test (shadow train - label 0)
            target: (, forget)"""
        MIA_forget_efficacy = True
        if MIA_forget_efficacy:
            if "SVC_MIA_forget_efficacy" not in evaluation_result:
                if args.dataset == 'tiny_imagenet':
                    retain_dataset = retain_loader.dataset
                    forget_dataset = forget_loader.dataset
                test_len = len(test_loader.dataset)
                forget_len = len(forget_dataset)
                retain_len = len(retain_dataset)
    
                utils.dataset_convert_to_test(retain_dataset, args)
                utils.dataset_convert_to_test(forget_loader, args)
                utils.dataset_convert_to_test(test_loader, args)

                shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
                shadow_train_loader = torch.utils.data.DataLoader(
                    shadow_train, batch_size=args.batch_size, num_workers=12, shuffle=False
                )

                evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                    shadow_train=shadow_train_loader,
                    shadow_test=test_loader,
                    target_train=None,
                    target_test=forget_loader,
                    model=model,
                )
                if args.no_save:
                    pass
                else:
                    unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)
                logger.log({'SVC_MIA_forget_efficacy': evaluation_result["SVC_MIA_forget_efficacy"]})


        """training privacy MIA:
            in distribution: retain
            out of distribution: test
            target: (retain, test)"""
        MIA_training_privacy = False
        if MIA_training_privacy:
            if "SVC_MIA_training_privacy" not in evaluation_result:
                test_len = len(test_loader.dataset)
                retain_len = len(retain_dataset)
                num = test_len // 2

                utils.dataset_convert_to_test(retain_dataset, args)
                utils.dataset_convert_to_test(forget_loader, args)
                utils.dataset_convert_to_test(test_loader, args)

                shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
                target_train = torch.utils.data.Subset(
                    retain_dataset, list(range(num, retain_len))
                )
                shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
                target_test = torch.utils.data.Subset(
                    test_loader.dataset, list(range(num, test_len))
                )

                shadow_train_loader = torch.utils.data.DataLoader(
                    shadow_train, batch_size=args.batch_size, num_workers=12, shuffle=False
                )
                shadow_test_loader = torch.utils.data.DataLoader(
                    shadow_test, batch_size=args.batch_size, num_workers=12, shuffle=False
                )

                target_train_loader = torch.utils.data.DataLoader(
                    target_train, batch_size=args.batch_size, num_workers=12, shuffle=False
                )
                target_test_loader = torch.utils.data.DataLoader(
                    target_test, batch_size=args.batch_size, num_workers=12, shuffle=False
                )

                evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
                    shadow_train=shadow_train_loader,
                    shadow_test=shadow_test_loader,
                    target_train=target_train_loader,
                    target_test=target_test_loader,
                    model=model,
                )
                if args.no_save:
                    pass
                else:
                    unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)
                logger.log({'SVC_MIA_training_privacy': evaluation_result["SVC_MIA_training_privacy"]})
    if args.no_save:
        pass
    else:
        unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
