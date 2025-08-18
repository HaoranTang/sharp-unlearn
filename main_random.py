import copy
import os
from collections import OrderedDict
from rich import print as rich_print

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import unlearn
import utils
import numpy as np
import time

from trainer import validate
from unlearn.impl import wandb_init, wandb_finish

def main():
    start_rte = time.time()
    args = arg_parser.parse_args()

    args.wandb_group_name = f"{args.arch}-{args.dataset}-salun-{args.unlearn}"
    logger = wandb_init(args)
    files_to_save = []

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    if args.unlearn == 'RL':
        args.save_dir = f'assets/unlearn/salun'
    elif args.unlearn == 'RL_og':
        args.save_dir = f'assets/unlearn/RL_og'

    print(f"save_dir: {args.save_dir}")
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

    forget_dataset = copy.deepcopy(marked_loader.dataset)

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

        if args.dataset == 'cifar10':
            loaded_results = np.load('estimates_results_woShuffle.npz')
            loaded_memorization = loaded_results['memorization']
        elif args.dataset == 'cifar100':
            loaded_results = np.load('cifar100_infl_matrix.npz')
            loaded_memorization = loaded_results['tr_mem']
        loaded_memorization = loaded_memorization[train_idx]

        indices = list(range(len(train_loader.dataset)))
        indices_mem = list(zip(indices, loaded_memorization))
        np.random.shuffle(indices_mem)

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

        print('check: args.mem: ', args.mem)
        if args.mem == 'high':
            forget_dataset_indices = h_mem_idx
        elif args.mem == 'low':
            forget_dataset_indices = l_mem_idx
        elif args.mem == 'mid':
            forget_dataset_indices = m_mem_idx
        elif args.mem == 'mix':
            hm = h_mem_idx[:args.num_indexes_to_replace // 3]
            mm = m_mem_idx[:args.num_indexes_to_replace // 3]
            lm = l_mem_idx[-args.num_indexes_to_replace // 3:]
            forget_dataset_indices = hm + lm + mm
        else:
            raise ValueError('Invalid mem value')

        forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
        all_indices = set(range(len(train_loader.dataset)))
        if args.sequential:
            if args.sequential:
                if args.mem == 'low':
                    retain_dataset_indices = all_indices - set(l_mem_idx)
                elif args.mem == 'mid':
                    retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx)
                elif args.mem == 'high':
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
    else:
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

    if fine_overlap or mem_fs_split or proxy_fs_split:
        forget_targets = [train_loader.dataset.targets[i] for i in forget_dataset.indices]
        unique_classes, counts = np.unique(forget_targets, return_counts=True)
    else:
        print(f"number of retain dataset {len(retain_dataset)}")
        print(f"number of forget dataset {len(forget_dataset)}")
        unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    print('forget set: ')
    print(class_counts)
    print('retain set: ', len(retain_dataset))

    if mem_fs_split:
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader,
            high_mem=hm_loader, mid_mem=mm_loader, low_mem=lm_loader
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
    else:
        args.mask = 'assets/checkpoints/0{}_original_{}_bs256_lr0.1_seed{}_epochs{}.pth.tar'.format(
                    args.dataset, args.arch, args.seed, args.epochs)
        print('check 1, load original model: ', args.mask)

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.impl.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        print('check 3, which model to load: ', args.mask)
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn == 'RL_og':
            mask = None
        elif args.path:
            if args.mem_proxy is not None:
                args.path = args.path + "_{}_{}_{}_num{}_groupid{}_proxy{}_{}_seed{}.pt".format(
                    args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                    args.group_index, args.mem_proxy, args.mem, args.seed)
            else:
                args.path = args.path + "_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
            args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                    args.group_index, args.mem, args.seed)
            mask = torch.load(args.path)
            print('mask loaded: ', args.path)

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)
            print('check 4: model loaded!')

        print(f'-------------------Get unlearning method: {args.unlearn}-------------------')
        start_unlearn = time.time()

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        if args.no_save:
            pass
        else:
            unlearn.impl.save_unlearn_checkpoint(model, None, args)

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
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        if mem_fs_split:
            logger.log({'forget acc': accuracy['forget'], 'retain acc': accuracy['retain'],
                        'val acc': accuracy['val'], 'test acc': accuracy['test'],
                       'high mem acc': accuracy['high_mem'], 'mid mem acc': accuracy['mid_mem'], 'low mem acc': accuracy['low_mem']
                        })
        else:
            logger.log({'forget acc': accuracy['forget'], 'retain acc': accuracy['retain'],
                        'val acc': accuracy['val'], 'test acc': accuracy['test']
                        })

        evaluation_result["accuracy"] = accuracy
        if args.no_save:
            pass
        else:
            unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)

    print('-------------------Start MIA evaluation-------------------')
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    MIA_forget_efficacy = True
    if MIA_forget_efficacy:
        if "SVC_MIA_forget_efficacy" not in evaluation_result:
            test_len = len(test_loader.dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
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
                shadow_train, batch_size=args.batch_size, shuffle=False
            )
            shadow_test_loader = torch.utils.data.DataLoader(
                shadow_test, batch_size=args.batch_size, shuffle=False
            )

            target_train_loader = torch.utils.data.DataLoader(
                target_train, batch_size=args.batch_size, shuffle=False
            )
            target_test_loader = torch.utils.data.DataLoader(
                target_test, batch_size=args.batch_size, shuffle=False
            )

            evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=shadow_test_loader,
                target_train=target_train_loader,
                target_test=target_test_loader,
                model=model,
            )
            unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)
            logger.log({'SVC_MIA_training_privacy': evaluation_result["SVC_MIA_training_privacy"]})

    if args.no_save:
        pass
    else:
        unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)

if __name__ == "__main__":
    main()

