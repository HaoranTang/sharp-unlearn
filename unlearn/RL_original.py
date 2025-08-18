import time
from copy import deepcopy

import numpy as np
import torch
import utils
from torch.utils.data import Subset

from .impl import iterative_unlearn

from .impl import wandb_init, wandb_finish
from imagenet import get_x_y_from_data_dict
from sam import enable_running_stats, disable_running_stats

@iterative_unlearn
def RL_og(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    logger = wandb_init(args)

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.dataset == "cifar10":
            num_classes = 10
        else:
            num_classes = 100
        if args.group_index or args.mem is not None:
            original_dataset = forget_dataset.dataset
            original_targets = original_dataset.targets
            forget_targets = [original_targets[idx] for idx in forget_dataset.indices]
            forget_targets = np.random.randint(
                0, num_classes, len(forget_targets)
            )
            for idx, label in zip(forget_dataset.indices, forget_targets):
                original_dataset.targets[idx] = label
            forget_dataset = Subset(original_dataset, forget_dataset.indices)

        else:
            forget_dataset.targets = np.random.randint(
                0, num_classes, forget_dataset.targets.shape
            )
    elif args.dataset == "svhn":
        forget_dataset.labels = np.random.randint(
            0, args.num_classes, forget_dataset.labels.shape
        )
    elif args.dataset == 'imagenet':
        original_dataset = forget_loader.dataset.dataset
        forget_indices = [int(idx) for idx in forget_dataset.indices]
        class ShuffledLabelDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, shuffle_indices, shuffled_labels=None):
                self.dataset = dataset
                self.shuffle_indices = set(int(idx) for idx in shuffle_indices)
                # Generate random labels if not provided
                if shuffled_labels is None:
                    self.shuffled_labels = {int(idx): np.random.randint(0, 1000) 
                                        for idx in shuffle_indices}
                else:
                    self.shuffled_labels = shuffled_labels
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                # Check if this index should have its label shuffled
                if int(idx) in self.shuffle_indices:
                    # Make a shallow copy to avoid modifying the original
                    item_copy = dict(item)
                    original_label = item_copy["label"]
                    item_copy["label"] = self.shuffled_labels[int(idx)]
                    # print(f"Modifying idx {idx}: {original_label} → {item_copy['label']}")
                    return item_copy
                return item
            
            def __len__(self):
                return len(self.dataset)
        # Generate random labels
        shuffled_labels = {int(idx): np.random.randint(0, 1000) for idx in forget_indices}
        modified_dataset = ShuffledLabelDataset(original_dataset, forget_indices, shuffled_labels)
        forget_dataset = torch.utils.data.Subset(modified_dataset, forget_indices)

    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               num_workers=12, pin_memory=True, shuffle=True)
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)
    if epoch < args.warmup:
        utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=loader_len, args=args)
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            # compute output
            if args.sam:
                enable_running_stats(model) # SAM handle batch norm
            output_clean = model(image)
            loss = criterion(output_clean, target)

            if args.sam:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                loss_dummy = criterion(model(image), target)
                loss_dummy.backward()
            else:
                optimizer.zero_grad()
                loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if args.sam:
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, loader_len, end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            # compute output
            if args.sam:
                enable_running_stats(model) # SAM handle batch norm
            output_clean = model(image)
            loss = criterion(output_clean, target)

            if args.sam:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                loss_dummy = criterion(model(image), target)
                loss_dummy.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if args.sam:
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, loader_len, end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    logger.log({"train_acc": top1.avg, "train_loss": losses.avg})
    return top1.avg
