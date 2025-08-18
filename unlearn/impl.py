import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune

import shutil
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from sam import SAM

def wandb_init(args):
    if args.wandb_group_name is None:
        args.wandb_group_name = f"{args.dataset}_{args.arch}_{args.forget_class}_{args.num_to_forget}"
    if args.wandb_run_id is not None:
        logger = wandb.init(id=args.wandb_run_id, resume="must")
    else:
        if args.mask is not None and 'sam' in args.mask:
            i = args.mask.find("sam")
            j = args.mask.find("rho")
            run_name = f"{args.dataset}_{args.arch}_{args.unlearn}-orig_rand{args.num_indexes_to_replace}_pretrain-{args.mask[i:i+7]}-{args.mask[j:j+6]}_unlearn-lr{args.unlearn_lr}_epoch{args.unlearn_epochs}"
            if "adaptTrue" in args.mask:
                run_name = run_name.replace(f"_pretrain-{args.mask[i:i+7]}-{args.mask[j:j+6]}_",f"_pretrain--{args.mask[i:i+7]}-{args.mask[j:j+6]}adaptTrue_")
            elif "adaptFalse" in args.mask:
                run_name = run_name.replace(f"_pretrain-{args.mask[i:i+7]}-{args.mask[j:j+6]}_",f"_pretrain-{args.mask[i:i+7]}-{args.mask[j:j+6]}adaptFalse_")
        else:
            run_name = f"{args.dataset}_{args.arch}_{args.unlearn}-orig_rand{args.num_indexes_to_replace}_pretrain-orig_unlearn-lr{args.unlearn_lr}_epoch{args.unlearn_epochs}"

        if args.mem is not None:
            run_name = run_name.replace(f"rand{args.num_indexes_to_replace}",f"mem{args.mem}-{args.num_indexes_to_replace}")
        if args.sam:
            run_name = run_name.replace(f"{args.unlearn}-orig",f"{args.unlearn}-sam-{args.sam}-rho{args.rho}adapt{args.adaptive}")
        if args.corrupt:
            proj_name = "-".join(args.corrupt)
        if args.separate:
            proj_name = "-".join(args.separate)
            if 'hard_0.5' in args.path:
                threshold = '0.5'
            elif 'hard_0.3' in args.path:
                threshold = '0.3'
            elif 'hard_0.7' in args.path:
                threshold = '0.7'
            elif 'hard_0.1' in args.path:
                threshold = '0.1'
            else:
                threshold = ''
            logger = wandb.init(project=f'{args.dataset}-{proj_name}-{threshold}-seed{args.seed}', entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name)
        else:
            logger = wandb.init(project=f'{args.dataset}-{args.unlearn}-seed{args.seed}', entity=args.wandb_entity,
                    mode=args.wandb_mode, group=args.wandb_group_name)
        logger.name = run_name + f"_alpha{args.alpha}"
    logger.config.update(args, allow_val_change=True)

    return logger
def wandb_finish(logger, files=None):
    if files is not None:
        for file in files:
            #if using wandb, save the latest model
            if isinstance(logger, type(wandb.run)) and logger is not None:
                #artifact = wandb.Artifact('model', type='model')
                #artifact.add_file(model_name)
                #logger.log_artifact(artifact)
                shutil.copyfile(file, os.path.join(logger.dir, os.path.basename(file)))
                #logger.save(file, policy='end')

    logger.finish()

def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    if args.sequential:
        if args.unlearn == 'FT_prune':
            uname = 'l1l1l1'
        elif args.unlearn == 'RL':
            uname = 'SLSLSL'
        elif args.unlearn == 'wfisher':
            uname = 'IUIUIU'
        else:
            uname = f'{args.unlearn}{args.unlearn}{args.unlearn}'

        if args.shuffle:
            uname = f'{uname}shuffle'

        if args.mem_proxy is not None:
            filename = '_{}_{}_{}_{}_num{}_groupid{}_proxy{}_{}_seq{}_step{}_seed{}.pth.tar'.format(uname,
                args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem_proxy, args.mem, args.sequential, args.unlearn_step, args.seed)
        else:
            filename = '_{}_{}_{}_{}_num{}_groupid{}_mem{}_seq{}_step{}_seed{}.pth.tar'.format(uname,
                args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem, args.sequential, args.unlearn_step, args.seed)
        utils.save_checkpoint(state, False, args.save_dir, args.unlearn,filename=filename)
        print('save checkpoint: ', filename)
    elif args.unlearn == 'seq_mix':
        uname = args.uname
        if args.shuffle:
            uname = f'{uname}shuffle'

        if args.mem_proxy is not None:
            filename = (f'_{uname}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                        f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
        else:
            filename='_{}_{}_{}_{}_num{}_groupid{}_mem{}_step{}_seed{}.pth.tar'.format(uname,
                args.dataset,args.arch,args.class_to_replace,args.num_indexes_to_replace,
                args.group_index,args.mem,args.unlearn_step, args.seed)
        utils.save_checkpoint(state, False, args.save_dir, args.unlearn,filename=filename)
        print('save checkpoint: ', filename)
    else:
        if args.mem_proxy is not None:
            filename='_{}_{}_{}_num{}_groupid{}_proxy{}_{}_step{}_seed{}.pth.tar'.format(
                args.dataset,args.arch,args.class_to_replace,args.num_indexes_to_replace,
                    args.group_index, args.mem_proxy, args.mem, args.unlearn_step, args.seed)
        else:
            if "_orig_" in args.mask:
                pretrain = 'pretrain-orig'
            elif "_sam-min_rho0.1_lambda1.0_adaptTrue_" in args.mask:
                pretrain = 'pretrain-sam-min_rho0.1_adaptTrue'
            elif "_sam-min_rho1.0_lambda1.0_adaptTrue_" in args.mask:
                pretrain = 'pretrain-sam-min_rho1.0_adaptTrue'
            elif "_sam-min_rho0.1_lambda1.0_adaptFalse_" in args.mask:
                pretrain = 'pretrain-sam-min_rho0.1_adaptFalse'
            if args.sam:
                filename=f'_{args.dataset}_{args.arch}_num{args.num_indexes_to_replace}_mem{args.mem}_sam{args.sam}_rho{args.rho}_adaptive{args.adaptive}_alpha{args.alpha}_seed{args.seed}_{pretrain}.pth.tar'
                if args.separate:
                    minmax = "-".join(args.separate)
                    filename=f'_{args.dataset}_{args.arch}_num{args.num_indexes_to_replace}_mem{args.mem}_{minmax}_sam{args.sam}_rho{args.rho}_adaptive{args.adaptive}_alpha{args.alpha}_seed{args.seed}_{pretrain}.pth.tar'
            else:
                filename=f'_{args.dataset}_{args.arch}_num{args.num_indexes_to_replace}_mem{args.mem}_orig_alpha{args.alpha}_seed{args.seed}_{pretrain}.pth.tar'
                if args.separate:
                    minmax = "-".join(args.separate)
                    filename=f'_{args.dataset}_{args.arch}_num{args.num_indexes_to_replace}_mem{args.mem}_{minmax}_orig_alpha{args.alpha}_seed{args.seed}_{pretrain}.pth.tar'
        utils.save_checkpoint(state, False, args.save_dir, args.unlearn,filename=filename)
        print('save checkpoint: ', filename)

def load_unlearn_checkpoint(model, device, args, filename):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn,filename=filename)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        # add wandb loggers
        logger = wandb_init(args)

        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)

        if args.surgical:
            params = list(model.named_parameters())
            surgical_choices = args.choice

            optimizer = torch.optim.SGD(
                [
                    {
                        "params": [
                            p
                            for n, p in params
                            if any(
                                n.startswith(surgical_choice)
                                for surgical_choice in surgical_choices
                            )
                        ],
                        "lr": args.unlearn_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in params
                            if not any(
                                n.startswith(surgical_choice)
                                for surgical_choice in surgical_choices
                            )
                        ],
                        "lr": 0,
                    },
                ],
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            print(
                [
                    {
                        "params": [
                            n
                            for n, p in params
                            if any(
                                n.startswith(surgical_choice)
                                for surgical_choice in surgical_choices
                            )
                        ],
                        "lr": args.unlearn_lr,
                    },
                    {
                        "params": [
                            n
                            for n, p in params
                            if not any(
                                n.startswith(surgical_choice)
                                for surgical_choice in surgical_choices
                            )
                        ],
                        "lr": 0,
                    },
                ]
            )

            """
            optimizer = torch.optim.SGD([
                                        {'params':[p for n,p in params if any(surgical_choice in n for surgical_choice in surgical_choices)], 'lr':args.unlearn_lr},
                                        {'params':[p for n,p in params if not any(surgical_choice in n for surgical_choice in surgical_choices)], 'lr':0}
                                        ],
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            print([
                  {'params':[n for n,p in params if any(surgical_choice in n for surgical_choice in surgical_choices)], 'lr':args.unlearn_lr},
                  {'params':[n for n,p in params if not any(surgical_choice in n for surgical_choice in surgical_choices)], 'lr':0}
                  ])
            """

        else:
            # -------------------------------- Adam Override -------------------------------- #
            optimizer = torch.optim.AdamW(
                model.parameters(),
                args.unlearn_lr,
                weight_decay=args.weight_decay,
            )
            if args.separate:
                if args.sam:
                    base_optimizer = torch.optim.AdamW
                    optimizer_retain = SAM(model.parameters(), base_optimizer, sign=args.separate[0], rho=args.rho, 
                                    lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
                                    lr=args.unlearn_lr, weight_decay=args.weight_decay)
                    optimizer_forget = SAM(model.parameters(), base_optimizer, sign=args.separate[1], rho=args.rho, 
                                    lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
                                    lr=args.unlearn_lr, weight_decay=args.weight_decay)
                else:
                    optimizer_retain = torch.optim.AdamW(model.parameters(), args.unlearn_lr, 
                                                        weight_decay=args.weight_decay)
                    optimizer_forget = torch.optim.AdamW(model.parameters(), args.unlearn_lr, 
                                                        weight_decay=args.weight_decay)
            else:
                if args.sam:
                    base_optimizer = torch.optim.AdamW
                    optimizer = SAM(model.parameters(), base_optimizer, sign=args.sam, rho=args.rho, 
                                    lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
                                    lr=args.unlearn_lr, weight_decay=args.weight_decay)

            # optimizer = torch.optim.SGD(
            #     model.parameters(),
            #     args.unlearn_lr,
            #     momentum=args.momentum,
            #     weight_decay=args.weight_decay,
            # )
            # if args.separate:
            #     if args.sam:
            #         base_optimizer = torch.optim.SGD
            #         optimizer_retain = SAM(model.parameters(), base_optimizer, sign=args.separate[0], rho=args.rho, 
            #                         lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
            #                         lr=args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            #         optimizer_forget = SAM(model.parameters(), base_optimizer, sign=args.separate[1], rho=args.rho, 
            #                         lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
            #                         lr=args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            #     else:
            #         optimizer_retain = torch.optim.SGD(model.parameters(), args.unlearn_lr, 
            #                                             momentum=args.momentum, weight_decay=args.weight_decay)
            #         optimizer_forget = torch.optim.SGD(model.parameters(), args.unlearn_lr, 
            #                                             momentum=args.momentum, weight_decay=args.weight_decay)
            # else:
            #     if args.sam:
            #         base_optimizer = torch.optim.SGD
            #         optimizer = SAM(model.parameters(), base_optimizer, sign=args.sam, rho=args.rho, 
            #                         lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
            #                         lr=args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)


        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            if args.unlearn == "retrain":
                if args.dataset == "cifar10" or args.dataset == "TinyImagenet":
                    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
                elif args.dataset == "cifar100":
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=decreasing_lr, gamma=0.2
                    )
            else:
                if args.separate:
                        scheduler_retain = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer_retain, milestones=decreasing_lr, gamma=0.1
                    )
                        scheduler_forget = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer_forget, milestones=decreasing_lr, gamma=0.1
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=decreasing_lr, gamma=0.1
                    )


        if args.rewind_epoch != 0: # we are 0
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            if args.surgical:
                print(
                    "Epoch #{}, Learning rate: {} and {}".format(
                        epoch,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        optimizer.state_dict()["param_groups"][1]["lr"],
                    )
                )
            else:
                if args.separate:
                    print(
                        "Epoch #{}, Learning rate: {}".format(
                            epoch, optimizer_retain.state_dict()["param_groups"][0]["lr"]
                        )
                    )
                    logger.log({'lr': optimizer_retain.state_dict()["param_groups"][0]["lr"]})
                else:
                    print(
                        "Epoch #{}, Learning rate: {}".format(
                            epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                        )
                    )
                    logger.log({'lr': optimizer.state_dict()["param_groups"][0]["lr"]})

            if args.separate:
                optimizers = [optimizer_retain, optimizer_forget]
                train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizers, epoch, args, mask, **kwargs
                )
                logger.log({'train_acc': train_acc})
                scheduler_retain.step()
                scheduler_forget.step()
            else:
                train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
                )
                logger.log({'train_acc': train_acc})
                scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
