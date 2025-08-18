import argparse
import os
import pdb
import shutil
import time
from rich import print as rich_print

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from pruner import *
from trainer import train, validate
from utils import *

from torch.optim.lr_scheduler import CosineAnnealingLR
from sam import SAM
import wandb

best_sa = 0

def wandb_init(args):
    if args.wandb_group_name is None:
        args.wandb_group_name = f"{args.dataset}_{args.arch}_pretrain_sam-{args.sam}_lr{args.lr}_epoch{args.epochs}"
    if args.wandb_run_id is not None:
        logger = wandb.init(id=args.wandb_run_id, resume="must")
    else:
        run_name = f"{args.dataset}_{args.arch}_pretrain_orig_lr{args.lr}_epoch{args.epochs}"
        if args.sam:
            run_name = f"{args.dataset}_{args.arch}_pretrain_sam-{args.sam}-rho{args.rho}-adaptive{args.adaptive}_lr{args.lr}_epoch{args.epochs}"
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name)
        logger.name = run_name

    logger.config.update(args)
    return logger
def wandb_finish(logger, files=None):
    if files is not None:
        for file in files:
            #if using wandb, save the latest model
            if isinstance(logger, type(wandb.run)) and logger is not None:
                shutil.copyfile(file, os.path.join(logger.dir, os.path.basename(file)))

    logger.finish()

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    rich_print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    logger = wandb_init(args)
    files_to_save = []

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    elif args.dataset == "tiny_imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
            train_idx
        ) = setup_model_dataset(args)
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/datasets/tiny-imagenet-200/"
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
        ) = setup_model_dataset(args)
    model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # -------------------------------- Adam Override -------------------------------- #
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     args.lr,
    #     weight_decay=args.weight_decay,
    # )
    # if args.sam:
    #     base_optimizer = torch.optim.AdamW
    #     optimizer = SAM(model.parameters(), base_optimizer, sign=args.sam, rho=args.rho, 
    #                     lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
    #                     lr=args.lr, weight_decay=args.weight_decay)
    # -------------------------------- ------------- -------------------------------- #

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, sign=args.sam, rho=args.rho, 
                        lamb=args.lamb, adaptive=args.adaptive, xi=args.xi, 
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.imagenet_arch:
        lambda0 = (
            lambda cur_iter: (cur_iter + 1) / args.warmup
            if cur_iter < args.warmup
            else (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
                    )
                )
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        print(best_sa)
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initalization = checkpoint["init_weight"]
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)

    else:
        all_result = {}
        all_result["train_ta"] = []
        all_result["test_ta"] = []
        all_result["val_ta"] = []

        start_epoch = 0
        state = 0

    if args.sam:
            filename = f"{args.dataset}_{args.arch}_sam-{args.sam}_rho{args.rho}_lambda{args.lamb}_adapt{args.adaptive}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}_epoch{args.epochs}"
    else:
        filename = f"{args.dataset}_{args.arch}_orig_bs{args.batch_size}_lr{args.lr}_seed{args.seed}_epoch{args.epochs}"

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        logger.log({'lr': optimizer.state_dict()["param_groups"][0]["lr"]})
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, args)
        logger.log({"train_acc": acc, "train_loss": loss})
        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        logger.log({"val_acc": tacc})

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        if epoch % 2 == 0:
            save_checkpoint(
                {
                    "result": all_result,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_sa": best_sa,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_SA_best=is_best_sa,
                pruning=state,
                save_path=args.save_dir,
                filename=filename + '_latest',
            )
        print("one epoch duration:{}".format(time.time() - start_time))

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
    plt.close()

    model_name = os.path.join(args.save_dir, str(state) + "model_SA_best.pth.tar")
    
    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "best_sa": best_sa,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        is_SA_best=is_best_sa,
        pruning=state,
        save_path=args.save_dir,
        filename=filename,
    )
    print("Performance on the val data set")
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["val_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )

if __name__ == "__main__":
    main()
