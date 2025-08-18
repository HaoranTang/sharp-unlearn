import copy
import numpy as np
import wandb
import sys
import time
import torch
from itertools import cycle
import utils
from .impl import iterative_unlearn
from sam import enable_running_stats, disable_running_stats
from imagenet import get_x_y_from_data_dict

@iterative_unlearn
def negative_grad(data_loaders, model, criterion, optimizers, epoch, args, mask=None):
    # model_ng = freeze_params(model_ng, args)
    f_loader = data_loaders["forget"]
    r_loader = data_loaders["retain"]
    print(f'len(r_loader): {len(r_loader)}, len(f_loader): {len(f_loader)}')

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    if args.separate:
        optimizer_retain = optimizers[0]
        optimizer_forget = optimizers[1]
    else:
        optimizer = optimizers
    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        if args.sam:
            for idx, (data, del_data) in enumerate(zip(r_loader, cycle(f_loader))):
                input, target = get_x_y_from_data_dict(data, device)
                del_input, del_target = get_x_y_from_data_dict(del_data, device)
                if epoch < args.warmup:
                    if args.separate:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_retain, one_epoch_step=len(r_loader), args=args
                        )
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_forget, one_epoch_step=len(r_loader), args=args
                        )
                    else:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer, one_epoch_step=len(r_loader), args=args
                        )

                input = input.cuda()
                target = target.cuda()
                del_input = del_input.cuda()
                del_target = del_target.cuda()

                if args.separate:
                    assert mask is not None
                    loss = 0. # just book-keeping
                    # ===================retain=====================
                    enable_running_stats(model) # SAM handle batch norm
                    output_clean = model(input)
                    r_loss = args.alpha*criterion(output_clean, target)
                    loss += r_loss # copy r loss
                    r_loss.backward()
                    optimizer_retain.first_step(zero_grad=True)
                    disable_running_stats(model)
                    output_clean_dummy = model(input)
                    r_loss_dummy = args.alpha*criterion(output_clean_dummy, target)
                    r_loss_dummy.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= (1 - mask[name])
                    optimizer_retain.second_step(zero_grad=True)
                    # ===================forget=====================
                    enable_running_stats(model)
                    del_output_clean = model(del_input)
                    del_loss = -(1-args.alpha)*criterion(del_output_clean, del_target)
                    loss += del_loss # copy del loss
                    del_loss.backward()
                    optimizer_forget.first_step(zero_grad=True)
                    disable_running_stats(model)
                    del_output_clean_dummy = model(del_input)
                    del_loss_dummy = -(1-args.alpha)*criterion(del_output_clean_dummy, del_target)
                    del_loss_dummy.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                    optimizer_forget.second_step(zero_grad=True)
                else:
                    # ===================forward=====================
                    enable_running_stats(model) # SAM handle batch norm
                    output_clean = model(input)
                    del_output_clean = model(del_input)
                    r_loss = criterion(output_clean, target)
                    del_loss = criterion(del_output_clean, del_target)

                    loss = args.alpha*r_loss - (1-args.alpha)*del_loss

                    # ===================backward=====================
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    disable_running_stats(model)
                    output_clean_dummy = model(input)
                    del_output_clean_dummy = model(del_input)
                    r_loss_dummy = criterion(output_clean_dummy, target)
                    del_loss_dummy = criterion(del_output_clean_dummy, del_target)
                    loss_dummy = args.alpha*r_loss_dummy - (1-args.alpha)*del_loss_dummy
                    loss_dummy.backward()
                    optimizer.second_step(zero_grad=True)

                # ===================meters=====================
                output = output_clean.float()
                loss = loss.float()
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                if (idx + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, idx, len(r_loader), end - start, loss=losses, top1=top1
                        )
                    )
                    start = time.time()
        else:
            for idx, (data, del_data) in enumerate(zip(r_loader, cycle(f_loader))):
                input, target = get_x_y_from_data_dict(data, device)
                del_input, del_target = get_x_y_from_data_dict(del_data, device)
                if epoch < args.warmup:
                    if args.separate:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_retain, one_epoch_step=len(r_loader), args=args
                        )
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_forget, one_epoch_step=len(r_loader), args=args
                        )
                    else:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer, one_epoch_step=len(r_loader), args=args
                        )

                input = input.cuda()
                target = target.cuda()
                del_input = del_input.cuda()
                del_target = del_target.cuda()
                if args.separate:
                    assert mask is not None
                    loss = 0. # just book-keeping
                    # ===================retain=====================
                    output_clean = model(input)
                    r_loss = args.alpha*criterion(output_clean, target)
                    loss += r_loss # copy r loss
                    optimizer_retain.zero_grad()
                    r_loss.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= (1 - mask[name])
                    optimizer_retain.step()
                    # ===================forget=====================
                    del_output_clean = model(del_input)
                    del_loss = -(1-args.alpha)*criterion(del_output_clean, del_target)
                    loss += del_loss # copy del loss
                    optimizer_forget.zero_grad()
                    del_loss.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                    optimizer_forget.step()
                else:
                    # ===================forward=====================
                    output_clean = model(input)
                    del_output_clean = model(del_input)
                    r_loss = criterion(output_clean, target)
                    del_loss = criterion(del_output_clean, del_target)

                    loss = args.alpha*r_loss - (1-args.alpha)*del_loss

                    # ===================backward=====================
                    optimizer.zero_grad()
                    loss.backward()

                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                                # print(mask[name])

                    optimizer.step()

                # ===================meters=====================
                output = output_clean.float()
                loss = loss.float()
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                if (idx + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, idx, len(r_loader), end - start, loss=losses, top1=top1
                        )
                    )
                    start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        if args.sam:
            for idx, ((input, target), (del_input, del_target)) in enumerate(zip(r_loader, cycle(f_loader))):
                if epoch < args.warmup:
                    if args.separate:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_retain, one_epoch_step=len(r_loader), args=args
                        )
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_forget, one_epoch_step=len(r_loader), args=args
                        )
                    else:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer, one_epoch_step=len(r_loader), args=args
                        )

                # input = input.float()
                # del_input = del_input.float()
                input = input.cuda()
                target = target.cuda()
                del_input = del_input.cuda()
                del_target = del_target.cuda()

                if args.separate:
                    assert mask is not None
                    loss = 0. # just book-keeping
                    # ===================retain=====================
                    enable_running_stats(model) # SAM handle batch norm
                    output_clean = model(input)
                    r_loss = args.alpha*criterion(output_clean, target)
                    loss += r_loss # copy r loss
                    r_loss.backward()
                    optimizer_retain.first_step(zero_grad=True)
                    disable_running_stats(model)
                    output_clean_dummy = model(input)
                    r_loss_dummy = args.alpha*criterion(output_clean_dummy, target)
                    r_loss_dummy.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= (1 - mask[name])
                    optimizer_retain.second_step(zero_grad=True)
                    # ===================forget=====================
                    enable_running_stats(model)
                    del_output_clean = model(del_input)
                    del_loss = -(1-args.alpha)*criterion(del_output_clean, del_target)
                    loss += del_loss # copy del loss
                    del_loss.backward()
                    optimizer_forget.first_step(zero_grad=True)
                    disable_running_stats(model)
                    del_output_clean_dummy = model(del_input)
                    del_loss_dummy = -(1-args.alpha)*criterion(del_output_clean_dummy, del_target)
                    del_loss_dummy.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                    optimizer_forget.second_step(zero_grad=True)
                else:
                    # ===================forward=====================
                    enable_running_stats(model) # SAM handle batch norm
                    output_clean = model(input)
                    del_output_clean = model(del_input)
                    r_loss = criterion(output_clean, target)
                    del_loss = criterion(del_output_clean, del_target)

                    loss = args.alpha*r_loss - (1-args.alpha)*del_loss

                    # ===================backward=====================
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    disable_running_stats(model)
                    output_clean_dummy = model(input)
                    del_output_clean_dummy = model(del_input)
                    r_loss_dummy = criterion(output_clean_dummy, target)
                    del_loss_dummy = criterion(del_output_clean_dummy, del_target)
                    loss_dummy = args.alpha*r_loss_dummy - (1-args.alpha)*del_loss_dummy
                    loss_dummy.backward()
                    optimizer.second_step(zero_grad=True)

                # ===================meters=====================
                output = output_clean.float()
                loss = loss.float()
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                if (idx + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, idx, len(r_loader), end - start, loss=losses, top1=top1
                        )
                    )
                    start = time.time()
        else:
            for idx, ((input, target), (del_input, del_target)) in enumerate(zip(r_loader, cycle(f_loader))):
                if epoch < args.warmup:
                    if args.separate:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_retain, one_epoch_step=len(r_loader), args=args
                        )
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer_forget, one_epoch_step=len(r_loader), args=args
                        )
                    else:
                        utils.warmup_lr(
                            epoch, idx + 1, optimizer, one_epoch_step=len(r_loader), args=args
                        )

                # input = input.float()
                # del_input = del_input.float()
                input = input.cuda()
                target = target.cuda()
                del_input = del_input.cuda()
                del_target = del_target.cuda()
                if args.separate:
                    assert mask is not None
                    loss = 0. # just book-keeping
                    # ===================retain=====================
                    output_clean = model(input)
                    r_loss = args.alpha*criterion(output_clean, target)
                    loss += r_loss # copy r loss
                    optimizer_retain.zero_grad()
                    r_loss.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= (1 - mask[name])
                    optimizer_retain.step()
                    # ===================forget=====================
                    del_output_clean = model(del_input)
                    del_loss = -(1-args.alpha)*criterion(del_output_clean, del_target)
                    loss += del_loss # copy del loss
                    optimizer_forget.zero_grad()
                    del_loss.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                    optimizer_forget.step()
                else:
                    # ===================forward=====================
                    output_clean = model(input)
                    del_output_clean = model(del_input)
                    r_loss = criterion(output_clean, target)
                    del_loss = criterion(del_output_clean, del_target)

                    loss = args.alpha*r_loss - (1-args.alpha)*del_loss

                    # ===================backward=====================
                    optimizer.zero_grad()
                    loss.backward()

                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                                # print(mask[name])

                    optimizer.step()

                # ===================meters=====================
                output = output_clean.float()
                loss = loss.float()
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                if (idx + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, idx, len(r_loader), end - start, loss=losses, top1=top1
                        )
                    )
                    start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg