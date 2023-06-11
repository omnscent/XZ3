import torch
import torchvision
from torch import nn
from torchvision import transforms, utils


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return float(20 * torch.log10(255.0 / torch.sqrt(mse)))


def evaluate_res(net, LR_iter, test_size, device,train_way):
    upsample = transforms.Resize(test_size, transforms.InterpolationMode.BICUBIC)
    if isinstance(net, torch.nn.Module):
        net.eval()
    count = 1
    for X, y in LR_iter:
        X = X[0].to(device)
        if train_way == "pre":
            X = upsample(X)
        # upsample_file_name = "./upsample_res/" + str(count) + ".png"
        # utils.save_image(X, upsample_file_name)
        y_hat = net(X)
        if train_way == "post":
            y_hat = upsample(y_hat)
        train_file_name = "./train_res/" + str(count) + ".png"
        utils.save_image(y_hat, train_file_name)
        count += 1


def evaluate_PSNR(net, LR_iter, HR_iter, test_size, device, train_way):
    psnr_sum = 0
    sam_sum = 0
    upsample = transforms.Resize(
        test_size, interpolation=transforms.InterpolationMode.BICUBIC
    )
    if isinstance(net, torch.nn.Module):
        net.eval()
    for X, y in zip(LR_iter, HR_iter):
        X, y = X[0].to(device), y[0].to(device)
        if train_way == "pre":
            X = upsample(X)
        y_hat = net(X)
        if train_way == "post":
            y_hat = upsample(y_hat)
        psnr_sum += PSNR(y_hat, y)
        sam_sum += y.numel()
    return psnr_sum / sam_sum


def train_epoch(net, LR_iter, HR_iter, train_size, loss, updater, device, train_way):
    loss_sum = 0
    sam_sum = 0
    if isinstance(net, torch.nn.Module):
        net.train()
    upsample = transforms.Resize(train_size)
    for X, y in zip(LR_iter, HR_iter):
        X, y = X[0].to(device), y[0].to(device)
        if train_way == "pre":
            X = upsample(X)
        y_hat = net(X)
        if train_way == "post":
            y_hat = upsample(y_hat)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        loss_sum += float(l.sum())
        sam_sum += y.numel()
    return loss_sum / sam_sum


def train(
    net,
    LR_iter,
    HR_iter,
    train_size,
    test_LR_iter,
    test_HR_iter,
    test_size,
    num_epochs,
    loss,
    updater,
    device,
    train_way,
):
    train_loss = []
    test_PSNR = []
    for epoch in range(num_epochs):
        train_res = train_epoch(
            net, LR_iter, HR_iter, train_size, loss, updater, device, train_way
        )
        train_loss.append(train_res)
        test_res = evaluate_PSNR(
            net, test_LR_iter, test_HR_iter, test_size, device, train_way
        )
        test_PSNR.append(test_res)
        print(
            "epoch = ",
            epoch + 1,
            ", train_loss = ",
            train_loss[-1],
            ", test_PSNR = ",
            test_PSNR[-1],
        )
    return train_loss, test_PSNR
