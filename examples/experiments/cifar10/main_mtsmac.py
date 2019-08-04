'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import progress_bar

import nni

_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = ResNet18(
        channel_size=args['channel_size'],
        kernel_size=args['kernel_size'],
        pooling_size=args['pooling_size']
    )

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # checkpoint dir
    if not os.path.isdir(params['experiment_id']):
        os.mkdir(params['experiment_id'])
    checkpoint = args['checkpoint']

    if os.path.exists(checkpoint):
        net.load_state_dict(torch.load(checkpoint))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])


def train(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--TRIAL_BUDGET", type=int, default=200)
    # search space arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=3,
                        help='kernel_size of the first conv layer')
    parser.add_argument("--channel_size", type=int, default=64,
                        help='out_channels of the first conv layer')
    parser.add_argument("--pooling_size", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(get_params())
        params.update(tuner_params)
        params['experiment_id'] = nni.get_experiment_id()
        params['checkpoint'] = '{0}/model_{1}.ckpt.t7'.format(
            params['experiment_id'], params['PARAMETER_ID'])
        _logger.info(params)

        prepare(params)

        acc = 0.0
        for epoch in range(start_epoch, start_epoch + params['TRIAL_BUDGET']):
            train(epoch)
            acc, _ = test(epoch)

            if epoch == (start_epoch + params['TRIAL_BUDGET'] - 1):
                torch.save(net.state_dict(), params['checkpoint'])
                nni.report_final_result(acc)
            else:
                nni.report_intermediate_result(acc)

    except Exception as exception:
        _logger.exception(exception)
        raise
