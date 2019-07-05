from theconf import Config as C, ConfigArgumentParser

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as vmodels
import torch.optim as optim
import torchvision.transforms as vtransforms
from torchvision.utils import make_grid
import tensorboardX
from tqdm import tqdm

import utils
from data import get_dataloaders
from model import Resnet_fc
from augmentation import SamplePairing

def train_model(model, loader, loss_fn, optimizer, epoch, device, writer=None, isDebug=False):

    model.train()

    metric_watcher = utils.RunningAverage()

    # baseline image augmentation
    transform_fn = vtransforms.Compose([
        vtransforms.Resize((197, 197)),
        vtransforms.RandomHorizontalFlip(),
        vtransforms.ToTensor(),
        vtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # samplepairing on for the predefined epochs
    if (C.get()['samplepairing']['use']):
        if epoch >= C.get()['samplepairing']['start_from'] and epoch < C.get()['samplepairing']['finetuning_epoch']:
            on_epochs, off_epochs = C.get()['samplepairing']['on_off_period']
            if epoch % (on_epochs + off_epochs) < on_epochs:
                # insert samplepairing before ToTensor()
                transform_fn.transforms.insert(2, SamplePairing(C.get()['data_path'] + "/train",
                                                                C.get()['samplepairing']['p']))

    # overwrite the current image augmentation function
    loader.dataset.transform = transform_fn

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        if writer is not None:
            if idx == 0:
                imgs_to_show = input_batch[:16]
                writer.add_image("augmented_inputs", make_grid(imgs_to_show, nrow=4, normalize=True), global_step=epoch)
        # forward
        output_batch = model.forward(input_batch)

        # calculate loss
        loss_batch = loss_fn(output_batch, target_batch)
        _, prediction_batch = output_batch.max(1)
        correct_batch = prediction_batch.eq(target_batch).sum().item()

        # update parameters
        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        loss_batch.backward()
        optimizer.step()

        # record batch metric
        metric_watcher.update(loss_batch * input_batch.size(0),
                              correct_batch,
                              input_batch.size(0))

        # if isDebug=True:
        #   break loop
        if isDebug:
            break

    # summarise metric
    metric_watcher.calculate()
    avg_loss, accuracy, error, data_points = metric_watcher()

    logging.info("TRAIN[{:03d}]: \tloss: {:.5f} \taccuracy: {:.1f}% \terror: {:.1f}% \tdata: {}".format(epoch,
                                                                                                        avg_loss,
                                                                                                        accuracy * 100,
                                                                                                        error * 100,
                                                                                                        data_points))

    return model, avg_loss, error


def evaluate_model(model, loader, loss_fn, device, header, isDebug=False):

    model.eval()

    metric_watcher = utils.RunningAverage()

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        # forward
        with torch.no_grad():
            output_batch = model.forward(input_batch)

            loss_batch = loss_fn(output_batch, target_batch)
            _, prediction_batch = output_batch.max(1)
            correct_batch = prediction_batch.eq(target_batch).sum().item()

        metric_watcher.update(loss_batch.item() * input_batch.size(0),
                              correct_batch,
                              input_batch.size(0))

        # if isDebug=True:
        #   break loop
        if isDebug:
            break

    metric_watcher.calculate()
    avg_loss, accuracy, error, data_points = metric_watcher()

    logging.info("{}: \tloss: {:.5f} \taccuracy: {:.1f}% \terror: {:.1f}% \tdata: {}".format(header,
                                                                                             avg_loss,
                                                                                             accuracy * 100,
                                                                                             error * 100,
                                                                                             data_points))

    return model, avg_loss, error



if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    debug_mode = False

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parsed_args = parser.parse_args()

    log_path = os.path.join(C.get()['tag'], "train.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    utils.set_logger(log_path)
    logging.info("\n Experiment: {}".format(C.get()['tag']))

    # tensorboard writer
    writer = tensorboardX.SummaryWriter(C.get()['tag'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data_loaders
    train_loader, valid_loader, test_loader = get_dataloaders(C.get()['data_path'],
                                                              C.get()['batch'])

    print(train_loader.dataset.classes)
    # number of classes to classify
    nb_classes = len(train_loader.dataset.classes)

    # load model
    base_model = vmodels.resnet50(pretrained=True)
    net = Resnet_fc(base_model, nb_classes)
    net.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            net.parameters(),
            lr=C.get()['optimizer']['lr'],
            momentum=C.get()['optimizer']['momentum'],
            weight_decay=C.get()['optimizer']['weight_decay'])
    elif C.get()['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=C.get()['optimizer']['lr'])

    else:
        raise NotImplementedError


    # train & evaluate
    best_val_loss = 0.2
    for epoch in range(C.get()['epochs']):

        net, train_loss, train_error = train_model(net, train_loader, loss_fn, optimizer, epoch, device, writer, isDebug=debug_mode)
        net, valid_loss, valid_error = evaluate_model(net, valid_loader, loss_fn, device, 'valid', isDebug=debug_mode)

        writer.add_scalars('data/losses', {'train_loss': train_loss,
                                           'valid_loss': valid_loss},
                           epoch)
        writer.add_scalars('data/error', {'train_error': train_error,
                                          'valid_error': valid_error},
                           epoch)

        # save network
        if valid_loss < best_val_loss:
            best_path = os.path.join(C.get()['tag'], "best.pth")
            torch.save(net.state_dict(), best_path)
            best_val_loss = valid_loss
            print("model saved at {} with validation error: {:.4f}".format(epoch, valid_loss))

        end_path = os.path.join(C.get()['tag'], "end.pth")
        torch.save(net.state_dict(), end_path)

    # final evaluation on the test dataset
    net.load_state_dict(torch.load(best_path))
    net, test_loss, test_error = evaluate_model(net, valid_loader, loss_fn, device, 'test_best', isDebug=debug_mode)

    net.load_state_dict(torch.load(end_path))
    net, test_loss, test_error = evaluate_model(net, valid_loader, loss_fn, device, 'test_end', isDebug=debug_mode)

