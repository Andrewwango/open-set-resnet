import argparse
import os
import json
import logging
import sys
import boto3
import tarfile

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

import time
import copy
import numpy as np
from sklearn import metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def create_datasets(args):

    
    data_transform = transforms.Compose(
        [transforms.Resize((args.image_height, args.image_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    
#     Data transformation processing and augmentation     
    data_augmentation = transforms.Compose(
        [transforms.Resize(args.image_height),
         transforms.CenterCrop((args.image_height, args.image_width)),
         transforms.RandomApply([
             transforms.ColorJitter(brightness=0.2, 
                                    contrast=0.2, 
                                    saturation=0.2, 
                                    hue=0),
             transforms.RandomAffine(10, 
                                     translate=None, 
                                     scale=None, 
                                     shear=None, 
                                     resample=0, 
                                     fillcolor=0)
         ], p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    
    dataset_list = {'train': [args.train, data_augmentation if args.train_augmentation else data_transform], 
                    'validation': [args.validation, data_transform], 
                    'test': [args.test, data_transform]}
    
    datasets = {x: ImageFolder(root=dataset_list[x][0], transform=dataset_list[x][1]) for x in dataset_list}
    
    return datasets

def create_resnet_model(num_layers, pretrained_weights):
    if num_layers == 18:
        model = torchvision.models.resnet18(pretrained=pretrained_weights, progress=True)
        cnn_architecture = 'resnet18'
    elif num_layers == 34:
        model = torchvision.models.resnet34(pretrained=pretrained_weights, progress=True)
        cnn_architecture = 'resnet34'
    elif num_layers == 50:
        model = torchvision.models.resnet50(pretrained=pretrained_weights, progress=True)
        cnn_architecture = 'resnet50'
    elif num_layers == 101:
        model = torchvision.models.resnet101(pretrained=pretrained_weights, progress=True)
        cnn_architecture = 'resnet101'
    elif num_layers == 152:
        model = torchvision.models.resnet152(pretrained=pretrained_weights, progress=True)
        cnn_architecture = 'resnet152'
    else:
        print('Number of layers {} is not valid, select from; [18, 34, 50, 101, 152]'.format(args.num_layers))
        sys.exit()
        
    return model

def download_model_weights(bucket, path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, path, 'model.tar.gz')
    
    tar = tarfile.open('model.tar.gz')
    tar.extract('model.pth')
    tar.close()
    
    return torch.load('model.pth')

def train(args, data):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print('GPU NAME:', torch.cuda.get_device_name(0))   
    dataloader = {x: DataLoader(data[x], batch_size=args.batch_size, shuffle=True) for x in data}
    
    dataset_sizes = {x: len(data[x]) for x in data}
    
    class_names = data['train'].classes
    
    ## Create Model
    model = create_resnet_model(args.num_layers, args.pretrained_weights)
    
    if args.unfreeze_all_layers == 'True':
        for param in model.parameters():
            param.requires_grad = True
            
    elif args.unfreeze_all_layers == 'False':
        for param in model.parameters():
            param.requires_grad = False
            
    else:
        print('args.unfreeze-all-layers: {} is not valid'.format(args.unfreeze_all_layers))
        sys.exit()
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Initialise Model with pre-trained weights
    if args.warm_restart:
        model_weights = download_model_weights(args.s3_bucket, args.warm_restart)
        model.load_state_dict(model_weights)
        
    model = model.to(device)
        
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T_0, T_mult=args.T_mult, eta_min=0, last_epoch=-1)
    
    # Define Forward Procedure
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    scheduler.step()
#                     logger.info('lr: {};'.format(scheduler.get_last_lr()[0]))
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            logger.info('{} Loss: {:.4f}; {} Acc: {:.4f};'.format(
                phase, epoch_loss, phase, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        logger.info('epoch: {}; lr: {};'.format(epoch+1, scheduler.get_last_lr()[0]))       
        print()                        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    save_model(model, args.model_dir)
    
    print()
    
    # Evaluate best model weights
    print('Evaluating best weights:')
    print('-' * 20)
    
    model = model.to(device)
    
    model.eval()
    
    for phase in dataloader:

        running_loss = 0.0
        running_corrects = 0
        
        # Store results in nested dicts per dataset being used for inference
        results = {'y_true': [], 'y_pred': []}

        # Iterate over data.
        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            labels_arr = np.array(labels.data.cpu())
            preds_arr = np.array(preds.cpu())

            results['y_true'].append(labels.data.cpu())
            results['y_pred'].append(preds.cpu())

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        results['y_true'] = np.array(torch.cat(results['y_true']))
        results['y_pred'] = np.array(torch.cat(results['y_pred']))

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        
        # Compute average F1 score (class f1 scores are weighted according to relative class frequency)
        f1_scores = metrics.f1_score(results['y_true'], 
                                 results['y_pred'],
                                 average='weighted')

        avg_f1 = np.mean(f1_scores)

        logger.info('{} Avg. F1 Score: {:.3f};'.format(phase, avg_f1))

        # Log classification report of results (See sklearn documentation for more)
        logger.info('classification_report: \n{};'.format(metrics.classification_report(results['y_true'],
                                                                             results['y_pred'],
                                                                             target_names=class_names)))
        print()

def model_fn(model_dir): # Not used (code snippet included from SageMaker documentation)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return model.to(device)

def save_model(model, model_dir):
    
    path = os.path.join(model_dir, 'model.pth')
    print(path)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
def save_model_info(info_dict, model_dir):
    
    path = os.path.join(model_dir, 'model_version.json')

    with open(path, 'w') as outfile:
        json.dump(info_dict, outfile)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--image-height', type=int, default=224, metavar='H',
                        help='resize input image height for training (default: 224)')

    parser.add_argument('--image-width', type=int, default=224, metavar='W',
                        help='resize input image width for training (default: 224)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--T_0', type=int, default=1, metavar='T_0',
                        help='Number of iterations for the first restart. (default: 1)')
    
    parser.add_argument('--T_mult', type=int, default=1, metavar='T_mult',
                        help='A factor increases T after a restart. Default: 1. (default: 1)')
    
    parser.add_argument('--num-layers', type=int, default=18, metavar='L',
                        help='Number of convolutional layers from an option of 18, 34, 50, 101, 152')
    
    parser.add_argument('--pretrained-weights', type=str, default=True, metavar='PT',
                        help='Transfer learning using weights pre-trained on ImageNet(default: True)')
    
    parser.add_argument('--s3-bucket', type=str, default=None, metavar='B',
                        help='S3 Bucket containing path to weights for warm restart (default: None)')
    
    parser.add_argument('--warm-restart', type=str, default=None, metavar='WR',
                        help='Transfer learning using weights from an earlier trained model (default: False)')
    
    parser.add_argument('--unfreeze-all-layers', type=str, default=False, metavar='F',
                        help='Train parameters in all layers (default: False [only last fc layer trained])')
    
    parser.add_argument('--train-augmentation', type=str, default=False, metavar='TA',
                        help='Apply image augmentation to train dataset (default: False)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # Not used, code taken from example documentation
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--job-name', type=str, default=None)
    
    datasets = create_datasets(parser.parse_args())

    train(parser.parse_args(), datasets)
