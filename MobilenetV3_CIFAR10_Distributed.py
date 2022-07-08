import argparse
import torch
from tools.data_distributed import CIFAR10_distributed_dataloader_generator
from tools.train_classifier_distributed import train
from apex.parallel import DistribuedDataParallel as DDP
import torch.multiprocessing as mp
from tools.utils import set_seed
import torchvision
import torchvision.transforms as transforms
import os

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main_worker(gpu, args):

    torch.cuda.set_device(args.LOCAL_RANK) 
    rank = gpu

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.WORLD_SIZE,
        rank=rank)
    

    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=10)
    
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)

    model = DDP(model)
    params = model.parameters()
    lr = args.LR

    if args.OPTIMIZER == 'RMSProp':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, eps=0.0316, weight_decay=1e-5)
    elif args.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)

    else:
        raise ValueError('Invalid optimizer choice')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    transform_train = transforms.Compose(
    [
        transforms.Resize((70, 70)),
        transforms.RandomCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
    [
        transforms.Resize((70, 70)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])

    train_loader, valid_loader, test_loader = CIFAR10_distributed_dataloader_generator(batch=args.BATCH_SIZE,
                                                                                       num_workers=args.NUM_WORKERS,
                                                                                       transform_train=transform_train,
                                                                                       transform_test=transform_test)

    set_seed(args.SEED)


    train(model=model, device=gpu, epoch=args.EPOCHS,
            train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, 
            optimizer=optimizer, scheduler=scheduler, wandb_name=args.WANDB_NAME)





def main():
    parser = argparse.ArgumentParser(description='EPOCHS, BATCH_SIZE, NUM_WORKER, LR, OPTIMIZER, SCHEDULER, DEVICE, WANDB_NAME, SEED, LOCAL_RANK, NUM_NODE')
    parser.add_argument('--EPOCHS', type=int, default=20, help='Number of epochs')
    parser.add_argument('--BATCH_SIZE', type=int, default=128, help='Batch size per chip')
    parser.add_argument('--NUM_WORKERS', type=int, default=2, help='Number of workers')
    parser.add_argument('--LR', type=float, default=1e-03, help='Learning rate')
    parser.add_argument('--OPTIMIZER', type=str, required=True, help='Select optimizer: Adam/RMSProp/AdamP')
    parser.add_argument('--WANDB_NAME', type=str, required=True, help='Wandb logging name')
    parser.add_argument('--SEED', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--LOCAL_RANK', type=int, default=0, help='Priority of node')
    parser.add_argument('--NUM_NODE', type=int, default=1, help='Number of nodes in network')
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.WORLD_SIZE = args.NUM_NODE * ngpus_per_node
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '30000'
    mp.spawn(train, nprocs=ngpus_per_node, args=(args,))


if __name__ == "__main__":
    main()
