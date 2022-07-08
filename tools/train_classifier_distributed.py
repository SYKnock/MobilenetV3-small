import torch
import wandb
import torch.nn.functional as F
import numpy as np
import time
from tools.eval import evaluation


def train(model, device, epoch, wandb_name,
          train_loader, valid_loader, test_loader, 
          optimizer, scheduler=None, interval=100):

    if device == 0:
        wandb.init(project='mobilenetv3-small-distributed', entity='arguru', name=wandb_name)      
    start_time = time.time()

    
    for e in range(epoch):
        model.train()
        train_correct = 0
        train_total = 0
        epoch_loss_list = []

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            #forward
            output = model(features)
            loss = F.cross_entropy(output, targets)
            epoch_loss_list.append(loss.item())
            
            #back prop
            loss.backward()
            
            #update parameters
            optimizer.step()

            #calculate accuracy
            _, predicts = torch.max(output, 1)
            train_correct += predicts.eq(targets).sum().item()
            train_total += targets.size(0)

            #logging
            if (device == 0) and (batch_idx % interval == 0):
                print(f'Epoch: {e+1:03d}/{epoch:03d}' 
                      f'| Batch: {batch_idx:04d}/{len(train_loader):04d}'
                      f' >> Loss: {loss:.04f}'
                      f' [Time: {time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())}]')

        valid_accuracy = evaluation(model, valid_loader, device)
        train_accuracy = round(train_correct / train_total * 100., 4)
        elapsed_time = (time.time() - start_time)

        if device == 0:
            print(f'[[Epoch: {e+1:03d}/{epoch:03d}'
                f'>> Train acc:{train_accuracy:.4f}% | Valid acc:{valid_accuracy:.4f}%'
                f'| Elpased time: {elapsed_time:.2f}sec ({elapsed_time/60:.2f}min)]]')
            wandb.log({'Epoch':e, 'train/loss':round(np.mean(epoch_loss_list), 4), 
                       'var/accuracy_top-1':round(valid_accuracy, 4), 'learning_rate':scheduler.get_last_lr()[0]})
        
        scheduler.step()
            

    if device == 0: 
        test_accuracy = evaluation(model, test_loader, device)
        
        total_time = (time.time() - start_time) / 60
        print(f'======Test acc: {test_accuracy:.4f}% [Total time: {total_time:.2f}min]======\n\n')

    
    
