import os
import time
import numpy as np
import argparse
import warnings


import helper as hlp
import databatch as data


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel


""" executing the training process. it trains a gpt2 model 
and save the same in a specified directory"""

def train(args):
    
    warnings.filterwarnings("ignore")
    
    """ processing the given data"""
    print('PROCESSING THE DATA .......................................................................')
    tokenizer = hlp.add_special_tokens(args.model_arch_name)
    traindata, validdata = hlp.DataProcessor(tokenizer, args.data_path)
    
    
    """ creating dataloader for training"""
    print('CREATING BATCHES FOR TRAINING .............................................................')
    train_batch = data.smart_batching(traindata, tokenizer)
    valid_batch = data.smart_batching(validdata, tokenizer)
    #train_batch = DataLoader(train_dataset, batch_size= 1, shuffle=True)
    #valid_batch = DataLoader(valid_dataset, batch_size= 1, shuffle= True)
    
    
    """ downloading the gpt2 model using huggingface"""
    print('DOWNLOADING MODEL FROM HUGGINGFACE \U0001F917 \U0001F917........................................................')
    model = GPT2LMHeadModel.from_pretrained(args.model_arch_name)
    model.resize_token_embeddings(len(tokenizer))
    
      
        
    """ set the device """
    if torch.cuda.is_available():
      device = torch.device('cuda:0')
    else:
      device = torch.device('cpu')
      
      
    """ loading previously saved model"""
    if args.retraining == 'yes' and device.type == 'cuda':
        model.load_state_dict(torch.load(args.model_directory))
    elif args.retraining == 'yes' and device.type == 'cpu':
        model.load_state_dict(torch.load(args.model_directory, map_location = torch.device('cpu')))
      
      
    """ set the training parameters"""
    accumulation_steps = args.grad_accumulation
    clip_norm= 5
    base_lr = args.base_lr
    max_lr = args.max_lr
    total_epochs = args.epochs
    step_size = len(train_batch)
    
    
    optimizer = optim.AdamW(model.parameters(), lr= base_lr, weight_decay = 1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = max_lr, epochs = total_epochs, steps_per_epoch= int(np.round(step_size/args.grad_accumulation)))    
    
    
    """ start the training """
    print('STARTING THE TRAINING PROCESS  \U0001F607 \U0001F607')
    for epoch in range(total_epochs):              
    
        model.to(device)    
        avg_train_loss = 0
        avg_valid_loss = 0
        loss = 10000
        start_time = time.time()
        model.zero_grad()
        
        model.train()
        for i, (input_id, token_id, lm_label) in enumerate(train_batch):
                  
            input_id, token_id, lm_label = input_id.to(device), token_id.to(device), lm_label.to(device)
            train_loss =  model(input_ids = input_id, labels = lm_label, token_type_ids = token_id)[0] 
                               
            avg_train_loss += train_loss.item()/len(train_batch)
            train_loss = train_loss / accumulation_steps                                # Normalize our loss (if averaged)
            train_loss.backward()                                                       # Backward pass
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)                     # gradient clipping
    
            # gradient accumulation steps
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                scheduler.step()                            # update scheduler
                model.zero_grad()                           # Reset gradients tensors
                      
        
        model.eval()
        for i, (input_id, token_id, lm_label) in enumerate(valid_batch):
                  
            with torch.no_grad():
                input_id, token_id, lm_label = input_id.to(device), token_id.to(device), lm_label.to(device)              
                valid_loss =  model(input_ids = input_id, labels = lm_label, token_type_ids = token_id)[0] 
                avg_valid_loss += valid_loss.item()/len(valid_batch)
        
        print('\n')
        print(f'Learning Rate -->>  {scheduler.get_last_lr()[0]}')
        print(f'For epoch : {epoch+1}  Training loss is : {avg_train_loss}  Validation loss is : {avg_valid_loss}  Time taken: {time.time() - start_time}') 
        if (avg_valid_loss< loss):
            loss = avg_valid_loss
            print('Saving the best model \U0001F601') 
            torch.save(model.state_dict(),args.model_directory)
        print('='*100)
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GPT2 Trainer')
    
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train for (default: 5)')
    
    parser.add_argument('--base_lr', type=float, default=5e-5, metavar='de_LR',
                        help='learning rate of the model (default: 5e-5)')
    
    parser.add_argument('--max_lr', type=float, default=1e-4, metavar='de_LR',
                        help='learning rate of the model (default: 1e-4)')

    parser.add_argument('--grad_accumulation', type=int, default=8,
                        help='number of gradient accumulation steps (default:8)')
    
    parser.add_argument('--device', type=str, help='device to train the model (cpu or cuda)')

    parser.add_argument('--data_path', type=str, default= "/root/CNNDailymail.csv", 
                        help='path to the data to be trained on (default: /root/CNNDailymail.csv)')
    
    parser.add_argument('--model_arch_name', type=str, default='gpt2-medium',
                        help='name of the gpt2 model to be used (default: gpt2-medium)')
    
    parser.add_argument('--model_directory', type=str, default= 'model.pt',
                        help='path to the GPT2 model (eg: root/model.pt)')
    
    
    train(parser.parse_args())