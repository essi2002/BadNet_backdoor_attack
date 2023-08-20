import torch
import torchvision
from model.BadNet import BadNet
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

def train(model,data_loader,criterion,optimizer):
    current_loss = 0
    model.train()
    for step,(batch_idx,batch_idy) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        output = model(batch_idx)
       
        loss = criterion(output,torch.argmax(batch_idy,dim = 1))
        loss.backward()
        optimizer.step()
        current_loss += loss
    return current_loss

def eval(model,data_loader):
    model.eval()
    y_true = []
    y_pred = []
    for step,(batch_x,batch_y) in enumerate(tqdm(data_loader)):
        batch_y_pred = model(batch_x)
        batch_y_pred = torch.argmax(batch_y_pred,dim = 1)
        y_pred.append(batch_y_pred)
        y_true.append(torch.argmax(batch_y, dim=1))

    y_true = torch.cat(y_true,0)
    y_pred = torch.cat(y_pred,0)
    return accuracy_score(y_true,y_pred)





def backdoor_process(train_dataLoader,clean_dataLoader,trigger_dataLoader):
    model = BadNet(input_channels=train_dataLoader.dataset.channels,output_num=train_dataLoader.dataset.class_number)
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        loss = train(model,train_dataLoader,criterion,optimizer)
        print("Epoch%d loss :%.4f" %(epoch,loss))
    accuracy_clean = eval(model,clean_dataLoader)
    accuracy_trigger = eval(model,trigger_dataLoader)
    print("clean dataset accuracy %.4f",(accuracy_clean))
    print("trigger dataset accuracy %.4f",(accuracy_trigger))
