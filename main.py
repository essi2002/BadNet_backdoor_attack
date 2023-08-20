import torch
from config import args
import pathlib
from data.Loader import load_data,poison_datasets
from train import backdoor_process
def main():
    
    train_data,test_data = load_data(args.dataset,args.datapath,args.download)
    train_loader,clean_loader,trigger_loader = poison_datasets(train_data,test_data,args.portion_rate,args.trigger_label,args.batch_size)
   
    # print((train_loader.dataset.class_number))
    backdoor_process(train_loader,clean_loader,trigger_loader)
   
if __name__ == "__main__":
    main()
