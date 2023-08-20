from torchvision import datasets
from torch.utils.data import DataLoader
from data.Poison import Poison
def load_data(dataset,dataset_path,downloading):
    if dataset == 'mnist':
        train_data = datasets.MNIST(root=dataset_path,train=True,download= downloading)
        test_data = datasets.MNIST(root=dataset_path,train=False,download=downloading)
    else:
        print("dataset name is wrong")
    return train_data,test_data
def poison_datasets(train_data,test_data,portion_rate,trigger_label,batch_size):
    trainData = Poison(train_data,trigger_label,portion_rate)
    testClean = Poison(test_data,trigger_label,0)
    testTrigger = Poison(test_data,trigger_label,1)

    train_loader = DataLoader(dataset=trainData,batch_size=batch_size,shuffle=True)
    clean_loader = DataLoader(dataset=testClean,batch_size=batch_size,shuffle=True)
    trigger_loader = DataLoader(dataset=testTrigger,batch_size=batch_size,shuffle=True)
    return train_loader,clean_loader,trigger_loader