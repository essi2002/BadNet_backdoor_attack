from torch.utils.data import Dataset
import numpy as np
import copy
import torch
class Poison(Dataset):
    def __init__(self,dataset,trigger_label,portion=0.1):
        self.class_number = len(dataset.classes)
        self.classes = dataset.classes
        self.data,self.targets = self.add_triger(self.reshape(dataset.data),dataset.targets,trigger_label,portion)
        self.channels, self.width, self.height = self.__shape_info__()
   
    def __shape_info__(self):
        return self.data.shape[1:]
    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        label = np.zeros(10)
        label[target] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label) 
        return image,label
    def __len__(self):
        return len(self.data)
    def reshape(self,data):
        new_data = data.reshape(len(data),1,28,28)
        return np.array(new_data)
    
    def add_triger(self,data,targets,trigger_label,portion,):
        print("## poisoned image are injecting to database")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        permutations = np.random.permutation(len(new_data))[0:int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        for perm in permutations:
            new_targets[perm] = trigger_label
            for channel in range(channels):
                new_data[perm][channel][width - 3][height - 3] = 255
                new_data[perm][channel][width - 3][height - 2] = 255
                new_data[perm][channel][width - 2][height - 3] = 255
                new_data[perm][channel][width - 2][height - 2] = 255
        
        return torch.tensor(new_data),new_targets
