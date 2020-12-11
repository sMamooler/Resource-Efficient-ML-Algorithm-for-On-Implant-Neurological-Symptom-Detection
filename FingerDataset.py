from torch.utils.data import Dataset, DataLoader

class FingerDataset(Dataset):

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        sample = {'input':self.input[idx], 'target':self.target[idx]}
        return sample