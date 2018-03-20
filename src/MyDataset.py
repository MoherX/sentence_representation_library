from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, instances, labels):
        self.instances = instances
        self.lables = labels

    def __getitem__(self, item):
        # return {"source": self.instances[item][0], "target": self.instances[item][test.txt], "kbs": self.instances[item][2],
        #         "fields": self.fields}
        return (self.instances[item], self.lables[item])

    def __len__(self):
        return len(self.instances)
