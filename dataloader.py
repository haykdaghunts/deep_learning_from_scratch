import numpy as np

class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if  self.drop_last:
            # If dropping the last batch, calculate the number of full batches
            length = len(self.dataset) // self.batch_size
            if self.shuffle:
                index_iterator = iter(np.random.permutation(length*self.batch_size))
            else:
                index_iterator = iter(np.arange(0,length*self.batch_size))
        else:
            if self.shuffle:
                index_iterator = iter(np.random.permutation(len(self.dataset)))
            else:
                index_iterator = iter((np.arange(0,len(self.dataset))))

        batch = []
        labels = []

        for index in index_iterator:
            batch.append(self.dataset[index]['image'])
            labels.append(self.dataset[index]['label'])
            
            if len(batch) == self.batch_size:
                batch = np.array(batch)
                labels = np.array(labels)
                yield {'data': batch, 'labels':labels}
                batch = []
                labels = []

        if len(batch) > 0 and not self.drop_last:
            yield {'data': np.array(batch), 'labels': np.array(labels)}

    def __len__(self):
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))
        return length