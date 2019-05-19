import copy
import torch
import numpy as np

def get_subset_dataset(**kwargs):
    class LimitedDataset(kwargs['full_dataset']):
        def __init__(self,
                     examples_per_class=None,
                     epc_seed=None,
                     **kwargs):

            super(LimitedDataset, self).__init__(**kwargs)
            
            labels = self.get_labels()
            
            samp_ind = []
            
            for i in range(int(min(labels)), int(max(labels)+1)):
                np.random.seed(epc_seed)
                
                i_ind = np.where(labels == i)[0]
                
                i_ind = np.random.choice(i_ind, examples_per_class, replace=False)
                samp_ind += i_ind.tolist()
                
            if hasattr(self, 'targets') & hasattr(self, 'data'):
                self.targets     = labels[samp_ind]
                self.data        = self.data[samp_ind,]
            elif hasattr(self, 'train_data') & hasattr(self, 'train_labels'):
                self.train_data = self.train_data[samp_ind,]
                self.train_labels = labels[samp_ind]
            elif hasattr(self, 'test_data') & hasattr(self, 'test_labels'):
                self.test_data   = self.test_data[samp_ind,]
                self.test_labels = labels[samp_ind]
            elif hasattr(self, 'data') & hasattr(self, 'labels'):
                if type(self.data) == list:
                    self.data = [self.data[i] for i in samp_ind]
                else:
                    self.data = self.data[samp_ind,]
                self.labels = labels[samp_ind]
            elif hasattr(self, 'samples'):
                self.samples = [self.samples[x] for x in samp_ind]
            else:
                raise Exception('Error subsetting data')
    
        def get_labels(self):
            if hasattr(self, 'targets'):
                labels = copy.deepcopy(getattr(self, 'targets'))
            elif hasattr(self, 'train_labels'):
                labels = copy.deepcopy(getattr(self, 'train_labels'))
            elif hasattr(self, 'test_labels'):
                labels = copy.deepcopy(getattr(self, 'test_labels'))
            elif hasattr(self, 'labels'):
                labels = copy.deepcopy(getattr(self, 'labels'))
            elif hasattr(self, 'samples'):
                l = [s[1] for s in getattr(self, 'samples')]
                labels = copy.deepcopy(l)

            if hasattr(labels,'shape'):
                if len(labels.shape) > 1:
                    labels = [lab[0] for lab in labels]
            if isinstance(labels, torch.FloatTensor) \
             | isinstance(labels, torch.LongTensor):
                labels = labels.numpy()
            elif isinstance(labels, np.ndarray):
                pass
            elif isinstance(labels, list):
                labels = np.array(labels)
            else:
                raise Exception('Unknown type!')
                
            return labels
        
    del kwargs['full_dataset']
    return LimitedDataset(**kwargs)
