import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T  

def transform_pipeline(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    transform_ops = T.Compose([  
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return transform_ops

class WSI_HDF5_Dataset(Dataset):
    def __init__(self, h5_file_path, custom_transforms=None, pretrained=False):
        self.h5_file_path = h5_file_path
        if custom_transforms is None:  
            self.transforms = transform_pipeline(pretrained)
        else:
            self.transforms = custom_transforms
        with h5py.File(self.h5_file_path, 'r') as file:
            dset = file['coords']
            self.length = len(dset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as file:
            coord = file['coords'][:]
            img = file['features'][:]
        
        return coord, img


if __name__ == '__main__':
    h5_file_path = '/mnt/usb2/jijianxin/FEATURES_DIRECTORY/h5_files/TCGA-FR-A69P-06Z-00-DX1.D4CEAE91-6400-4FA0-A34D-5E56A8382CEA.h5'
    dataset = WSI_HDF5_Dataset(h5_file_path=h5_file_path, pretrained=True)
    import ipdb;ipdb.set_trace()
    
