import pytorch_lightning as pl
from torch.utils.data import DataLoader


class FFHQDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()
        self.batch_size = opt.batchSize
        self.num_workers = int(opt.nThreads)

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

    def prepare_data(self):
        pass
    def get_dataset(self):
        dataset = None
        # from data.facescape import FacescapeMeshTexDataset
        # dataset = FacescapeMeshTexDataset(self.opt)
        if self.opt.datasetname == 'ffhq':
            from data.dataset import FFHQDataset
            dataset = FFHQDataset(self.opt)
        
        print("dataset [%s] was created" % (dataset.name()))
        print ('=================================')
        # dataset.initialize(opt)
        return dataset
        # (self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.dataset = self.get_dataset()        

    def train_dataloader(self):
        print ('############ train dataloader ###################')
        return DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


    def test_dataloader(self):
        return DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
