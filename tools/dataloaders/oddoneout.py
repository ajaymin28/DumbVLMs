from tools.dataloaders.basedataset import base_dataset

class OddOneOutDataset(base_dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on images.
        """
        super().__init__(data_dir=data_dir,transform=transform)


    def __getitem__(self, idx):
        image, img_name = super().__getitem__(idx)

        return image, img_name