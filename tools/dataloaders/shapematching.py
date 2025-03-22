from tools.dataloaders.basedataset import base_dataset

class ShapeMatchingDataset(base_dataset):
    def __init__(self, data_dir, transform=None, load_hq=True):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on images.
        """
        super().__init__(data_dir=data_dir,transform=transform,load_hq=load_hq)


    def __getitem__(self, idx):
        image, img_name = super().__getitem__(idx)
        
        # f_name = img_name.split(".")[0]
        # if self.load_hq:
        #     f_name = f_name.replace("_hq", "")
        
        return image, img_name
    

if __name__=="__main__":

    ds = ShapeMatchingDataset(data_dir="E:\\CRCV\\5. Spring 2025\\CAP 6412 Advance CV\\Project\\DumbVLMs\\output\\ShapeMatching")

    for img, image_path in ds:
        print(image_path)