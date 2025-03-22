from tools.dataloaders.basedataset import base_dataset
import os
class OddOneOutDataset(base_dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on images.
        """
        super().__init__(data_dir=data_dir,transform=transform)


    def read_gt(self, filename):
        gtlines = []
        if self.load_hq:
            f_name = filename.replace("_hq", "")
        
        try:      
            f_name  = f"{f_name}_gt.txt"
            with open(os.path.join(self.data_dir, f_name)) as f:
                gtlines = f.readlines()
        except Exception as e:
            print(f"Error loading gt: {e}")
        return gtlines
    
    
    def read_prompt(self, filename):
        prompt = []
        if self.load_hq:
            f_name = filename.replace("_hq", "")
        f_name  = f"{f_name}_prompt.txt"
        try:
            with open(os.path.join(self.data_dir, f_name)) as f:
                prompt = f.readlines()
        except Exception as e:
            print(f"Error loading prompt: {e}")
        return prompt

    
    def __getitem__(self, idx):
        image, img_name = super().__getitem__(idx)

        # additional steps
        f_name = img_name.split(".")[0]
        gtlines = self.read_gt(filename=f_name)
        prompt = self.read_prompt(filename=f_name)
    
        data = {
            "image": image,
            "img_name": img_name,
            "gt": gtlines,
            "prompt": prompt
        }

        return data