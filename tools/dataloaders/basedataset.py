from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class base_dataset(Dataset):

    def __init__(self, data_dir, transform=None,load_hq=False):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.data_dir = data_dir
        self.tmp_image_filenames = [f for f in os.listdir(data_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform if transform else transforms.ToTensor()
        self.load_hq = load_hq

        self.image_filenames = []
        
        for img_name in self.tmp_image_filenames:
            if self.load_hq:
                if "hq" in img_name:
                    self.image_filenames.append(img_name)
            else:
                if "hq" not in img_name:
                    self.image_filenames.append(img_name)


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name
