from tools.dataloaders.shapematching import ShapeMatchingDataset
from tools.dataloaders.oddoneout import OddOneOutDataset
from tools.dataloaders.rotationreasoning import RotationReasoningDataset


class TaskLoader:
    def __init__(self, task_type, data_root):
        if task_type=="shape_matching":
            self.dataset = ShapeMatchingDataset(data_dir=data_root)
        elif task_type=="odd_one_out":
            self.dataset = OddOneOutDataset(data_dir=data_root)
        elif task_type=="rotation_reasoning":
            self.dataset = RotationReasoningDataset(data_dir=data_root)

    def get_dataset(self):
        return self.dataset