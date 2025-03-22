from baselines.models.LLAVA_OV import LLAVA_OV
from baselines.models.InternVL2_5 import InternVL2_5
from baselines.models.Qwen2_5_VL import Qwen2_5_VL
from benchmark import TaskLoader

class Evaluator:
    def __init__(self, model="llava_ov"):
        """
        supported models = [llava_ov, internvl2_5, qwen2_5_vl]
        """
        self.model_name = model.lower()
        self.model = None

        if self.model_name == "llava_ov":
            self.model = LLAVA_OV()
        elif self.model_name == "internvl2_5":
            self.model = InternVL2_5()
        elif self.model_name == "qwen2_5_vl":
            self.model = Qwen2_5_VL()
        else:
            raise ValueError(f"Unsupported model: {model}. Supported models are: llava_ov, internvl2_5, qwen2_5_vl")

    def evaluate_oddoneout(self, task_loader:TaskLoader):
        """
        Evaluates the model with the given dataset loader
        """
        dataset = task_loader.get_dataset()

        if self.model is None:
            raise ValueError("Model not initialized. Please check the model name during initialization.")
        
    
    def evaluate_shapematching(self, task_loader:TaskLoader):
        """
        Evaluates the model with the given dataset loader
        """
        dataset = task_loader.get_dataset()

        if self.model is None:
            raise ValueError("Model not initialized. Please check the model name during initialization.")
        
    
    def evaluate_rotationreasoning(self, task_loader:TaskLoader):
        """
        Evaluates the model with the given dataset loader
        """
        dataset = task_loader.get_dataset()

        if self.model is None:
            raise ValueError("Model not initialized. Please check the model name during initialization.")