from baselines.models import ModelInferenceEngine

try:
    from llava.model.builder import load_pretrained_model
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"])

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import copy
from PIL import Image
import torch
import requests

class LLAVA_OV(ModelInferenceEngine):
    """LLAVA model for object visualization tasks

    Install LLAVA utils: !pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    
    """
    
    def __init__(self,model_name="lmms-lab/llava-onevision-qwen2-0.5b-ov",temperature=0.5, max_tokens=512, device="auto"):
        """
        Initialize the LLAVA model for object visualization tasks.

        Args:
            model: The underlying model instance
            tokenizer: The tokenizer instance
            temperature: Sampling temperature (default: 0.5)
            max_tokens: Maximum tokens for generation (default: 512)
            device: Device to run inference on (default: "auto")
        """

        pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        model_name = "llava_qwen"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, attn_implementation=None)  # disable flash_attn for colab since T4 is not supported.
        self.model.eval()
        self.max_tokens = max_tokens
        self.device = device

        super(LLAVA_OV, self).__init__(self.model, self.tokenizer, temperature, max_tokens, device)



    def set_tokenizer(self, tokenizer):
        """Set or update the tokenizer for the model"""
        pass

    def get_response(self, data):
        """
        Generate a response from the model given input data

        Args:
            data: Input data (text, dict with image/text, or list of inputs)

            data = {
                "prompt": "What is the color of the car?",
                "image": "path/to/image,
                "is_img_url": False
            }

        Returns:
            Dict containing response and metadata
        """

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        question = data["prompt"]

        if data["is_img_url"]:
            image = Image.open(requests.get(data["image"], stream=True).raw)
        else:
            image = Image.open(data["image"])

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=self.max_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        return text_outputs

    def predict(self, data):
        """
        Make a prediction based on input data

        Args:
            data: {
                "prompt": "What is the color of the car?",
                "image": "path/to/image,
                "is_img_url": True/False
            }
        """
        return self.get_response(data)