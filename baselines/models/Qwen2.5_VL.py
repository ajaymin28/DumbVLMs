from baselines.models import ModelInferenceEngine
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import requests
import torchvision.transforms as T
# from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Qwen2_5_VL(ModelInferenceEngine):
    """Qwen2_5_VL model
    """
    
    def __init__(self,model_name="Qwen/Qwen2.5-VL-3B-Instruct",temperature=0.5, max_tokens=512, device="auto", do_sample=False):
        """
        Initialize the Qwen2_5_VL model

        Args:
            model: The underlying model instance
            tokenizer: The tokenizer instance
            temperature: Sampling temperature (default: 0.5)
            max_tokens: Maximum tokens for generation (default: 512)
            device: Device to run inference on (default: "auto")
        """
        
        # default: Load the model on the available device(s)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        )

        self.processor = AutoProcessor.from_pretrained(model_name)


        self.max_tokens = max_tokens
        self.device = device
        self.do_sample = do_sample
        self.temperature = temperature

        super(Qwen2_5_VL, self).__init__(self.model, self.tokenizer, temperature, max_tokens, device)



    def set_tokenizer(self, tokenizer):
        """Set or update the tokenizer for the model"""
        self.tokenizer = tokenizer


    def get_response(self, data):
        """
        Generate a response from the model given input data

        Args:
            data: Input data (text, dict with image/text, or list of inputs)

            for single image:
                data = {
                    "prompt": "What is the color of the car?",
                    "image": ["path/to/image"],
                    "is_img_url": False
                }
            
            for multi-image:
            
                data = {
                    "prompt": "What is the color of the car?",
                    "image": ["path/to/image1","path/to/image2"]
                    "is_img_url": False
                }


        Returns:
            Dict containing response and metadata
        """
        prompt = data["prompt"]

        messages = [
            {
                "role": "user",
                "content": [
                
                ],
            }
        ]

        for img in data["image"]:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response

    def predict(self, data):
        """
        Make a prediction based on input data

        Args:
            data: Input data (text, dict with image/text, or list of inputs)

            for single image:
                data = {
                    "prompt": "What is the color of the car?",
                    "image": ["path/to/image"],
                    "is_img_url": False
                }
            
            for multi-image:
            
                data = {
                    "prompt": "What is the color of the car?",
                    "image": ["path/to/image1","path/to/image2"]
                    "is_img_url": False
                }


        Returns:
            string containing text response
        """
        return self.get_response(data)