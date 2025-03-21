from baselines.models import ModelInferenceEngine
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch
import requests
import torchvision.transforms as T
# from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL2_5(ModelInferenceEngine):
    """InternVL2_5 model
    """
    
    def __init__(self,model_name="OpenGVLab/InternVL2_5-1B",temperature=0.5, max_tokens=512, device="auto", do_sample=False):
        """
        Initialize the InternVL2_5 model

        Args:
            model: The underlying model instance
            tokenizer: The tokenizer instance
            temperature: Sampling temperature (default: 0.5)
            max_tokens: Maximum tokens for generation (default: 512)
            device: Device to run inference on (default: "auto")
        """

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        self.max_tokens = max_tokens
        self.device = device
        self.do_sample = do_sample
        self.temperature = temperature

        super(InternVL2_5, self).__init__(self.model, self.tokenizer, temperature, max_tokens, device)



    def set_tokenizer(self, tokenizer):
        """Set or update the tokenizer for the model"""
        pass


    def build_transform(self,input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self,aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self,image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    
    def preprocess_image(self, image_file, input_size=448, max_num=12, isURL=False):
        if isURL:
            image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

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

        if len(data["image"]) == 1:
            pixel_values = self.preprocess_image(data["image"][0], max_num=12, isURL=data["is_img_url"]).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=self.max_tokens, do_sample=self.do_sample, temperature=self.temperature)

            # single-image single-round conversation
            question = f'<image>\n{prompt}'
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            # print(f'User: {question}\nAssistant: {response}')
            return response

        # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
        pixel_values1 = self.preprocess_image(data["image"][0], max_num=12, isURL=data["is_img_url"]).to(torch.bfloat16).cuda()
        pixel_values2 = self.preprocess_image(data["image"][1], max_num=12, isURL=data["is_img_url"]).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = f'Image-1: <image>\nImage-2: <image>\n{prompt}'
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')

        # question = 'What are the similarities and differences between these two images.'
        # response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
        #                             num_patches_list=num_patches_list,
        #                             history=history, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')


        return response

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