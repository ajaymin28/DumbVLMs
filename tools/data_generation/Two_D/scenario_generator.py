# scenario_generators.py
import os
import random
import numpy as np
import math
import string
from .shape_generator import ShapeBenchmarkGenerator

class BaseScenario:
    """Base class for all scenario generators"""
    
    def __init__(self, config, output_dir, seed=None):
        """
        Initialize the scenario generator.
        
        Args:
            config (dict): Configuration dictionary
            output_dir (str): Directory to save generated datasets
            seed (int): Random seed for reproducibility
        """
        self.config = config
        self.output_dir = output_dir
        self.seed = seed
        
        self.scenario_dir = os.path.join(output_dir, config.get("scenario_name", self.__class__.__name__))
        os.makedirs(self.scenario_dir, exist_ok=True)
    
    def generate(self):
        """Generate the dataset for this scenario"""
        raise NotImplementedError("Subclasses must implement this method")


class ShapeMatchingScenario(BaseScenario):
    """
    Scenario 1: Shape, position and rotation matching
    
    Features:
    - Number of shapes: 2-5 shapes between query and gallery
    - Spatial complexity: Easy (column-based), Medium (clustered), Hard (random)
    - Rotation: None, Random, or Custom angle
    """
    
    def generate(self):
        """Generate the shape matching dataset"""
        num_shapes = self.config.get("num_shapes", 5)
        min_sides = self.config.get("min_sides", 3)
        max_sides = self.config.get("max_sides", 10)
        img_size = tuple(self.config.get("img_size", [800, 600]))
        colors = self.config.get("colors", None)
        num_samples = self.config.get("num_samples", 10)
        spatial_complexity = self.config.get("spatial_complexity", "easy")
        rotation_degree = self.config.get("rotation_degree", "default")
        
        generator = ShapeBenchmarkGenerator(
            num_shapes=num_shapes,
            min_sides=min_sides,
            max_sides=max_sides,
            img_size=img_size,
            colors=colors,
            output_dir=self.scenario_dir,
            spatial_complexity=spatial_complexity,
            rotation_degree=rotation_degree,
            seed=self.seed
        )
        
        samples_info = generator.generate_dataset(
            num_samples=num_samples,
            include_ground_truth_in_image=False
        )
        
        with open(os.path.join(self.scenario_dir, "summary.txt"), 'w') as f:
            f.write("Shape Matching Scenario\n")
            f.write("=====================\n\n")
            f.write(f"Number of shapes: {num_shapes}\n")
            f.write(f"Spatial complexity: {spatial_complexity}\n")
            f.write(f"Rotation degree: {rotation_degree}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Image size: {img_size}\n")
            f.write("\nSample details:\n")
            
            for sample in samples_info:
                f.write(f"\nSample {sample['sample_idx']:03d}:\n")
                f.write(f"  Image: {os.path.basename(sample['image_path'])}\n")
                f.write(f"  Ground truth: {os.path.basename(sample['ground_truth_path'])}\n")
                
                match_info = ", ".join([f"{m[0]}-{m[1]}" for m in sorted(sample['matchings'], key=lambda x: x[0])])
                f.write(f"  Matchings: {match_info}\n")
        
        print(f"Generated {num_samples} samples for shape matching scenario")
        return samples_info


class RotationReasoningScenario(BaseScenario):
    """
    Scenario 2: Rotation reasoning
    
    Features:
    - Single query shape (red) with N gallery shapes (green)
    - Specific rotation angles to reason about
    - Prompts for each image asking which gallery shape the query would match after rotation
    """
    
    def generate(self):
        """Generate the rotation reasoning dataset"""
        gallery_size = self.config.get("gallery_size", 5)  
        min_sides = self.config.get("min_sides", 3)
        max_sides = self.config.get("max_sides", 10)
        img_size = tuple(self.config.get("img_size", [800, 600]))
        colors = self.config.get("colors", None)
        num_samples = self.config.get("num_samples", 10)
        rotation_angles = self.config.get("rotation_angles", [90, 180, 270])
        
        # Ensure we have enough sides for gallery_size + 1 (query) unique shapes
        if max_sides - min_sides + 1 < gallery_size + 1:
            raise ValueError(f"Not enough unique shapes available (need {gallery_size + 1}, have {max_sides - min_sides + 1})")
        
        samples_info = []
        
        for sample_idx in range(num_samples):
            sample_dir = os.path.join(self.scenario_dir, f"sample_{sample_idx:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            for rot_idx, rotation_angle in enumerate(rotation_angles):
                rotation_dir = os.path.join(sample_dir, f"rotation_{rotation_angle}")
                os.makedirs(rotation_dir, exist_ok=True)
                
                generator = ShapeBenchmarkGenerator(
                    num_shapes=gallery_size + 1,  # +1 for query shape
                    min_sides=min_sides,
                    max_sides=max_sides,
                    img_size=img_size,
                    colors=colors,
                    output_dir=rotation_dir,
                    spatial_complexity="easy",  
                    rotation_degree="default",
                    seed=self.seed + sample_idx + rot_idx if self.seed else None
                )
                
                base_shapes = generator.generate_base_shapes()
                
                query_shape_idx = 0
                query_shape = base_shapes[query_shape_idx]
                gallery_base_shapes = base_shapes[1:]

                # Randomly select a gallery shape to be the correct match   
                correct_gallery_idx = random.randint(0, gallery_size - 1)                 
                gallery_indices = list(range(gallery_size))
                
                query_pos = [(img_size[0] // 4, img_size[1] // 2)]
                gallery_positions = []
                
                gallery_height = img_size[1] * 0.8 
                gallery_y_start = (img_size[1] - gallery_height) / 2
                
                for i in range(gallery_size):
                    y_pos = gallery_y_start + (i + 0.5) * (gallery_height / gallery_size)
                    gallery_positions.append((3 * img_size[0] // 4, y_pos))
                
                query_shapes = [base_shapes[query_shape_idx]]
                
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', img_size, color='white')
                draw = ImageDraw.Draw(img)
                
                query_pos_center = query_pos[0]
                query_vertices = [(x + query_pos_center[0], y + query_pos_center[1]) 
                                 for x, y in query_shape['vertices']]
                
                draw.polygon(query_vertices, fill=colors[0][0] if colors else "#FF5733", 
                            outline=colors[0][1] if colors else "#C70039")
                
                draw.text(query_pos_center, "?", fill="black")
                
                gallery_shapes = []
                alphabet = string.ascii_uppercase
                
                for i, base_shape_idx in enumerate(range(1, gallery_size + 1)):
                    base_shape = base_shapes[base_shape_idx]
                    pos = gallery_positions[i]
                    
                    vertices = [(x + pos[0], y + pos[1]) for x, y in base_shape['vertices']]
                    
                    draw.polygon(vertices, fill=colors[1][0] if colors else "#33FF57", 
                                outline=colors[1][1] if colors else "#0039C7")
                    
                    alpha_label = alphabet[i]
                    draw.text(pos, alpha_label, fill="black")
                    
                    gallery_shapes.append({
                        'id': alpha_label,
                        'position': pos,
                        'vertices': vertices,
                        'base_shape_id': base_shape_idx
                    })
                
                img_path = os.path.join(rotation_dir, "rotation_challenge.png")
                img.save(img_path)
                
                prompt_path = os.path.join(rotation_dir, "prompt.txt")
                with open(prompt_path, 'w') as f:
                    f.write(f"If the shape in Red is rotated by {rotation_angle} degrees, "
                            f"what shape would it match to? Give a single alphabet answer.")
                
                correct_letter = alphabet[correct_gallery_idx]
                gt_path = os.path.join(rotation_dir, "ground_truth.txt")
                with open(gt_path, 'w') as f:
                    f.write(f"{correct_letter}")
                
                samples_info.append({
                    'sample_idx': sample_idx,
                    'rotation_angle': rotation_angle,
                    'image_path': img_path,
                    'prompt_path': prompt_path,
                    'ground_truth_path': gt_path,
                    'correct_answer': correct_letter
                })
                
                print(f"Generated rotation sample {sample_idx:03d} with {rotation_angle}° rotation")
        
        with open(os.path.join(self.scenario_dir, "summary.txt"), 'w') as f:
            f.write("Rotation Reasoning Scenario\n")
            f.write("==========================\n\n")
            f.write(f"Gallery size: {gallery_size}\n")
            f.write(f"Rotation angles: {rotation_angles}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Image size: {img_size}\n")
            f.write("\nSample details:\n")
            
            for sample in samples_info:
                f.write(f"\nSample {sample['sample_idx']:03d} - {sample['rotation_angle']}°:\n")
                f.write(f"  Image: {os.path.basename(sample['image_path'])}\n")
                f.write(f"  Prompt: {os.path.basename(sample['prompt_path'])}\n")
                f.write(f"  Correct answer: {sample['correct_answer']}\n")
        
        print(f"Generated {len(samples_info)} rotation reasoning samples")
        return samples_info


class OddOneOutScenario(BaseScenario):
    """
    Scenario 3: Odd one out
    
    Features:
    - Generate sequences of shapes where one is different
    - Oddity criteria: size, shape, color, rotation_angle
    - Prompt asking to identify the odd shape
    """
    
    def generate(self):
        """Generate the odd one out dataset"""
        sequence_length = self.config.get("sequence_length", 5)  # Number of shapes per sequence
        min_sides = self.config.get("min_sides", 3)
        max_sides = self.config.get("max_sides", 10)
        img_size = tuple(self.config.get("img_size", [800, 600]))
        num_samples = self.config.get("num_samples", 10)
        oddity_criteria = self.config.get("oddity_criteria", ["shape", "color", "rotation_angle", "size"])
        
        samples_info = []
        
        for sample_idx in range(num_samples):
            for criterion in oddity_criteria:
                criterion_dir = os.path.join(self.scenario_dir, f"sample_{sample_idx:03d}_{criterion}")
                os.makedirs(criterion_dir, exist_ok=True)
                
                if criterion == "shape":
                    samples_info.append(self._generate_shape_oddity(
                        criterion_dir, sample_idx, sequence_length, min_sides, max_sides, img_size
                    ))
                    
                elif criterion == "color":
                    samples_info.append(self._generate_color_oddity(
                        criterion_dir, sample_idx, sequence_length, min_sides, max_sides, img_size
                    ))
                    
                elif criterion == "rotation_angle":
                    samples_info.append(self._generate_rotation_oddity(
                        criterion_dir, sample_idx, sequence_length, min_sides, max_sides, img_size
                    ))
                    
                elif criterion == "size":
                    samples_info.append(self._generate_size_oddity(
                        criterion_dir, sample_idx, sequence_length, min_sides, max_sides, img_size
                    ))
                    
                else:
                    raise ValueError(f"Unknown oddity criterion: {criterion}")
                
                print(f"Generated odd one out sample {sample_idx:03d} with {criterion} oddity")
        
        with open(os.path.join(self.scenario_dir, "summary.txt"), 'w') as f:
            f.write("Odd One Out Scenario\n")
            f.write("===================\n\n")
            f.write(f"Sequence length: {sequence_length}\n")
            f.write(f"Oddity criteria: {oddity_criteria}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Image size: {img_size}\n")
            f.write("\nSample details:\n")
            
            for sample in samples_info:
                f.write(f"\nSample {sample['sample_idx']:03d} - {sample['criterion']}:\n")
                f.write(f"  Image: {os.path.basename(sample['image_path'])}\n")
                f.write(f"  Prompt: {os.path.basename(sample['prompt_path'])}\n")
                f.write(f"  Correct answer: {sample['correct_answer']}\n")
        
        print(f"Generated {len(samples_info)} odd one out samples")
        return samples_info
    
    def _generate_shape_oddity(self, output_dir, sample_idx, sequence_length, min_sides, max_sides, img_size):
        """Generate a sample where one shape has a different number of sides"""
        regular_sides = random.randint(min_sides, max_sides)
        
        available_sides = list(range(min_sides, max_sides + 1))
        available_sides.remove(regular_sides)
        odd_sides = random.choice(available_sides)
        
        odd_position = random.randint(0, sequence_length - 1)
        
        shapes = []
        for i in range(sequence_length):
            sides = odd_sides if i == odd_position else regular_sides
            shapes.append(self._generate_shape(sides, radius=40))
        
        return self._create_odd_one_out_sample(
            output_dir, sample_idx, shapes, odd_position, 
            "shape", img_size, description=f"Different number of sides ({odd_sides} vs {regular_sides})"
        )
    
    def _generate_color_oddity(self, output_dir, sample_idx, sequence_length, min_sides, max_sides, img_size):
        """Generate a sample where one shape has a different color"""
        sides = random.randint(min_sides, max_sides)
        
        regular_color = ("#3498db", "#2980b9")  # Blue
        odd_color = ("#e74c3c", "#c0392b")      # Red
        
        odd_position = random.randint(0, sequence_length - 1)
        
        shapes = []
        for i in range(sequence_length):
            shapes.append(self._generate_shape(sides, radius=40))
        
        return self._create_odd_one_out_sample(
            output_dir, sample_idx, shapes, odd_position, 
            "color", img_size, 
            colors=[regular_color if i != odd_position else odd_color for i in range(sequence_length)],
            description=f"Different color (Red vs Blue)"
        )
    
    def _generate_rotation_oddity(self, output_dir, sample_idx, sequence_length, min_sides, max_sides, img_size):
        """Generate a sample where one shape has a different rotation"""
        sides = random.randint(min_sides, max_sides)
        
        regular_rotation = 0
        odd_rotation = 45  
        
        odd_position = random.randint(0, sequence_length - 1)
        
        shapes = []
        for i in range(sequence_length):
            shapes.append(self._generate_shape(sides, radius=40))
        
        return self._create_odd_one_out_sample(
            output_dir, sample_idx, shapes, odd_position, 
            "rotation_angle", img_size, 
            rotations=[regular_rotation if i != odd_position else odd_rotation for i in range(sequence_length)],
            description=f"Different rotation (0° vs 45°)"
        )
    
    def _generate_size_oddity(self, output_dir, sample_idx, sequence_length, min_sides, max_sides, img_size):
        """Generate a sample where one shape has a different size"""
        sides = random.randint(min_sides, max_sides)
        
        regular_radius = 40
        odd_radius = 25  
        
        odd_position = random.randint(0, sequence_length - 1)
        
        shapes = []
        for i in range(sequence_length):
            radius = odd_radius if i == odd_position else regular_radius
            shapes.append(self._generate_shape(sides, radius=radius))
        
        return self._create_odd_one_out_sample(
            output_dir, sample_idx, shapes, odd_position, 
            "size", img_size, 
            description=f"Different size (25 vs 40 radius)"
        )
    
    def _generate_shape(self, num_sides, radius=40):
        """Generate a shape with the specified number of sides and radius"""
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        
        angles += np.random.uniform(-0.1, 0.1, num_sides)
        
        radii = np.random.uniform(0.85 * radius, radius, num_sides)
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        return list(zip(x, y))
    
    def _create_odd_one_out_sample(self, output_dir, sample_idx, shapes, odd_position, criterion, 
                                  img_size, colors=None, rotations=None, description=""):
        """Create an odd one out sample with the given shapes and oddity"""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        positions = []
        width = img_size[0]
        height = img_size[1]
        
        sequence_length = len(shapes)
        spacing = width / (sequence_length + 1)
        
        for i in range(sequence_length):
            positions.append((spacing * (i + 1), height / 2))
        
        if colors is None:
            colors = [("#3498db", "#2980b9")] * sequence_length  # Blue for all
        
        if rotations is None:
            rotations = [0] * sequence_length  # No rotation
        
        alphabet = string.ascii_uppercase
        shape_info = []
        
        for i, (shape, pos, color, rotation) in enumerate(zip(shapes, positions, colors, rotations)):
            # Get shape vertices centered at position
            vertices = [(x + pos[0], y + pos[1]) for x, y in shape]
            
            if rotation != 0:
                # Rotate vertices around center
                angle_rad = math.radians(rotation)
                rotated_vertices = []
                for x, y in vertices:
                    # Translate to origin
                    x_centered = x - pos[0]
                    y_centered = y - pos[1]
                    
                    x_rotated = x_centered * math.cos(angle_rad) - y_centered * math.sin(angle_rad)
                    y_rotated = x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)
                    
                    rotated_vertices.append((x_rotated + pos[0], y_rotated + pos[1]))
                
                vertices = rotated_vertices
            
            draw.polygon(vertices, fill=color[0], outline=color[1])
            
            alpha_label = alphabet[i]
            draw.text(pos, alpha_label, fill="black")
            
            shape_info.append({
                'id': alpha_label,
                'position': pos,
                'vertices': vertices
            })
        
        img_path = os.path.join(output_dir, "odd_one_out.png")
        img.save(img_path)
        
        prompt_path = os.path.join(output_dir, "prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write("Pick the odd one out from the image. Give only the alphabet of the diagram as answer.")
        
        correct_letter = alphabet[odd_position]
        gt_path = os.path.join(output_dir, "ground_truth.txt")
        with open(gt_path, 'w') as f:
            f.write(f"{correct_letter}")
        
        return {
            'sample_idx': sample_idx,
            'criterion': criterion,
            'image_path': img_path,
            'prompt_path': prompt_path,
            'ground_truth_path': gt_path,
            'correct_answer': correct_letter,
            'odd_position': odd_position,
            'description': description
        }