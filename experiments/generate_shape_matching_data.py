import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import os
from PIL import Image, ImageDraw, ImageFont
import math
import string

class ShapeBenchmarkGenerator:
    def __init__(self, 
                 num_shapes=5, 
                 min_sides=3, 
                 max_sides=10,
                 img_size=(800, 600),
                 colors=None,
                 output_dir="shape_benchmark",
                 seed=None):
                 
        if max_sides - min_sides + 1 < num_shapes:
            raise ValueError(f"Cannot generate {num_shapes} unique shapes when min_sides={min_sides} and max_sides={max_sides}. "
                             f"Please ensure max_sides - min_sides + 1 >= num_shapes")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.num_shapes = num_shapes
        self.min_sides = min_sides
        self.max_sides = max_sides
        self.img_size = img_size
    
        self.colors = colors if colors else (("#FF5733", "#C70039"), ("#33FF57", "#0039C7"))
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.gt_file = open(os.path.join(output_dir, "ground_truth.txt"), "w")
            
    def generate_random_polygon(self, center, radius, num_sides):
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)        
        angles += np.random.uniform(-0.2, 0.2, num_sides)        
        radii = np.random.uniform(0.7 * radius, 1.0 * radius, num_sides)        
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        return list(zip(x, y))
    
    def generate_random_position(self, shape_radius, positions, min_distance):
        max_attempts = 100
        for _ in range(max_attempts):
            margin = shape_radius * 1.5
            x = random.uniform(margin, self.img_size[0] - margin)
            y = random.uniform(margin, self.img_size[1] - margin)            
            valid_position = True
            for pos in positions:
                distance = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
                if distance < min_distance:
                    valid_position = False
                    break
                    
            if valid_position:
                return (x, y)

        return (random.uniform(shape_radius, self.img_size[0] - shape_radius),
                random.uniform(shape_radius, self.img_size[1] - shape_radius))
    
    def generate_dataset(self, num_samples=10):
        for sample_idx in range(num_samples):
            base_shapes = []            
            if self.max_sides - self.min_sides + 1 < self.num_shapes:
                raise ValueError(f"Cannot generate {self.num_shapes} unique shapes when min_sides={self.min_sides} and max_sides={self.max_sides}")            
            available_sides = list(range(self.min_sides, self.max_sides + 1))
            random.shuffle(available_sides)
            unique_sides = available_sides[:self.num_shapes]
            
            for i, num_sides in enumerate(unique_sides):
                base_shapes.append({
                    'id': i + 1,
                    'sides': num_sides,
                    'vertices': self.generate_random_polygon(
                        center=(0, 0),
                        radius=40,
                        num_sides=num_sides
                    )
                })
            
            img = Image.new('RGB', self.img_size, color='white')
            draw = ImageDraw.Draw(img)
            
            positions = []
            min_distance = 120 
            
            red_shapes = []
            for i, base_shape in enumerate(base_shapes):
                pos = self.generate_random_position(40, positions, min_distance)
                positions.append(pos)
                
                rotation_angle = random.uniform(0, 2 * np.pi)
                rotated_vertices = []
                for x, y in base_shape['vertices']:
                    # Rotate the vertex
                    new_x = x * math.cos(rotation_angle) - y * math.sin(rotation_angle)
                    new_y = x * math.sin(rotation_angle) + y * math.cos(rotation_angle)
                    rotated_vertices.append((new_x + pos[0], new_y + pos[1]))
                
                draw.polygon(rotated_vertices, fill=self.colors[0][0], outline=self.colors[0][1])
                
                number_label = str(i + 1)
                text_pos = (pos[0], pos[1])
                draw.text(text_pos, number_label, fill="black")
                
                red_shapes.append({
                    'id': i + 1,
                    'position': pos,
                    'vertices': rotated_vertices
                })
            
            green_shapes = []
            alphabet = string.ascii_uppercase
            shape_indices = list(range(self.num_shapes))
            random.shuffle(shape_indices)  
            
            matchings = []  
            
            for i, shape_idx in enumerate(shape_indices):
                base_shape = base_shapes[shape_idx]
                
                pos = self.generate_random_position(40, positions, min_distance)
                positions.append(pos)
                
                rotation_angle = random.uniform(0, 2 * np.pi)
                rotated_vertices = []
                for x, y in base_shape['vertices']:
                    new_x = x * math.cos(rotation_angle) - y * math.sin(rotation_angle)
                    new_y = x * math.sin(rotation_angle) + y * math.cos(rotation_angle)
                    rotated_vertices.append((new_x + pos[0], new_y + pos[1]))
                
                draw.polygon(rotated_vertices, fill=self.colors[1][0], outline=self.colors[1][1])
                
                alpha_label = alphabet[i]
                text_pos = (pos[0], pos[1])
                draw.text(text_pos, alpha_label, fill="black")
                
                green_shapes.append({
                    'id': alpha_label,
                    'position': pos,
                    'vertices': rotated_vertices
                })
                
                matchings.append((shape_idx + 1, alpha_label))
            
            output_path = os.path.join(self.output_dir, f"sample_{sample_idx:03d}.png")
            img.save(output_path)
            
            self.gt_file.write(f"Sample {sample_idx:03d}:\n")
            sorted_matchings = sorted(matchings, key=lambda x: x[0])  # Sort by red shape number
            for match in sorted_matchings:
                self.gt_file.write(f"Red {match[0]} -> Green {match[1]}\n")
            self.gt_file.write("\n")
            print(f"Generated sample {sample_idx:03d}")
            self.generate_higher_quality_image(sample_idx, red_shapes, green_shapes, sorted_matchings)
        
        self.gt_file.close()
        print(f"Dataset generation completed. {num_samples} samples saved to {self.output_dir}")

    def generate_higher_quality_image(self, sample_idx, red_shapes, green_shapes, matchings):
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        for shape in red_shapes:
            polygon = Polygon(shape['vertices'], closed=True, 
                             facecolor=self.colors[0][0], edgecolor=self.colors[0][1], alpha=0.8)
            ax.add_patch(polygon)
            plt.text(shape['position'][0], shape['position'][1], 
                    str(shape['id']), ha='center', va='center', fontsize=12, color='black')
        
        for shape in green_shapes:
            polygon = Polygon(shape['vertices'], closed=True, 
                             facecolor=self.colors[1][0], edgecolor=self.colors[1][1], alpha=0.8)
            ax.add_patch(polygon)
            plt.text(shape['position'][0], shape['position'][1], 
                    shape['id'], ha='center', va='center', fontsize=12, color='black')
        
        ax.set_title(f"Shape Matching Challenge #{sample_idx:03d}", fontsize=14)
        ax.set_xlim(0, self.img_size[0])
        ax.set_ylim(0, self.img_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # TODO: Validate if the GT text is needed within the image
        gt_text = "Ground Truth: " + ", ".join([f"{m[0]}-{m[1]}" for m in matchings])
        plt.figtext(0.5, 0.02, gt_text, ha='center', fontsize=12)
        
        output_path = os.path.join(self.output_dir, f"sample_{sample_idx:03d}_high_quality.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    generator = ShapeBenchmarkGenerator(
        num_shapes=5,
        min_sides=3,
        max_sides=8,
        img_size=(1000, 800),
        colors=(("#FF5733", "#C70039"), ("#33FF57", "#0039C7")),  
        output_dir="shape_benchmark",
        seed=42 # Set seed for reproducibility
    )
    
    generator.generate_dataset(num_samples=10)
    
    print("Dataset generation completed!")