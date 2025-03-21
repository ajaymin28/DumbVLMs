"""
@Vi-Sri 
Shape Benchmark Generator
This module generates a dataset of shape matching challenges with various configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import os
import math
import string
from PIL import Image, ImageDraw, ImageFont

class ShapeBenchmarkGenerator:
    """
    Core class for generating shape benchmarks with various configurations.
    This class handles the low-level shape generation and rendering.
    """
    
    def __init__(self, 
                 num_shapes=5, 
                 min_sides=3, 
                 max_sides=10,
                 img_size=(800, 600),
                 colors=None,
                 output_dir="shape_benchmark",
                 spatial_complexity="easy",
                 rotation_degree="default",
                 seed=None):
        """
        Initialize the shape benchmark generator.
        
        Args:
            num_shapes (int): Number of shapes to generate per image
            min_sides (int): Minimum number of sides for random polygons
            max_sides (int): Maximum number of sides for random polygons
            img_size (tuple): Size of the output image (width, height)
            colors (tuple): Colors for query (numbered) and gallery (alphabetical) shapes
            output_dir (str): Directory to save generated images
            spatial_complexity (str): Complexity of spatial arrangement ("easy", "medium", "hard")
            rotation_degree (str): Degree of rotation ("default", "random", or a number in degrees)
            seed (int): Random seed for reproducibility
        """
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
        
        # Default colors: red for query (numbered) shapes, green for gallery (alphabetical) shapes
        self.colors = colors if colors else (("#FF5733", "#C70039"), ("#33FF57", "#0039C7"))
        self.output_dir = output_dir
        
        self.spatial_complexity = spatial_complexity.lower()
        if self.spatial_complexity not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid spatial complexity: {spatial_complexity}. Must be 'easy', 'medium', or 'hard'")
        
        self.rotation_degree = rotation_degree
        self._validate_rotation_degree()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def _validate_rotation_degree(self):
        """Validate and normalize rotation degree settings"""
        if self.rotation_degree == "default":
            self.rotation_value = 0
            self.random_rotation = False
        elif self.rotation_degree == "random":
            self.rotation_value = 0
            self.random_rotation = True
        else:
            try:
                self.rotation_value = float(self.rotation_degree)
                self.random_rotation = False
            except ValueError:
                raise ValueError(f"Invalid rotation degree: {self.rotation_degree}. Must be 'default', 'random', or a number")
            
    def generate_random_polygon(self, center, radius, num_sides):
        """
        Generate a random polygon with the given number of sides.
        
        Args:
            center (tuple): Center coordinates (x, y)
            radius (float): Radius of the circumscribed circle
            num_sides (int): Number of sides for the polygon
            
        Returns:
            list: List of (x, y) coordinates for polygon vertices
        """
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        
        angles += np.random.uniform(-0.2, 0.2, num_sides)
        
        radii = np.random.uniform(0.7 * radius, 1.0 * radius, num_sides)
        
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        return list(zip(x, y))
    
    def generate_random_position(self, shape_radius, positions, min_distance):
        """
        Generate a random position for a shape ensuring no overlap with existing shapes.
        
        Args:
            shape_radius (float): Approximate radius of the shape
            positions (list): List of existing shape positions
            min_distance (float): Minimum distance between shape centers
            
        Returns:
            tuple: (x, y) coordinates for the new shape
        """
        max_attempts = 100
        
        if self.spatial_complexity == "easy":
            return None
            
        elif self.spatial_complexity == "medium":
            center_x = self.img_size[0] / 2
            center_y = self.img_size[1] / 2
            radius = min(self.img_size[0], self.img_size[1]) / 4
            
            for _ in range(max_attempts):
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, radius)
                x = center_x + distance * math.cos(angle)
                y = center_y + distance * math.sin(angle)
                
                valid_position = True
                for pos in positions:
                    if pos is None:
                        continue
                    distance = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
                    if distance < min_distance:
                        valid_position = False
                        break
                        
                if valid_position:
                    return (x, y)
                    
        elif self.spatial_complexity == "hard":
            for _ in range(max_attempts):
                margin = shape_radius * 1.5
                x = random.uniform(margin, self.img_size[0] - margin)
                y = random.uniform(margin, self.img_size[1] - margin)
                
                valid_position = True
                for pos in positions:
                    if pos is None:
                        continue
                    distance = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
                    if distance < min_distance:
                        valid_position = False
                        break
                        
                if valid_position:
                    return (x, y)
        
        return (random.uniform(shape_radius * 2, self.img_size[0] - shape_radius * 2),
                random.uniform(shape_radius * 2, self.img_size[1] - shape_radius * 2))
    
    def apply_rotation(self, vertices, center, angle_degrees):
        """
        Apply rotation to vertices around a center point.
        
        Args:
            vertices (list): List of (x, y) coordinates
            center (tuple): Center point (x, y) for rotation
            angle_degrees (float): Rotation angle in degrees
            
        Returns:
            list: Rotated vertices
        """
        angle_rad = math.radians(angle_degrees)
        rotated_vertices = []
        
        for x, y in vertices:
            x_centered = x - center[0]
            y_centered = y - center[1]
            
            x_rotated = x_centered * math.cos(angle_rad) - y_centered * math.sin(angle_rad)
            y_rotated = x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)
            
            rotated_vertices.append((x_rotated + center[0], y_rotated + center[1]))
            
        return rotated_vertices
    
    def get_shape_positions_easy(self):
        """
        Get positions for shapes in the 'easy' spatial complexity (column-based).
        
        Returns:
            tuple: (query_positions, gallery_positions) lists of positions
        """
        query_positions = []
        gallery_positions = []
        
        col_width = self.img_size[0] // 2
        col_height = self.img_size[1]
        
        for i in range(self.num_shapes):
            y_pos = (i + 0.5) * (col_height / self.num_shapes)
            query_positions.append((col_width // 2, y_pos))
        
        for i in range(self.num_shapes):
            y_pos = (i + 0.5) * (col_height / self.num_shapes)
            gallery_positions.append((col_width + col_width // 2, y_pos))
            
        return query_positions, gallery_positions
    
    def generate_base_shapes(self):
        """
        Generate base shapes with unique numbers of sides.
        
        Returns:
            list: List of shape dictionaries with 'id', 'sides', and 'vertices' keys
        """
        base_shapes = []
        
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
            
        return base_shapes
    
    def get_rotation_angle(self, is_gallery=False):
        """
        Get rotation angle based on settings.
        
        Args:
            is_gallery (bool): Whether this is for gallery shapes
            
        Returns:
            float: Rotation angle in degrees
        """
        if self.random_rotation:
            return random.uniform(0, 360)
        elif is_gallery and self.rotation_value != 0:
            return self.rotation_value
        else:
            return 0
    
    def render_shapes_on_image(self, base_shapes, query_positions, gallery_positions, gallery_indices=None):
        """
        Render shapes on an image with the specified positions and rotations.
        
        Args:
            base_shapes (list): List of base shapes to render
            query_positions (list): Positions for query (numbered) shapes
            gallery_positions (list): Positions for gallery (alphabetical) shapes
            gallery_indices (list): Indices of shapes to use for gallery (shuffled if None)
            
        Returns:
            tuple: (PIL Image, list of query shapes, list of gallery shapes, list of matchings)
        """
        img = Image.new('RGB', self.img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        query_shapes = []
        for i, base_shape in enumerate(base_shapes):
            pos = query_positions[i]
            
            shape_vertices = [(x + pos[0], y + pos[1]) for x, y in base_shape['vertices']]
            
            rotation_angle = self.get_rotation_angle(is_gallery=False)
            if rotation_angle != 0:
                shape_vertices = self.apply_rotation(shape_vertices, pos, rotation_angle)
            
            draw.polygon(shape_vertices, fill=self.colors[0][0], outline=self.colors[0][1])
            
            number_label = str(i + 1)
            text_pos = (pos[0], pos[1])
            draw.text(text_pos, number_label, fill="black")
            
            query_shapes.append({
                'id': i + 1,
                'position': pos,
                'vertices': shape_vertices,
                'rotation': rotation_angle
            })
        
        gallery_shapes = []
        alphabet = string.ascii_uppercase
        
        if gallery_indices is None:
            gallery_indices = list(range(self.num_shapes))
            random.shuffle(gallery_indices)
        
        matchings = []  
        
        for i, shape_idx in enumerate(gallery_indices):
            base_shape = base_shapes[shape_idx]
            pos = gallery_positions[i]
            
            shape_vertices = [(x + pos[0], y + pos[1]) for x, y in base_shape['vertices']]
            
            rotation_angle = self.get_rotation_angle(is_gallery=True)
            if rotation_angle != 0:
                shape_vertices = self.apply_rotation(shape_vertices, pos, rotation_angle)
            
            draw.polygon(shape_vertices, fill=self.colors[1][0], outline=self.colors[1][1])
            
            alpha_label = alphabet[i]
            text_pos = (pos[0], pos[1])
            draw.text(text_pos, alpha_label, fill="black")
            
            gallery_shapes.append({
                'id': alpha_label,
                'position': pos,
                'vertices': shape_vertices,
                'rotation': rotation_angle,
                'base_shape_id': shape_idx + 1
            })
            
            matchings.append((shape_idx + 1, alpha_label))
        
        return img, query_shapes, gallery_shapes, matchings
    
    def render_high_quality_image(self, output_path, query_shapes, gallery_shapes, matchings=None):
        """
        Render a high-quality image using matplotlib for better visualization.
        
        Args:
            output_path (str): Path to save the output image
            query_shapes (list): List of query shapes
            gallery_shapes (list): List of gallery shapes
            matchings (list): List of matchings between query and gallery shapes
        """
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        for shape in query_shapes:
            polygon = Polygon(shape['vertices'], closed=True, 
                             facecolor=self.colors[0][0], edgecolor=self.colors[0][1], alpha=0.8)
            ax.add_patch(polygon)
            plt.text(shape['position'][0], shape['position'][1], 
                    str(shape['id']), ha='center', va='center', fontsize=12, color='black')
        
        for shape in gallery_shapes:
            polygon = Polygon(shape['vertices'], closed=True, 
                             facecolor=self.colors[1][0], edgecolor=self.colors[1][1], alpha=0.8)
            ax.add_patch(polygon)
            plt.text(shape['position'][0], shape['position'][1], 
                    shape['id'], ha='center', va='center', fontsize=12, color='black')
        
        ax.set_title(f"Shape Matching Challenge", fontsize=14)
        ax.set_xlim(0, self.img_size[0])
        ax.set_ylim(0, self.img_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        if matchings:
            gt_text = "Ground Truth: " + ", ".join([f"{m[0]}-{m[1]}" for m in matchings])
            plt.figtext(0.5, 0.02, gt_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_ground_truth(self, output_path, matchings):
        """
        Save ground truth matchings to a file.
        
        Args:
            output_path (str): Path to save the ground truth file
            matchings (list): List of (query_id, gallery_id) matchings
        """
        sorted_matchings = sorted(matchings, key=lambda x: x[0])
        
        with open(output_path, 'w') as f:
            for match in sorted_matchings:
                f.write(f"Red {match[0]} -> Green {match[1]}\n")
    
    def generate_dataset(self, num_samples=10, include_ground_truth_in_image=False):
        """
        Generate a dataset of shape matching samples.
        
        Args:
            num_samples (int): Number of samples to generate
            include_ground_truth_in_image (bool): Whether to include ground truth in the image
            
        Returns:
            list: List of dictionaries with information about each sample
        """
        samples_info = []
        
        for sample_idx in range(num_samples):
            sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            base_shapes = self.generate_base_shapes()
            
            if self.spatial_complexity == "easy":
                query_positions, gallery_positions = self.get_shape_positions_easy()
            else:
                positions = []
                min_distance = 100
                
                query_positions = []
                for _ in range(self.num_shapes):
                    pos = self.generate_random_position(40, positions, min_distance)
                    query_positions.append(pos)
                    positions.append(pos)
                
                gallery_positions = []
                for _ in range(self.num_shapes):
                    pos = self.generate_random_position(40, positions, min_distance)
                    gallery_positions.append(pos)
                    positions.append(pos)
            
            img, query_shapes, gallery_shapes, matchings = self.render_shapes_on_image(
                base_shapes, query_positions, gallery_positions
            )
            
            output_path = os.path.join(sample_dir, "sample.png")
            img.save(output_path)
            
            if include_ground_truth_in_image:
                hq_output_path = os.path.join(sample_dir, "sample_high_quality.png")
                self.render_high_quality_image(hq_output_path, query_shapes, gallery_shapes, matchings)
            else:
                hq_output_path = os.path.join(sample_dir, "sample_high_quality.png")
                self.render_high_quality_image(hq_output_path, query_shapes, gallery_shapes)
            
            gt_output_path = os.path.join(sample_dir, "ground_truth.txt")
            self.save_ground_truth(gt_output_path, matchings)
            
            samples_info.append({
                'sample_idx': sample_idx,
                'image_path': output_path,
                'high_quality_image_path': hq_output_path,
                'ground_truth_path': gt_output_path,
                'query_shapes': query_shapes,
                'gallery_shapes': gallery_shapes,
                'matchings': matchings
            })
            
            print(f"Generated sample {sample_idx:03d}")
        
        return samples_info