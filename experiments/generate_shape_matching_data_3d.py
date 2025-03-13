import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import string
import math
from PIL import Image, ImageDraw, ImageFont
import io

class Shape3DBenchmarkGenerator:

    def __init__(self, 
                 num_shapes=5, 
                 img_size=(800, 600),
                 colors=None,
                 output_dir="shape3d_benchmark",
                 seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.num_shapes = num_shapes
        self.img_size = img_size
        
        self.colors = colors if colors else (("#FF5733", "#C70039"), ("#33FF57", "#0039C7"))
        self.output_dir = output_dir
        
        self.shape_types = [
            "tetrahedron",     # 4 faces
            "cube",            # 6 faces
            "octahedron",      # 8 faces
            "dodecahedron",    # 12 faces
            "icosahedron",     # 20 faces
            # "truncated_cube",  # 14 faces #TODO: Fix the truncated cube shape
            "cuboctahedron",   # 14 faces
            "star",            # Variable faces
            "prism_3",         # Triangular prism
            "prism_5",         # Pentagonal prism
            "prism_6",         # Hexagonal prism
            "prism_8",         # Octagonal prism
            "pyramid_3",       # Triangular pyramid
            "pyramid_4",       # Square pyramid
            "pyramid_5",       # Pentagonal pyramid
            "pyramid_6"        # Hexagonal pyramid
        ]
        
        if len(self.shape_types) < num_shapes:
            raise ValueError(f"Not enough shape types ({len(self.shape_types)}) to generate {num_shapes} unique shapes")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.gt_file = open(os.path.join(output_dir, "ground_truth.txt"), "w")
    
    def generate_3d_shape(self, shape_type, scale=1.0):
        if shape_type == "tetrahedron":
            vertices = np.array([
                [1, 0, -1/np.sqrt(2)],
                [-1, 0, -1/np.sqrt(2)],
                [0, 1, 1/np.sqrt(2)],
                [0, -1, 1/np.sqrt(2)]
            ]) * scale
            faces = [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]
            
        elif shape_type == "cube":
            vertices = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1]
            ]) * scale
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # front
                [2, 3, 7, 6],  # back
                [0, 3, 7, 4],  # left
                [1, 2, 6, 5]   # right
            ]
            
        elif shape_type == "octahedron":
            vertices = np.array([
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1]
            ]) * scale
            faces = [
                [0, 2, 4], # top front
                [0, 4, 3], # top right
                [0, 3, 5], # top back
                [0, 5, 2], # top left
                [1, 2, 4], # bottom front
                [1, 4, 3], # bottom right
                [1, 3, 5], # bottom back
                [1, 5, 2] # bottom left
            ]
            
        elif shape_type == "dodecahedron":
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            vertices = np.array([
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
                [0, phi, 1/phi],
                [0, phi, -1/phi],
                [0, -phi, 1/phi],
                [0, -phi, -1/phi],
                [1/phi, 0, phi],
                [-1/phi, 0, phi],
                [1/phi, 0, -phi],
                [-1/phi, 0, -phi],
                [phi, 1/phi, 0],
                [-phi, 1/phi, 0],
                [phi, -1/phi, 0],
                [-phi, -1/phi, 0]
            ]) * scale
            faces = [
                [0, 8, 9, 1, 16],
                [0, 16, 18, 2, 12],
                [0, 12, 13, 4, 8],
                [1, 9, 5, 15, 14],
                [1, 14, 3, 18, 16],
                [2, 18, 3, 11, 10],
                [2, 10, 6, 13, 12],
                [3, 14, 15, 7, 11],
                [4, 13, 6, 19, 17],
                [4, 17, 5, 9, 8],
                [5, 17, 19, 7, 15],
                [6, 10, 11, 7, 19]
            ]
            
        elif shape_type == "icosahedron":
            phi = (1 + np.sqrt(5)) / 2 
            vertices = np.array([
                [0, 1, phi],
                [0, -1, phi],
                [0, 1, -phi],
                [0, -1, -phi],
                [1, phi, 0],
                [-1, phi, 0],
                [1, -phi, 0],
                [-1, -phi, 0],
                [phi, 0, 1],
                [-phi, 0, 1],
                [phi, 0, -1],
                [-phi, 0, -1]
            ]) * scale
            faces = [
                [0, 1, 8],
                [0, 8, 4],
                [0, 4, 5],
                [0, 5, 9],
                [0, 9, 1],
                [1, 6, 8],
                [1, 7, 6],
                [1, 9, 7],
                [2, 3, 10],
                [2, 10, 4],
                [2, 4, 5],
                [2, 5, 11],
                [2, 11, 3],
                [3, 6, 10],
                [3, 7, 6],
                [3, 11, 7],
                [4, 8, 10],
                [5, 9, 11],
                [6, 8, 10],
                [7, 9, 11]
            ]
            
        elif shape_type.startswith("prism_"):
            # n-sided prism
            n = int(shape_type.split("_")[1])
            
            # Generate n-sided polygon for top and bottom faces
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            bottom = np.array([[np.cos(a), np.sin(a), -1] for a in angles])
            top = np.array([[np.cos(a), np.sin(a), 1] for a in angles])
            
            # Combine vertices
            vertices = np.vstack([bottom, top]) * scale
            
            faces = []
            faces.append(list(range(n)))
            faces.append([i + n for i in range(n)][::-1])  # Reverse for correct normal
            for i in range(n):
                faces.append([i, (i + 1) % n, (i + 1) % n + n, i + n])
                
        elif shape_type.startswith("pyramid_"):
            n = int(shape_type.split("_")[1])
            
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            base = np.array([[np.cos(a), np.sin(a), 0] for a in angles])
            tip = np.array([[0, 0, 1.5]])  # Tip of pyramid
            
            vertices = np.vstack([base, tip]) * scale
            
            faces = []
            faces.append(list(range(n)))
            for i in range(n):
                faces.append([i, (i + 1) % n, n])
                
        # TODO: Fix the truncated cube shape - Bug in the face generation            
        elif shape_type == "truncated_cube":
            cube_verts = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1]
            ])
            
            # Truncate by finding points 1/3 of the way along each edge
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
            ]
            
            trunc_verts = []
            for edge in edges:
                v1 = cube_verts[edge[0]]
                v2 = cube_verts[edge[1]]
                trunc_verts.append((2*v1 + v2) / 3)
                trunc_verts.append((v1 + 2*v2) / 3)
                
            vertices = np.array(trunc_verts) * scale
            
            # The truncated cube has 14 faces: 8 hexagons (from the original vertices)
            # and 6 octagons (from the original faces)
            # TODO: Remove this, This is a simplified version
            faces = []
            for i in range(0, 24, 6):
                faces.append([i, i+1, i+3, i+5, i+7, i+9])
            
            # Add more faces to approximate the truncated cube
            # Creating the octagonal faces would require a more complex mapping
            
        elif shape_type == "cuboctahedron":
            # Has 12 vertices at the midpoints of the edges of a cube
            phi = 1 / np.sqrt(2)
            vertices = np.array([
                [1, 1, 0],
                [1, -1, 0],
                [-1, 1, 0],
                [-1, -1, 0],
                [1, 0, 1],
                [1, 0, -1],
                [-1, 0, 1],
                [-1, 0, -1],
                [0, 1, 1],
                [0, 1, -1],
                [0, -1, 1],
                [0, -1, -1]
            ]) * scale
            
            # 8 triangular faces and 6 square faces
            faces = [
                [0, 8, 4],
                [0, 5, 9],
                [1, 4, 10],
                [1, 11, 5],
                [2, 6, 8],
                [2, 9, 7],
                [3, 7, 11],
                [3, 10, 6],
                [0, 4, 5],
                [1, 5, 4],
                [2, 8, 9],
                [3, 11, 10],
                [4, 8, 10],
                [5, 11, 9],
                [6, 10, 8],
                [7, 9, 11]
            ]
            
        elif shape_type == "star":
            n_spikes = random.randint(5, 8)  # Random number of spikes
            
            # Generate points for a star
            angles = np.linspace(0, 2 * np.pi, 2 * n_spikes, endpoint=False)
            radii = [1.0, 0.4] * n_spikes  # Alternating between outer and inner points
            
            # Create the base (circular part)
            xy_coords = np.array([[r * np.cos(a), r * np.sin(a), 0] for r, a in zip(radii, angles)])
            
            # Add a top and bottom point
            top = np.array([[0, 0, 0.7]])
            bottom = np.array([[0, 0, -0.7]])
            
            # Combine vertices
            vertices = np.vstack([xy_coords, top, bottom]) * scale
            
            faces = []
            # Side triangular faces (from outer points to top)
            for i in range(0, 2 * n_spikes, 2):
                faces.append([i, (i + 2) % (2 * n_spikes), 2 * n_spikes])
            
            # Side triangular faces (from inner points to bottom)
            for i in range(1, 2 * n_spikes, 2):
                faces.append([i, (i + 2) % (2 * n_spikes), 2 * n_spikes + 1])
            
            # Connect outer to inner points
            for i in range(2 * n_spikes):
                next_i = (i + 1) % (2 * n_spikes)
                next_next_i = (i + 2) % (2 * n_spikes)
                faces.append([i, next_i, next_next_i])
            
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        return vertices, faces
    
    def apply_random_rotation(self, vertices):
        alpha = random.uniform(0, 2 * np.pi)  # rotation around x-axis
        beta = random.uniform(0, 2 * np.pi)   # rotation around y-axis
        gamma = random.uniform(0, 2 * np.pi)  # rotation around z-axis
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        return vertices @ R
    
    def render_3d_shape(self, ax, vertices, faces, color, edge_color, alpha=0.8):
        poly3d = [[vertices[idx] for idx in face] for face in faces]
        collection = Poly3DCollection(poly3d, alpha=alpha)
        collection.set_facecolor(color)
        collection.set_edgecolor(edge_color)
        ax.add_collection3d(collection)
    
    def generate_dataset(self, num_samples=10):
        for sample_idx in range(num_samples):
            fig = plt.figure(figsize=(12, 10))
            sample_shape_types = random.sample(self.shape_types, self.num_shapes)            
            matchings = []
            red_shapes = []
            green_shapes = []
            
            for i, shape_type in enumerate(sample_shape_types):
                ax = fig.add_subplot(2, self.num_shapes, i + 1, projection='3d')
                ax.set_title(f"{i + 1}", fontsize=14, color='black')
                vertices, faces = self.generate_3d_shape(shape_type, scale=0.7)
                rotated_vertices = self.apply_random_rotation(vertices)
                
                red_color = self.colors[0][0]
                red_edge = self.colors[0][1]
                self.render_3d_shape(ax, rotated_vertices, faces, red_color, red_edge)
                
                red_shapes.append({
                    'id': i + 1,
                    'type': shape_type,
                    'vertices': rotated_vertices,
                    'faces': faces
                })
                
                ax.set_xlim([-1.2, 1.2])
                ax.set_ylim([-1.2, 1.2])
                ax.set_zlim([-1.2, 1.2])
                ax.set_axis_off()
            
            shape_indices = list(range(self.num_shapes))
            random.shuffle(shape_indices)
            alphabet = string.ascii_uppercase
            
            for i, shape_idx in enumerate(shape_indices):
                shape_type = sample_shape_types[shape_idx]
                
                ax = fig.add_subplot(2, self.num_shapes, i + 1 + self.num_shapes, projection='3d')
                ax.set_title(alphabet[i], fontsize=14, color='black')
                
                vertices, faces = self.generate_3d_shape(shape_type, scale=0.7)                
                rotated_vertices = self.apply_random_rotation(vertices)
                
                green_color = self.colors[1][0]
                green_edge = self.colors[1][1]
                self.render_3d_shape(ax, rotated_vertices, faces, green_color, green_edge)
                
                green_shapes.append({
                    'id': alphabet[i],
                    'type': shape_type,
                    'vertices': rotated_vertices,
                    'faces': faces
                })
                
                ax.set_xlim([-1.2, 1.2])
                ax.set_ylim([-1.2, 1.2])
                ax.set_zlim([-1.2, 1.2])
                ax.set_axis_off()
                
                matchings.append((shape_idx + 1, alphabet[i]))
            
            plt.tight_layout()
            plt.suptitle(f"3D Shape Matching Challenge #{sample_idx:03d}", fontsize=16, y=0.98)
            
            sorted_matchings = sorted(matchings, key=lambda x: x[0])
            gt_text = "Ground Truth: " + ", ".join([f"{m[0]}-{m[1]}" for m in sorted_matchings])
            plt.figtext(0.5, 0.02, gt_text, ha='center', fontsize=12)
            
            output_path = os.path.join(self.output_dir, f"sample_{sample_idx:03d}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.gt_file.write(f"Sample {sample_idx:03d}:\n")
            for match in sorted_matchings:
                self.gt_file.write(f"Red {match[0]} -> Green {match[1]}\n")
            self.gt_file.write("\n")
            
            print(f"Generated sample {sample_idx:03d}")
        
        self.gt_file.close()
        print(f"Dataset generation completed. {num_samples} samples saved to {self.output_dir}")
        
        self.generate_shape_catalog()
    
    def generate_shape_catalog(self):
        """
        Generate a catalog showing all available 3D shapes.
        """
        # Determine grid dimensions
        n_shapes = len(self.shape_types)
        n_cols = min(5, n_shapes)
        n_rows = (n_shapes + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 3 * n_rows))
        
        for i, shape_type in enumerate(self.shape_types):
            # Create 3D subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            ax.set_title(shape_type, fontsize=12)
            
            vertices, faces = self.generate_3d_shape(shape_type, scale=0.7)
            
            color = "#4287f5"  # Blue
            edge_color = "#2c5aa0"  # Darker blue
            self.render_3d_shape(ax, vertices, faces, color, edge_color)
            
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.suptitle("Available 3D Shapes", fontsize=16, y=0.98)
        
        output_path = os.path.join(self.output_dir, "shape_catalog.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Shape catalog saved to {output_path}")


if __name__ == "__main__":
    generator = Shape3DBenchmarkGenerator(
        num_shapes=5,
        img_size=(1200, 800),
        colors=(("#FF5733", "#C70039"), ("#33FF57", "#0039C7")), 
        output_dir="shape3d_benchmark",
        seed=42  # For reproducibility
    )
    
    generator.generate_dataset(num_samples=10)
    
    print("3D dataset generation completed!")