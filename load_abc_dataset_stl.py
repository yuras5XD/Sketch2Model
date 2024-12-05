import os
from stl import mesh
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import random_split
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

# Path to the folder containing the STL model directories
base_path = r'C:\Users\Yuri\Desktop\stanford\ABC_Dataset_Chunk_0000\stl'

# Function to load STL models
def load_stl_models(base_path, max_vertices=3000, max_models=2000):
    """
    Loads STL models from a given directory, filtering out models with more than max_vertices.
    Limits the total number of loaded models to max_models.

    Args:
      base_path: Path to the folder containing model subdirectories.
      max_vertices: Maximum allowed number of vertices for a model to be loaded.
      max_models: Maximum number of models to load.

    Returns:
      A dictionary of loaded models, each with 'vertices' and 'faces'.
    """
    model_data = {}
    model_count = 0

    for folder_name in os.listdir(base_path):
        if model_count >= max_models:
            break

        folder_path = os.path.join(base_path, folder_name)

        # Check if folder contains files
        if not os.path.isdir(folder_path):
            continue

        # Look for STL files in the folder
        stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
        if not stl_files:
            continue  # Skip empty folders

        for stl_file in stl_files:
            if model_count >= max_models:
                break

            stl_path = os.path.join(folder_path, stl_file)
            try:
                # Load the STL file
                mesh_data = mesh.Mesh.from_file(stl_path)

                # Extract vertices and faces
                vertices = mesh_data.vectors.reshape(-1, 3)
                
                # Skip models with too many vertices
                if vertices.shape[0] > max_vertices:
                    print(f"Skipping {folder_name}/{stl_file}: Too many vertices ({vertices.shape[0]}).")
                    continue

                faces = np.arange(len(vertices)).reshape(-1, 3)

                # Store the model data
                model_data[folder_name] = {
                    'vertices': vertices,
                    'faces': faces
                }

                model_count += 1
            except Exception as e:
                print(f"Error loading {stl_path}: {e}")

    print(f"Loaded {model_count} models (filtered by max {max_vertices} vertices and max {max_models} models).")
    return model_data


def remove_duplicate_nodes(pos, edge_index):
    """
    Removes duplicate nodes and updates the edge_index accordingly.

    Args:
    - pos (torch.Tensor): A tensor of shape (N, 3) representing node positions.
    - edge_index (torch.Tensor): A tensor of shape (2, E) representing edges.

    Returns:
    - torch.Tensor: Updated pos with unique nodes.
    - torch.Tensor: Updated edge_index with reordered and valid edges.
    """
    pos_np = pos.cpu().numpy()  # Convert to numpy array for comparison
    unique_pos, inverse_indices = np.unique(pos_np, axis=0, return_inverse=True)
    
    # Map original edge_index to new unique indices
    edge_index_np = edge_index.cpu().numpy()
    new_edge_index = []
    
    for edge in edge_index_np.T:
        new_start, new_end = inverse_indices[edge[0]], inverse_indices[edge[1]]
        if new_start != new_end:  # Avoid self-loops caused by duplicates
            new_edge_index.append([new_start, new_end])
    
    # Remove duplicate edges and sort
    new_edge_index = np.unique(new_edge_index, axis=0)

    # Convert back to PyTorch tensors
    unique_pos_tensor = torch.tensor(unique_pos, dtype=pos.dtype, device=pos.device)
    new_edge_index_tensor = torch.tensor(new_edge_index.T, dtype=edge_index.dtype, device=edge_index.device)
    
    return unique_pos_tensor, new_edge_index_tensor


# Assuming load_stl_models is already defined as above
def create_graph_from_STL(vertices, faces):
    """
    Creates a graph representation from STL data.

    Args:
      vertices: A numpy array of vertices (Nx3).
      faces: A numpy array of triangular faces (Mx3).

    Returns:
      A PyG Data object containing the graph representation.
    """
    pos = torch.tensor(vertices, dtype=torch.float)
    
    # Create edge list from faces
    edges = []
    for face in faces:
        v1, v2, v3 = face
        edges.extend([[v1, v2], [v2, v3], [v3, v1]])

    # Remove duplicate edges
    edges = list(set((min(e), max(e)) for e in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Remove duplicate nodes
    pos, edge_index = remove_duplicate_nodes(pos, edge_index)

    pos = rotate_pos(pos)
    true_z = pos[:, 2]
    
    return Data(pos=pos[:, :2], edge_index=edge_index, true_z=true_z)


def rotate_pos(pos):
    """
    Rotates the positional data in isometric view.

    Args:
      pos: A tensor of shape (N, 3) representing the positions.

    Returns:
      Rotated positional data.
    """

    theta_x = np.radians(35)
    R_x = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = np.radians(45)
    R_y = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_tot = R_x @ R_y

    R_tot = R_tot.float()
    return torch.matmul(pos, R_tot)


def remove_redundant_edges(data):
    edge_index = data.edge_index
    pos = data.pos
    true_z = data.true_z

    # Step 1: Get the start and end points of each edge in 3D (x, y, z)
    start_points = torch.cat([pos[edge_index[0]], true_z[edge_index[0]].unsqueeze(1)], dim=1)
    end_points = torch.cat([pos[edge_index[1]], true_z[edge_index[1]].unsqueeze(1)], dim=1)

    # Step 2: Ensure each edge is represented consistently by sorting vertices
    edge_points = torch.stack([start_points, end_points], dim=1)  # Shape: [num_edges, 2, 3]
    sorted_edges, original_order = torch.sort(edge_points, dim=1)  # Sort each edge pair along dim 1

    # Step 3: Flatten sorted edges and hash them
    flat_edges = sorted_edges.view(sorted_edges.size(0), -1)  # Flatten into [num_edges, 6]
    _, unique_indices = torch.unique(flat_edges, dim=0, return_inverse=True)

    # Step 4: Detect boundaries - edges without duplicates
    counts = torch.bincount(unique_indices)
    unique_mask = counts[unique_indices] == 1

    # Keep all unique edges and avoid boundary removal
    data.edge_index = edge_index[:, unique_mask | (counts[unique_indices] > 1)]

    return data




# Main function to load models and convert them to graphs
def prepare_stl_graphs(base_path):
    """
    Loads models from STL files and prepares graph data for training.

    Args:
      base_path: Path to the STL models.
      num_graphs: Number of graphs to prepare.
      max_vertices: Maximum number of vertices to include a model.

    Returns:
      A list of PyG Data objects.
    """
    models = load_stl_models(base_path)
    ready_database = []
    
    for model_name, data in models.items():
        vertices = data['vertices']
        faces = data['faces']
        graph = create_graph_from_STL(vertices, faces)
        # graph = remove_redundant_edges(graph)
        ready_database.append(graph)
            

    return ready_database



# Path to STL models
base_path = r'C:\Users\Yuri\Desktop\stanford\ABC_Dataset_Chunk_0000\stl'
ready_database = prepare_stl_graphs(base_path)

# Summary
print(f"Prepared {len(ready_database)} graphs for training.")

save_path = r'C:\Users\Yuri\Desktop\stanford\ABC_dataset_out\ABC_processed.pt'
torch.save(ready_database, save_path)
print(f"Processed database saved at {save_path}.")

def visualize_point_cloud(data, title="Point Cloud Visualization"):
    """
    Visualize a point cloud in 3D space, including edges.
    
    Args:
    - data: A PyG data object containing the point cloud and edge information.
    - title: Title for the plot.
    """
    if hasattr(data, 'pos') and data.pos is not None:
        # Extract node positions
        pos = data.pos.numpy()
        x, y = pos[:, 0], pos[:, 1]
        z = data.true_z.numpy()
        
        # Extract edges from edge_index
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index.numpy().T
            edge_lines = []
            for edge in edge_index:
                start = torch.cat((data.pos[edge[0]], data.true_z[edge[0]].unsqueeze(0)), dim=0)
                end = torch.cat((data.pos[edge[1]], data.true_z[edge[1]].unsqueeze(0)), dim=0)
                edge_lines.append([start, end])
        else:
            edge_lines = []

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(x, y, z, c='blue', marker='o', s=10, alpha=0.7)

        # Plot edges
        if edge_lines:
            edge_collection = Line3DCollection(edge_lines, colors='gray', linewidths=0.5, alpha=0.7)
            ax.add_collection3d(edge_collection)

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        
        plt.show()
    else:
        print("The data object does not contain positional information.")




"""
# Example usage of the remove_redundant_edges functions
# the function able to reduce the number of edges 1172 -> 780  
# but it not work well in general
# for example it can't reduce the number of edges for box 18 edges stays 18 instead of reduced to 12
data = ready_database[12]
visualize_point_cloud(data)
print(f"total edges: {len(data.edge_index[1])}")
data = remove_redundant_edges(data)
visualize_point_cloud(data)
print(f"total edges after removal: {len(data.edge_index[1])}")
"""