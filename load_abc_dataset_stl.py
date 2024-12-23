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





def compute_normals(vertices, faces):
    """
    Computes normals for the given faces of the mesh.
    
    Args:
    - vertices (np.ndarray): Array of vertices of shape (N, 3).
    - faces (np.ndarray): Array of faces of shape (M, 3).

    Returns:
    - np.ndarray: Normals for each face of shape (M, 3).
    """
    normals = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)  # Normalize
        normals.append(normal)

    return np.array(normals)




def remove_duplicates(pos, edge_index, normals, faces, tolerance_degrees=2.0):
    """
    Removes duplicate nodes and edges from a graph, including bidirectional, coplanar shared edges,
    and edges shared between triangles whose normals are within a given tolerance for parallelism.

    Args:
    - pos (torch.Tensor): A tensor of shape (N, 3) representing node positions.
    - edge_index (torch.Tensor): A tensor of shape (2, E) representing edges.
    - normals (np.ndarray): A numpy array of shape (M, 3) containing face normals.
    - faces (np.ndarray): A numpy array of shape (M, 3) representing triangle face indices.
    - tolerance_degrees (float): Angle tolerance in degrees for detecting parallel or antiparallel triangles.

    Returns:
    - torch.Tensor: Updated pos with unique nodes.
    - torch.Tensor: Updated edge_index with reordered and valid edges.
    """
    # Convert tolerance to cosine
    tolerance_radians = np.radians(tolerance_degrees)
    cos_tolerance = np.cos(tolerance_radians)

    # Remove duplicate nodes
    pos_np = pos.cpu().numpy()
    unique_pos, inverse_indices = np.unique(pos_np, axis=0, return_inverse=True)

    # Update edges with new indices
    edge_index_np = edge_index.cpu().numpy()
    reindexed_edges = np.array([
        sorted([inverse_indices[edge[0]], inverse_indices[edge[1]]])  # Sort to normalize edge direction
        for edge in edge_index_np.T
        if inverse_indices[edge[0]] != inverse_indices[edge[1]]  # Avoid self-loops
    ])

    # Map edges to the triangles sharing them
    edge_to_faces = {}
    reindexed_faces = np.array([
        [inverse_indices[vertex] for vertex in face]
        for face in faces
    ])

    for i, face in enumerate(reindexed_faces):
        for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            edge_tuple = tuple(sorted(edge))
            if edge_tuple not in edge_to_faces:
                edge_to_faces[edge_tuple] = []
            edge_to_faces[edge_tuple].append(i)

    # Filter edges based on face normals and tolerance
    filtered_edges = []
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:
            normal1 = normals[face_indices[0]]
            normal2 = normals[face_indices[1]]
            dot_product = np.dot(normal1, normal2)

            # Check if the angle between normals is within the tolerance
            if cos_tolerance <= dot_product <= 1.0 or -1.0 <= dot_product <= -cos_tolerance:
                continue  # Remove edges shared by triangles with parallel/antiparallel normals
        filtered_edges.append(edge)

    # Convert to PyTorch tensors
    unique_pos_tensor = torch.tensor(unique_pos, dtype=pos.dtype, device=pos.device)
    unique_edge_index_tensor = torch.tensor(filtered_edges, dtype=edge_index.dtype, device=edge_index.device).T

    return unique_pos_tensor, unique_edge_index_tensor






def create_graph_from_STL(vertices, faces):
    """
    Creates a graph representation from STL data.

    Args:
    - vertices: A numpy array of vertices (Nx3).
    - faces: A numpy array of triangular faces (Mx3).

    Returns:
    - A PyG Data object containing the graph representation.
    """
    pos = torch.tensor(vertices, dtype=torch.float)
    
    # Compute normals for all faces
    normals = compute_normals(vertices, faces)
    
    # Create initial edge list from faces
    edges = []
    for face in faces:
        edges.extend([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Remove duplicate nodes and edges
    pos, edge_index = remove_duplicates(pos, edge_index, normals, faces, tolerance_degrees=2.0)
    edge_index = torch.Tensor(edge_index)

    # Rotate positions and prepare for visualization
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
    random_angle = np.radians(np.random.uniform(-20, 20))
    theta_x += random_angle
    R_x = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = np.radians(45)
    random_angle = np.radians(np.random.uniform(-20, 20))
    theta_y += random_angle
    R_y = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_tot = R_x @ R_y

    R_tot = R_tot.float()
    return torch.matmul(pos, R_tot)


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
        #visualize_point_cloud(graph, title="After inner edge removal")
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





