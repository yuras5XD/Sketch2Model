import torch
import numpy as np
from stl import mesh
from torch_geometric.data import Data
from utils import SE3GNNPredictor
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from stl import mesh
from utils import visualize_data_and_prediction, rotate_pos
import os
from matplotlib import pyplot as plt
from matplotlib import widgets

def select_stl_file():
    """
    Open a file dialog to select an STL file.
    Returns:
        str: Path to the selected STL file.
    """
    Tk().withdraw()  # Hide the root Tkinter window
    filename = askopenfilename(title="Select STL File", filetypes=[("STL Files", "*.stl")])
    return filename

def select_stl_file_alternative():
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

    plt.ioff()  # Turn off interactive mode if running in Jupyter
    root = tk.Tk()
    root.withdraw()
    filename = askopenfilename(title="Select STL File", filetypes=[("STL Files", "*.stl")])
    root.destroy()  # Cleanly destroy window
    plt.ion()  # Turn interactive mode back on
    return filename

def stl_to_graph(stl_path):
    """
    Convert an STL file to a 2D graph.
    Args:
        stl_path (str): Path to the STL file.
    Returns:
        torch_geometric.data.Data: Graph data object.
    """
    stl_mesh = mesh.Mesh.from_file(stl_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)

    edges = []
    for triangle in indices.reshape(-1, 3):
        for i in range(3):
            edges.append([triangle[i], triangle[(i + 1) % 3]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Use x and y from vertices as 2D positions and treat z as "true_z"
    pos = torch.tensor(unique_vertices, dtype=torch.float)
    pos = rotate_pos(pos) # rotate to trimetric view
    true_z = torch.tensor(pos[:, 2], dtype=torch.float)

    graph = Data(pos=pos[:, :2], edge_index=edge_index, true_z=true_z)
    return graph


# Function to save predictions in STL format
def save_to_stl(pos, predictions, graph, filename="output.stl", threshold_factor=3.0):
    """
    Save the graph's positions and predicted z-values as an STL file, removing outlier nodes.
    Args:
        pos (torch.Tensor): Tensor of node positions (x, y).
        predictions (torch.Tensor): Predicted z-values.
        graph (torch_geometric.data.Data): Input graph containing edge_index.
        filename (str): Name of the output STL file.
        threshold_factor (float): Multiplier for the standard deviation to define outliers.
    """
    # Combine (x, y) positions with predicted z values
    points = np.column_stack((pos[:, 0].cpu().numpy(),
                              pos[:, 1].cpu().numpy(),
                              predictions.cpu().numpy()))
    
    # Identify outliers in predicted z-values
    z_mean = predictions.mean().item()
    z_std = predictions.std().item()
    z_threshold = threshold_factor * z_std
    valid_mask = torch.abs(predictions - z_mean) < z_threshold

    # Filter valid nodes and corresponding edges
    valid_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()
    points = points[valid_indices]
    
    # Update edge indices based on filtered nodes
    edge_index = graph.edge_index.cpu().numpy()
    valid_edges = np.all(np.isin(edge_index, valid_indices), axis=0)
    filtered_edges = edge_index[:, valid_edges]

    # Re-index edges to match the filtered nodes
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    reindexed_edges = np.vectorize(index_map.get)(filtered_edges)

    # Reconstruct faces from the updated edges
    faces = []
    for edge in reindexed_edges.T:
        neighbors = reindexed_edges.T[np.any(reindexed_edges == edge[0], axis=0) | 
                                      np.any(reindexed_edges == edge[1], axis=0)]
        for neighbor in neighbors:
            shared = set(edge) & set(neighbor)
            if len(shared) == 1:  # Common vertex found
                face = list(set(edge).union(neighbor))
                if len(face) == 3:
                    faces.append(face)

    # Remove duplicate faces and ensure valid indices
    faces = np.unique(np.sort(faces, axis=1), axis=0)

    # Create STL mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = points[face[j], :]

    # Save the mesh as STL
    stl_mesh.save(filename)
    print(f"STL file saved to {filename}")

# Function to load the model
def load_trained_model(model_path, device):
    model = SE3GNNPredictor(input_dim=2, hidden_dim=32, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

# Function to predict z-values
def predict_z(model, graph, device):
    graph = graph.to(device)
    with torch.no_grad():
        predictions = model(graph)
    return predictions

if __name__ == "__main__":
    model_path = 'C:/Users/Yuri/Desktop/stanford/ABC_dataset_out/SE3GNN_model.pth'
    output_file = 'C:/Users/Yuri/Desktop/stanford/tmp/output.stl'
    save_path = 'C:/Users/Yuri/Desktop/stanford/tmp/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(model_path, device)

    # Select STL file
    stl_path = select_stl_file_alternative()
    if not stl_path:
        print("No STL file selected.")
        exit()

    # Convert STL to graph
    graph = stl_to_graph(stl_path)
    print("Graph created from STL file.")

    # Predict z-values
    z_predictions = predict_z(model, graph, device)
    print("Predictions generated.")

    visualize_data_and_prediction(graph.cpu(), device, model, save_path=save_path)

    # Save predictions to STL format
    save_to_stl(graph.pos[:, :2], z_predictions, graph, filename=output_file)
