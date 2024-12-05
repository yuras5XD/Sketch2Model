import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import  SAGEConv
import torch.optim as optim
from e3nn.o3 import Linear as SE3Linear
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class SE3GNNPredictor(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1, dropout_rate=0.15):
        super(SE3GNNPredictor, self).__init__()
        self.dropout_rate = dropout_rate
       
        # Define SE(3)-equivariant GNN layers
        self.se3_gnn1 = SE3Linear("32x0e", "32x0e")  # First SE(3) GNN layer
        self.se3_gnn2 = SE3Linear("32x0e", "32x0e")  # Second SE(3) GNN layer

        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr='sum')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='sum')
        
        # Output MLP layers
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        self.mlp4 = torch.nn.Linear(hidden_dim, output_dim) # No activation for regression output

        # Learnable parameter alpha for skip connection
        self.alpha = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, data):
        x, edge_index = data.pos[:, :2], data.edge_index  # Use `x` and `y` as input

        # First GNN layer
        x_gnn1 = self.conv1(x, edge_index)
        x_gnn1 = F.leaky_relu(x_gnn1)
        x_gnn1 = F.dropout(x_gnn1, p=self.dropout_rate, training=self.training)
        x_gnn1 = self.se3_gnn1(x_gnn1)

        # Second GNN layer with skip connection
        x_gnn2 = self.conv2(x_gnn1, edge_index)
        x_gnn2 = F.leaky_relu(x_gnn2)  # Adjust contribution of x_gnn1 using learnable alpha
        x_gnn2 = F.dropout(x_gnn2, p=self.dropout_rate, training=self.training)
        x_gnn2 = self.se3_gnn2(x_gnn2)

        # Additional skip connection from x_gnn1 to x_gnn2
        x_skip = self.alpha * x_gnn1 + x_gnn2  # Combined output after the second GNN layer

        # Final MLP layers
        x_out = self.mlp3(x_skip)  # Pass aggregated features through mlp3
        predictions = self.mlp4(x_out)  # Final layer to produce output

        return predictions.squeeze(-1)  # Squeeze for single-dimension output
    

def train_and_evaluate_model(model, train_loader, val_loader, device, num_epochs=10,lr=0.001):
    """
    Train the GNN model on the dataset.

    Args:
    - model (torch.nn.Module): The GNN model to train.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - optimizer: Optimizer for training.
    - criterion: Loss function.
    - device (torch.device): The device to train on (CPU or GPU).
    - epochs (int): Number of epochs to train for.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - losses (list): List of average losses per epoch.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss() # Use MSE loss for regression
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            target = batch.true_z  # Ground truth `z` values

            # Compute loss
            loss = criterion(predictions, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loos_per_node = loss.item() / batch.pos.shape[0]
            epoch_loss += avg_loos_per_node

        # Record average loss for the epoch
        avg_train_loss  = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

       # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.true_z)
                val_loss_per_node = loss.item() / data.pos.shape[0]
                val_loss += val_loss_per_node
        avg_val_loss = val_loss / len(val_loader)
        eval_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, eval_losses

def test_model(model, test_loader, device):
    """
    Test the model on the test set.
    
    Args:
    - model: The trained GNN model.
    - test_loader: DataLoader for the test set.
    - criterion: Loss function.
    - device: Device ('cuda' or 'cpu').
    """
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Use Cross Entropy Loss for regression
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.true_z)
            test_loss_per_node = loss.item() / data.pos.shape[0]
            total_loss += test_loss_per_node

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


# Function to visualize predictions
def visualize_predictions(model, loader, device):
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            predictions = model(batch)
        actual = batch.true_z.cpu().numpy()
        predicted = predictions.cpu().numpy()

        # Scatter plot of actual vs predicted z-values
        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predicted, alpha=0.7, c='blue', label='Predictions')
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Ideal')
        plt.xlabel("Actual z-values")
        plt.ylabel("Predicted z-values")
        plt.legend()
        plt.title("Prediction vs Actual")
        plt.show()
        break  # Visualize one batch only


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

def visualize_point_cloud(x, y, z, edge_index, title="Point Cloud Visualization", save_path=None):
    """
    Visualize a point cloud in 3D space, including edges.

    Args:
    - x, y, z: Coordinates of points.
    - edge_index: Array of edges.
    - title: Title for the plot.
    - save_path: Path to save the plot as a PNG file. If None, the plot is not saved.
    """
    pos = np.array([x, y, z], dtype=np.float32).T
    edge_lines = [[pos[edge[0]], pos[edge[1]]] for edge in edge_index]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(x, y, z, c='blue', marker='o', s=10, alpha=0.7)

    # Plot edges
    if edge_lines:
        edge_collection = Line3DCollection(edge_lines, colors='gray', linewidths=0.5, alpha=0.7)
        ax.add_collection3d(edge_collection)

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_2d_graph(x, y, edge_index, title="2D Graph Visualization", save_path=None):
    """
    Visualizes the 2D input graph (nodes and edges in X-Y space).

    Args:
    - x, y: 2D coordinates of the nodes.
    - edge_index: Array of edges.
    - title: Title for the plot.
    - save_path: Path to save the plot as a PNG file. If None, the plot is not saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c='blue', s=10, alpha=0.7, label="Nodes")

    for edge in edge_index:
        start, end = edge
        ax.plot([x[start], x[end]], [y[start], y[end]], 'gray', linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_data_and_prediction(data, device, model, save_path=None):
    """
    Visualizes the 2D input graph, true model, and predicted model.

    Args:
    - data: The data object containing node positions and ground truth values.
    - device: The device (CPU or GPU) for running the model.
    - model: The model used for predictions.
    - save_path (str, optional): Path to save the plots. If provided, plots are saved with appropriate suffixes.
    """
    edge_index = data.edge_index.numpy().T
    pos = data.pos.numpy()
    x, y = pos[:, 0], pos[:, 1]

    # Visualize the 2D input graph
    visualize_2d_graph(x, y, edge_index, title="Input 2D Graph", save_path=f"{save_path}_input" if save_path else None)

    # Visualize the true model
    z = data.true_z.numpy()
    visualize_point_cloud(x, y, z, edge_index, title="True Model", save_path=f"{save_path}_true" if save_path else None)

    # Visualize the predicted model
    data = data.to(device)
    pred_z = model(data).cpu().detach().numpy()
    visualize_point_cloud(x, y, pred_z, edge_index, title="Predicted Model", save_path=f"{save_path}_predicted" if save_path else None)


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