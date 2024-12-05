import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from utils import SE3GNNPredictor, visualize_data_and_prediction
from utils import train_and_evaluate_model, test_model
import matplotlib.pyplot as plt

# Define split proportions
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Define the path to the saved database
# load_path = 'C:/Users/Yuri/Desktop/stanford/tmp/ModelNet_processed.pt'
load_path = r'C:\Users\Yuri\Desktop\stanford\ABC_dataset_out\ABC_processed.pt'

# Load the processed database
database = torch.load(load_path)
print(f"Processed database loaded. Number of samples: {len(database)}")

# Calculate lengths of each subset
dataset_length = len(database)  # The dataset created in previous steps
train_length = int(train_ratio * dataset_length)
val_length = int(val_ratio * dataset_length)
test_length = dataset_length - train_length - val_length

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(database, [train_length, val_length, test_length])

# Create data loaders for each set
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("the length of train_loader is ", len(train_loader)) 
print("the length of val_loader is ", len(val_loader))
print("the length of test_loader is ", len(test_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# define model
model = SE3GNNPredictor(input_dim=2, hidden_dim=32, output_dim=1, dropout_rate=0.3)
model.to(device)

### Train and evaluate the model
lr = 0.00005
num_epochs=35000
train_losses, eval_losses = train_and_evaluate_model(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=lr)

# Plot training and validation losses
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
save_path = 'C:/Users/Yuri/Desktop/stanford/ABC_dataset_out/out/'
plt.savefig(save_path)
plt.show()

# Test the model
test_model(model, test_loader, device)

# Define the path to save the trained model
save_model_path = 'C:/Users/Yuri/Desktop/stanford/ABC_dataset_out/SE3GNN_model.pth'

# Save the trained model
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

# Visualize predictions
from utils import visualize_predictions
visualize_predictions(model, test_loader, device)

# Visualize data and prediction
save_path = 'C:/Users/Yuri/Desktop/stanford/ABC_dataset_out/out'
data = next(iter(test_loader))
visualize_data_and_prediction(data, device, model, save_path)
print(f"The plots are saved seccessfully saved at {save_path}")