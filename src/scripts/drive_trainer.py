import torch
from torch_geometric.data import Data
import os

from tqdm import tqdm

from src.model.lightgcn_plus import LightGNN
from src.model.trainer import Trainer
from torch_geometric.loader import LinkNeighborLoader


# --- Configuration ---
# Graph Stats
NUM_PLAYLISTS = 315_220
NUM_TRACKS = 176_768
NUM_NODES = NUM_PLAYLISTS + NUM_TRACKS

# Hyperparameters
EMBEDDING_DIM = 32
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024    
STEPS_PER_EPOCH = 100
NUM_EPOCHS = 5000
REG_LAMBDA = 1e-4

# File/Script Configuration
DATA_FILE = "data/processed/train_graph.pt"
CHECKPOINT_DIR = "data/train/checkpoints"
CHECKPOINT_FREQ = 100 
PLOT_FILE = "data/train/images/loss_history_v2.png"
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def main():
    tqdm.write(f"💪 STARTING SAMPLING TRAINER FOR GNN 💪")
    
    # Keep your original large graph stats
    NUM_PLAYLISTS = 315_220
    NUM_TRACKS = 176_768  
    NUM_NODES = NUM_PLAYLISTS + NUM_TRACKS

    # Setup
    if not os.path.exists(DATA_FILE):
        tqdm.write(f"Error: Data file '{DATA_FILE}' not found.")
        return
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

    # Clear MPS cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    trainer = Trainer()

    # 2. Load Data
    tqdm.write(f"Loading data from {DATA_FILE}...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False) 
    data.num_nodes = NUM_NODES

    # 3. Prepare Data for Loader
    tqdm.write("Extracting training labels (edge_label_index)...")
    mask = data.edge_index[0] < NUM_PLAYLISTS
    edge_label_index = data.edge_index[:, mask]
    tqdm.write(f"Total training edges: {edge_label_index.size(1):,}")

    tqdm.write("Setting up MICRO LinkNeighborLoader...")
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[2, 1],          
        batch_size=64,                 # Tiny batch size  
        edge_label_index=edge_label_index,
        neg_sampling_ratio=1.0,        
        shuffle=True,
        num_workers=0,
        replace=False,                 
    )
    
    # Check if negative sampling is working
    tqdm.write("Testing negative sampling...")
    first_batch = next(iter(train_loader))
    tqdm.write(f"Batch attributes: {list(first_batch.keys())}")
    
    if hasattr(first_batch, 'edge_label') and first_batch.edge_label is not None:
        pos_count = (first_batch.edge_label == 1).sum().item()
        neg_count = (first_batch.edge_label == 0).sum().item()  
        tqdm.write(f"Automatic negative sampling working! Pos: {pos_count}, Neg: {neg_count}")
    else:
        tqdm.write("❌ No automatic negative sampling detected")
        tqdm.write("Available attributes:", [attr for attr in dir(first_batch) if not attr.startswith('_')])

    # Initialize Model  
    tqdm.write("Initializing model...")
    model = LightGNN(
        num_playlists=NUM_PLAYLISTS,
        num_tracks=NUM_TRACKS,
        embedding_dim=EMBEDDING_DIM,      
        num_layers=EMBEDDING_DIM          
    )
    
    # Run Training 
    loss_history = trainer.train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,        
        lr=LEARNING_RATE,
        reg_lambda=REG_LAMBDA,
        device=DEVICE,
        save_dir=CHECKPOINT_DIR,
        checkpoint_freq=CHECKPOINT_FREQ,
        plot_file=PLOT_FILE,
        steps_per_epoch=STEPS_PER_EPOCH
    )
    trainer.plot_learning_curve(losses=loss_history, filepath="data/train/images/train_image.png")


if __name__ == "__main__":
    main()
