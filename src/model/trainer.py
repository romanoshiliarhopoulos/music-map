import os
import time
from h11 import Data
import torch
import torch.optim as optim
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.loader import LinkNeighborLoader
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from src.model.lightgcn_plus import LightGNN

class Trainer:
    def train_model(self, model: LightGNN, 
                    train_loader: LinkNeighborLoader,
                    num_epochs: int, 
                    lr: float, 
                    reg_lambda: float,
                    device: torch.device,
                    save_dir: str,
                    checkpoint_freq: int,
                    plot_file: str,
                    steps_per_epoch: int = 1000):
        
        epoch_losses = []
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        tqdm.write(f"--- Starting Training on {device} using LinkNeighborLoader ---")
        
        for epoch in trange(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            step_count = 0
            
            # Clear MPS cache before each epoch
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            with tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False) as pbar:

                for step_count, batch in enumerate(pbar, start=1):
                    if step_count == 0:
                        pbar.set_postfix({
                            "nodes": batch.num_nodes,
                            "edges": batch.edge_index.size(1),
                        })
                        # Debug the batch format
                        if not hasattr(batch, 'edge_label'):
                            tqdm.write("No negative sampling format detected")
                    
                    batch = batch.to(device)
                    
                    try:
                        # Run GNN on the subgraph
                        all_emb = model(batch.n_id, batch.edge_index, batch.num_nodes)
                        
                        #  Handle negative sampling 
                        if hasattr(batch, 'edge_label') and batch.edge_label is not None:
                            #edge_label contains 1s and 0s
                            pos_mask = batch.edge_label == 1
                            neg_mask = batch.edge_label == 0
                            
                            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                                # Get positive and negative edge indices
                                pos_edges = batch.edge_label_index[:, pos_mask]  # [2, num_pos]
                                neg_edges = batch.edge_label_index[:, neg_mask]  # [2, num_neg]
                                
                                # Get embeddings for positive edges
                                pos_p_emb = all_emb[pos_edges[0]]  # [num_pos, embed_dim]
                                pos_t_emb = all_emb[pos_edges[1]]  # [num_pos, embed_dim]
                                
                                # Get embeddings for negative edges  
                                neg_p_emb = all_emb[neg_edges[0]]  # [num_neg, embed_dim]
                                neg_t_emb = all_emb[neg_edges[1]]  # [num_neg, embed_dim]
                                
                                # Calculate scores
                                pos_scores = (pos_p_emb * pos_t_emb).sum(dim=1)  # [num_pos]
                                neg_scores = (neg_p_emb * neg_t_emb).sum(dim=1)  # [num_neg]
                                
                                
                            else:
                                tqdm.write("Warning: No valid pos/neg split, skipping batch")
                                continue
                            
                        # Fallback to manual (should NOT happen now)
                        else:
                            tqdm.write("Warning: Using manual negative sampling")
                            p_emb = all_emb[batch.edge_label_index[0]]
                            pos_t_emb = all_emb[batch.edge_label_index[1]]
                            
                            # Manual random negative sampling within batch
                            neg_nodes = torch.randint(0, batch.num_nodes, (batch.edge_label_index.size(1),), device=device)
                            neg_t_emb = all_emb[neg_nodes]
                            
                            pos_scores = (p_emb * pos_t_emb).sum(dim=1)
                            neg_scores = (p_emb * neg_t_emb).sum(dim=1)
                        
                        #Calculate Loss and Backpropagate
                        optimizer.zero_grad()
                        loss = model.bpr_loss(pos_scores, neg_scores, reg_lambda)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        step_count += 1
                        pbar.set_postfix({"loss": loss.item(), "nodes": batch.num_nodes})
                        
                        # Clear MPS cache periodically
                        if device.type == 'mps' and step_count % 10 == 0:
                            torch.mps.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            tqdm.write(f"OOM at step {step_count}, clearing cache and continuing...")
                            if device.type == 'mps':
                                torch.mps.empty_cache()
                            continue
                        else:
                            raise e
                    
                    # Limit steps per epoch
                    if step_count >= steps_per_epoch:
                        break
            
            avg_epoch_loss = epoch_loss / max(step_count, 1)
            epoch_losses.append(avg_epoch_loss)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "nodes": batch.num_nodes,
                "edges": batch.edge_index.size(1)
            })

            # Memory cleanup
            if device.type == 'mps':
                torch.mps.empty_cache()

            # Checkpointing
            if epoch % checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
                tqdm.write(f"Saving checkpoint to {checkpoint_path}...")
                torch.save(model.state_dict(), checkpoint_path)
                self.plot_learning_curve(epoch_losses, plot_file, save_only=True)

        tqdm.write("Training complete.")
        return epoch_losses


    def plot_learning_curve(self, losses: List[float], filepath: str, save_only: bool = False):
        """
        Plots and saves the training loss curve.
        """
        if not losses:
            tqdm.write("No loss history to plot.")
            return
            
        tqdm.write(f"Saving loss plot to {filepath}...")
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Average Epoch BPR Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        
        if not save_only:
            try:
                plt.show()
            except Exception as e:
                tqdm.write(f"Could not display plot (running in non-GUI environment?): {e}")
        
        plt.close() 
