# Music Map

### A GNN based music recommendation system

---

## Notes on implementation

# 🏗️ System Architecture

### Data Architecture - Spotify Million Playlist Dataset

The Spotify Million Playlist Dataset (MPD) consists of **1 million user-created playlists** from US Spotify users (January 2010 - November 2017), containing:

- **2,262,292 unique tracks**
- **295,860 unique artists**
- **734,684 unique albums**
- **66+ million track occurrences**
- Average playlist length: **66.35 tracks**

**Data Flow:**

```
Spotify MPD (1M playlists, JSON format)
    ↓
Graph Construction:
    - Nodes: [Playlists, Tracks]
    - Edges: [Co-occurrence × position_weight × edit_session]
    - Features: [Audio_features via Spotify API, Metadata]
    ↓
Temporal Split: Train(80%) / Val(10%) / Test(10%)
```

### Pre-processing and Initial Graph Construction

The original dataset had to be reduced for memory reasons and after pre-processing data we build an initial graph with the following:

```
--- Graph Statistics ---
  Total Nodes: 491,988
    - Playlists: 315,220
    - Tracks: 176,768
  Total Edges (bidirectional): 30,992,538
  Actual Interactions (unidirectional): 15,496,269
  Max Possible Interactions: 55,720,808,960
  Bipartite Sparsity: 0.00027811 (or 0.027811 %)

--- Node Degree Statistics ---
  Degree Mean: 62.9945
  Degree Std: 197.6243
  Degree Min: 0
  Degree Max: 11986

  Playlist Node Degree:
    - Mean: 49.1602
    - Min: 0
    - Max: 250

  Track Node Degree:
    - Mean: 87.6644
    - Min: 0
    - Max: 11986
```

This is enough for our GNN to be trained on as we have more that 15M interactions and a total of 180k songs after the initial filtering, that filtered out very rare and niche songs with less than 5 playlist including them.

# 🧠 Model Architecture: LightGNN

This project uses a LightGNN (Light Graph Convolutional Network), a model designed specifically for collaborative filtering. It was chosen because it simplifies traditional GNNs, allowing it to focus only on the graph's structure.

The model's goal is to learn an embedding for every playlist and track. A node's final embedding is a combination of its own unique embedding and the embeddings of its neighbors.

Initialization: A single, large embedding table is created for all 491,988 nodes.

Propagation: The model uses PyTorch Geometric's LGConv layers to perform message passing. In each layer, a node's embedding is updated by averaging the embeddings of its neighbors.

Aggregation: After passing through num_layers (2), the final embedding for any node is the mean of its embeddings from all layers. This process captures multi-hop relationships (e.g., "tracks liked by playlists that also liked...").

The entire model is just this propagation logic stacked on top of a single nn.Embedding lookup table.

# 🏋️ Training

The model's embeddings are optimized using the Bayesian Personalized Ranking (BPR) loss function.

Instead of predicting a rating, BPR learns to rank items. Its goal is to make the model score a positive track (one that is in a playlist) higher than a negative track (one that is not in the playlist).

Sampling: The graph is too large to train on all at once. We use PyTorch Geometric's LinkNeighborLoader to manage this. At each step, the loader:

- Grabs a small batch of positive (playlist, track) edges.
- Samples an equal number of negative (playlist, random_track) edges.
- Creates a subgraph containing just the nodes needed for that batch, and feeds it to the model.
- The model was trained ntil the BPR loss converged.

![Loss over time](https://github.com/romanoshiliarhopoulos/music-map/blob/main/data/train/images/loss_history_v2.png)


# 💡 Inference & Recommendation
Once trained, the model's embedding.weight is a static lookup table representing the entire music space. The recommend.py script uses this table to generate cold-start recommendations.

Recommendation Flow:

- Input: A user provides one or more "seed" track IDs.
- Lookup: The script retrieves the pre-trained embedding vectors for these seed tracks.
- Aggregate: It calculates the mean of these vectors. This creates a new playlist embedding that represents the average embeddings of the seed list.
- Score: This new prototype embedding is multiplied  against the embeddings of all 176,768 tracks in the dataset.
- Rank: The tracks with the highest dot-product scores (the closest vectors in the embedding space) are returned as the Top-K recommendations.
