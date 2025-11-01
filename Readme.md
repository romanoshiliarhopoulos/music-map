# Music Map

### A GNN based music recommendation system!

---

## Notes on implementation

1. Data architecure - How to get collaborative data

```
Yambda-50M Dataset (50M interactions, 850k tracks)
    ↓
Graph Construction:
    - Nodes: [Users, Songs]
    - Edges: [Listen_count × temporal_weight (something like recency bias) × organic_flag]
    - Features: [Audio_embeddings, Metadata]
    ↓
Temporal Split: Train(80%) / Val(10%) / Test(10%)

```

2. Broad GNN architecture and basic idea
```
Input Graph + Audio Features
    ↓
LightGCN++ Encoder (3 layers)
    ├─> Flexible neighbor aggregation (α, β parameters)
    ├─> Multimodal feature fusion
    └─> Adaptive layer pooling (γ parameter)
    ↓
User Embeddings + Song Embeddings
    ↓
Debiased Contrastive Loss + Explicit Negative Loss
```

3. Inference Architecture

```
Input Playlist [song_1, song_2, ..., song_n] could be one song [song_1]
    ↓
Attention-weighted Playlist Embedding
    ↓
Similarity Search in Song Embedding Space
    ↓
Top-K Recommendations (filtered)
```

---
## Data collection and Pre-Processing

Data was collected from Hugging faced and raw data is saved under `data/raw`. Data collection done through scripts located under `scripts/`.

Data preprocessing is crucial for this music recommendation system because it transforms the raw Yambda dataset into a format suitable for Graph Neural Network training. 

The preprocessor loads four Parquet files from the Yambda dataset and combines them into a unified interaction matrix:​

Each interaction type receives different weights: listens use play_count as implicit feedback, likes get a strong positive weight of 5.0, and dislikes receive a negative weight of -3.0. The system merges these with audio embeddings to create rich multimodal features.​ 

#### User and Item Filtering
Filtering dramatically reduces dataset size while preserving the most informative interactions.​

- Minimum Interaction Threshold: Only users and items with at least 10 interactions are retained to ensure meaningful patterns​

- Label Encoding: String IDs are converted to continuous integer indices using sklearn's LabelEncoder​

- Mapping Creation: Dictionaries map original IDs to encoded indices for efficient graph construction​

Final data after pre-processing: (45M interactions, 10k users ~ 850k songs)

```
Loaded preprocessed data: {'total_interactions': 45881950, 'num_users': 9984, 'num_items': 253939, 'train_size': 36705560, 'val_size': 4588195, 'test_size': 4588195, 'min_interactions': 10, 'chunk_size': 100000}
```
The data is split temporally to avoid data leakage. This ensures that the model 'does not look into the future', as what a user listens to in October is influenced by what they liked in September, creating an unusually high accuracy. 

---
## Initial Graph Build for Training
Graph statistics after initial build: 
```json
{
  "num_nodes": 263923,
  "num_users": 9984,
  "num_items": 253939,
  "num_edges": 67822208,
  "num_negative_edges": 81813,
  "avg_degree": 256.97725472959917,
  "density": 0.013375436115739889,
  "edge_weight_stats": {
    "min": 0.009999999776482582,
    "max": 5.0,
    "mean": 0.7861198782920837,
    "std": 0.7221791744232178
  }
}
```
#### Graph Structure
- 263,923 total nodes: Bipartite graph with 9,984 users and 253,939 songs after filtering.
- 67,822,208 edges: This represents ~33.9M actual interactions (since edges are bidirectional). Good data richness. 

User Engagement Patterns
Average degree of 256.97: Each node connects to ~257 others on average. For users, this means they've interacted with ~257 songs; for songs, they've been played by ~257 users. This indicates healthy engagement levels.

Density of 0.0134 (1.34%): Your graph is appropriately sparse. In recommendation systems, densities of 1-5% are typical and desirable - too dense would indicate overfitting risks, too sparse would lack collaborative filtering signals.

Interaction Quality
Edge weights (0.01 to 5.0):

Minimum 0.01: Represents songs barely listened to (1% played)

Maximum 5.0: Explicit likes (your strongest positive signal)

Mean 0.79: Most interactions are partial listens (~79% completion rate)

Standard deviation 0.72: Good variation in engagement levels

Negative Feedback
81,813 negative edges: Represents explicit dislikes. The ratio of ~1:830 (negative:positive) is realistic for music - people rarely explicitly dislike songs compared to positive interactions.

