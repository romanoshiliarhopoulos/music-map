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