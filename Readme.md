# Music Map

### A GNN based music recommendation system!

---

## Notes on how to implement

1. Data architecure - How to get collaborative data

```
Yambda-5B Dataset (5B interactions, 9.39M tracks, 1M users)
    ↓
Graph Construction:
    - Nodes: [Users, Songs]
    - Edges: [Listen_count × temporal_weight × organic_flag]
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
Top-K Recommendations (filtered & diversified)
```