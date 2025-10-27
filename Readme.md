# Music Map

### A GNN based music recommendation system!

---

## Notes on how to implement

1. Data architecure - How to get collaborative data

```
Yambda-5B Dataset (5B interactions, 9.39M tracks, 1M users)
    ↓
Graph Construction:
    - Nodes: [Users, Songs, Artists]
    - Edges: [Listen_count × temporal_weight × organic_flag]
    - Features: [Audio_embeddings, Metadata]
    ↓
Temporal Split: Train(80%) / Val(10%) / Test(10%)
```

