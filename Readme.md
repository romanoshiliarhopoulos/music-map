# Music Map

### A GNN based music recommendation system

---

## Notes on implementation
## 🏗️ System Architecture

### Phase 1: Data Architecture - Spotify Million Playlist Dataset

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

