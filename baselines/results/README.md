# Baseline results

Every JSON in this directory was produced on the dataset as released in this
repository. Results generated on earlier, superseded builds have been removed
rather than shipped, so numbers here match the tables in the top-level README.

| File | Contents |
|---|---|
| `task1_baselines.json` | Task 1 room labeling: DT, RF, GB, GCN, GraphSAGE |
| `task1_modern_arch.json` | GraphGPS, RGCN, Graph Transformer, GATv2 |
| `task1_architecture_ablation.json` | Layer depth, hidden width, architecture sweep |
| `task1_leakage_eval.json` | Near-duplicate leakage effect on Task 1 |
| `task1_generation.json` | Task 2 constrained generation (CVAE, CVAE+Graph) |
| `task2_cross_dataset.json` | Cross-dataset transfer, ResPlan and RPLAN |
| `task3_results.json` | Task 3 edge detection |
| `task3_edge_type.json` | Task 3 typed-edge classification |
