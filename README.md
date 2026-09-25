# Operating Envelopes of Classical and Learned Path Planners

Code, trained models and evaluation logs for the paper *Operating Envelopes of Classical and Learned Path Planners* (submitted to Robotics and Autonomous Systems).

It compares two planners:
- a goal-scaled **Artificial Potential Field (APF)** planner with a fixed rotational deflection term;
- an **Actor-Recurrent-Critic (ARC)** agent, an LSTM-augmented DDPG policy.

They are compared on one shared 2D layout under one evaluation protocol: the same start, goal radius, collision criterion, travel budget and ground speed. APF is also characterised on dynamic, multi-layout and 3D workspaces.

---

## Repository structure

```
├── experiments/            # Everything that produces the paper's numbers and figures
│   ├── README.md           #   how to reproduce each table and figure
│   ├── FINDINGS.md         #   verification log: what was checked, found and corrected
│   ├── harness.py          #   shared APF/ARC evaluation protocol
│   ├── *.py                #   training, paired evaluation, sweeps, timing, report, figures
│   └── out/                #   JSON results, report.md, figures, trained ARC seeds
│
├── static_apf/             # 2D APF on five static layouts (static_apf.py + results/)
├── dynamic_apf/            # 2D APF with moving obstacles (apf.py)
├── apf_3d/                 # 3D APF (core in create_project_unified.sh, Flask front end)
├── arc_2d/                 # 2D ARC: environments, networks, training/testing code,
│   ├── code/               #   original results and the original trained checkpoints
│   └── models_original/
├── arc_3d/                 # 3D ARC (TD3 + HER + PER + curriculum), Google Colab notebook code
└── docs/legacy/            # Development-era methodology documents (superseded by the paper)
```

## Reproducing the paper

See [`experiments/README.md`](experiments/README.md). In short:

```bash
pip install numpy scipy torch shapely matplotlib tqdm tensorboard openpyxl psutil
cd experiments
python verify_apf_static.py          # static APF, Table 6, rotation sweep
python run_apf3d.py                  # Table 7 (APF, 3D)
python apf_in_arc3d.py               # Table 9 (APF in the 3D ARC arena)
python train_arc_seed.py --seed 0    # one ARC seed on the shared layout (~2 h, CPU)
python evaluate.py --layout corridor --models out/arc_models/seed*/models/arc_best.pth --tag final_corridor_a
python speed_sweep.py --layout corridor --models out/arc_models/seed*/models/arc_best.pth
python timing.py --models <successful ARC checkpoints>
python report.py && python figures.py --tag final_corridor_a
```

All 2D and APF results in the paper were produced on one laptop (AMD Ryzen 7 5800HS, 16 GB RAM, RTX 3050 4 GB; 2D ARC trained on the CPU). The 3D ARC development in `arc_3d/` was run on Google Colab.

## Subprojects

- **[2D ARC](arc_2d/README.md).** LSTM + DDPG agent for UAV navigation and target tracking. The paper's shared layout is `corridor_gauntlet` in `arc_2d/code/config.py`.
- **[3D ARC](arc_3d/README.md).** TD3, Hindsight Experience Replay, Prioritised Experience Replay and curriculum learning, with ablation switches.
- **APF baselines.**
  - `static_apf/static_apf.py`: five static layouts, path plots, GIFs and force heatmaps in `static_apf/results/`.
  - `dynamic_apf/apf.py`: seeded dynamic runs.
  - `apf_3d/`: 3D static and dynamic.

## Dependencies

Python 3.12, NumPy, SciPy, PyTorch, Shapely, Matplotlib, tqdm, TensorBoard (logging only), openpyxl and psutil. A CPU build of PyTorch is sufficient for everything in the paper.
