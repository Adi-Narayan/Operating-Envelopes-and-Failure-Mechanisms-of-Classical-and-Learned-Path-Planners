# UAV Target Tracking and Path Planning (ARC & APF)

This repository contains the source code, simulation environments, evaluation scripts, and research documentation for UAV path planning and target tracking. It implements two main methodology paradigms:
1.  **Actor-Recurrent-Critic (ARC)**: A deep reinforcement learning (DRL) framework using LSTM networks and TD3 to navigate and track targets under partial observability.
2.  **Artificial Potential Field (APF)**: Classical potential field models applied to static, dynamic, and 3D navigation scenarios as high-performance baselines.

---

## 📂 Repository Structure

The repository is organized as follows:

```
├── arc_2d/                 # 2D Actor-Recurrent-Critic (ARC) framework
│   ├── code/               # Simulation, networks, training, and testing scripts
│   └── README.md           # Guide for training and testing 2D ARC models
│
├── arc_3d/                 # 3D ARC framework with Ablation Studies (TD3, HER, PER)
│   ├── arc_3d.py           # Unified script containing 3D environments, agents, and visualization
│   └── README.md           # Ablation configurations and running guide for 3D ARC
│
├── static_apf/             # 2D APF implementation for static obstacle environments
│   └── static_apf.py       # Core execution script for static APF tests
│
├── dynamic_apf/            # 2D APF implementation for dynamic tracking environments
│   └── apf.py              # APF target tracking simulation script
│
├── apf_3d/                 # 3D APF implementations and unified setup scripts
│
├── APF_Methodology.docx    # Document explaining the potential field formulation
└── ARC Metrics.pdf         # Comparative evaluation metrics and results
```

---

## 🚀 Overview of Subprojects

### 1. [2D ARC (Actor-Recurrent-Critic)](file:///Users/jsreshta/Desktop/Research-Paper/arc_2d/README.md)
*   **Method**: Combines recurrent neural networks (LSTMs) with Actor-Critic architectures.
*   **Environments**: Supports 8 environments (`env1` - `env8`) ranging from simple barrier-free fields to dense obstacle zones with stationary or moving targets.
*   **Setup**: Run `python setup.py` in the `arc_2d/code/` directory to configure paths and dependencies.

### 2. [3D ARC (with Ablation Studies)](file:///Users/jsreshta/Desktop/Research-Paper/arc_3d/README.md)
*   **Method**: Combines Recurrent Actor-Critic with Twin Delayed Deep Deterministic Policy Gradient (TD3), Hindsight Experience Replay (HER), and Prioritized Experience Replay (PER) under a curriculum learning structure.
*   **Ablation**: Supports systematic evaluation of individual components (`full`, `no_td3`, `no_her`, `no_per`, `no_curriculum`).
*   **Visualization**: Produces static plots, animated GIFs, and interactive 3D HTML trajectory visualizers using Three.js.

### 3. APF Baselines (Static, Dynamic, and 3D)
*   Used to model baseline performance against DRL models.
*   The algorithms calculate attractive forces toward targets and repulsive forces away from obstacles.
*   Includes scripts in `static_apf/`, `dynamic_apf/`, and `apf_3d/` with output metrics saved in Excel spreadsheets (`.xlsx`) and trajectory GIFs.

---

## 🛠️ Getting Started & Dependencies

All scripts are written in Python 3. The general dependencies for the repository include:

*   **Deep Learning**: PyTorch (`torch`)
*   **Numerical & Geospatial**: NumPy (`numpy`), Shapely (`shapely`), pandas (`pandas`), openpyxl (`openpyxl` - for metrics sheets)
*   **Visualization**: Matplotlib (`matplotlib`), tqdm
*   **Web Viewer**: A web browser (for interactive Three.js 3D HTML output)

To install core requirements:
```bash
pip install torch numpy matplotlib shapely pandas openpyxl tqdm tensorboard
```
