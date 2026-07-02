# 3D ARC: Actor-Recurrent-Critic with Ablation Studies

This folder contains the 3D implementation of the **Actor-Recurrent-Critic (ARC)** algorithm for UAV path planning and target tracking in complex 3D environments, including support for ablation studies.

The implementation includes wind models, Signed Distance Field (SDF) representation of 3D obstacles (spheres, boxes, and cylinders), and curriculum learning stages.

---

## 📂 Project Structure

*   **[`arc_3d_500_ablation (1).py`](file:///Users/jsreshta/Desktop/Research-Paper/arc_3d/arc_3d_500_ablation%20(1).py)**: The main script containing:
    *   **Environment (`UAVEnvironmentPro`)**: 3D UAV target tracking environment with wind dynamics, 3D obstacles, collision margins, and Line-of-Sight (LOS) checks.
    *   **Models**: LSTM Encoder for sequence feature extraction, along with Actor and Critic network architectures.
    *   **Agent (`TD3Agent`)**: Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning agent.
    *   **Memory**: Prioritized Replay Buffer (PER) and Hindsight Experience Replay (HER).
    *   **Training & Testing Orchestration**: `MultiUAVTrainer` supports stage-based curriculum learning.
*   **[`final_trajectory.png`](file:///Users/jsreshta/Desktop/Research-Paper/arc_3d/final_trajectory.png)**: Static visualization of the final UAV flight path around obstacles.
*   **[`training_curves.png`](file:///Users/jsreshta/Desktop/Research-Paper/arc_3d/training_curves.png)**: Plots of training metrics (rewards, critic loss, success rates, and curriculum stage progress).
*   **[`trajectory_viewer.html`](file:///Users/jsreshta/Desktop/Research-Paper/arc_3d/trajectory_viewer.html)**: Interactive 3D trajectory viewer powered by Three.js/WebGL (open in any web browser to rotate and inspect the path).

---

## 🔬 Ablation Configuration

To analyze the contribution of each component, the script supports five ablation modes:

| Mode | TD3 | HER (Hindsight Experience Replay) | PER (Prioritized Replay) | Curriculum Learning | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`"full"`** | Yes | Yes | Yes | Yes | Baseline configuration (all components enabled). |
| **`"no_td3"`** | No | Yes | Yes | Yes | Single critic, no delayed policy updates, no target smoothing. |
| **`"no_her"`** | Yes | No | Yes | Yes | No hindsight relabeling of failed episodes. |
| **`"no_per"`** | Yes | Yes | No | Yes | Uniform experience replay sampling. |
| **`"no_curriculum"`**| Yes | Yes | Yes | No | Starts training directly in the hard stage (Stage 2). |

To configure a run, modify the parameters at the bottom of the script:
```python
ABLATION_MODE = "full"     # Choices: "full", "no_td3", "no_her", "no_per", "no_curriculum"
SEED          = 0          # Seed for reproducibility (e.g., 0, 1, 2)
NUM_EPISODES  = 500        # Number of episodes to run training
```

Each run automatically creates a dedicated output folder (e.g., `abl_full_seed0/`) containing saved model weights (`arc_multi_best.pt`), training CSV logs, and final figures.

---

## 🚀 Execution

You can run the script using standard Python commands. 

### 1. Run Complete Ablation Study / Training
To run the complete training workflow configured by the ablation parameters:
```bash
python "arc_3d_500_ablation (1).py"
```

### 2. Run Test / Evaluation Only
To load a trained model checkpoint and run testing/evaluation:
```python
from arc_3d_500_ablation import run_test

# Evaluates the model on test episodes and produces trajectory plots
run_test(model_path="abl_full_seed0/arc_multi_best.pt")
```

### 3. Regenerate Visualizations
To regenerate static figures, GIFs, and the interactive HTML viewer from a saved trajectory `.npy` file:
```python
from arc_3d_500_ablation import run_visualise

run_visualise(npy_path="abl_full_seed0/test_trajectory_final.npy")
```
