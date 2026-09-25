# 2D ARC: Actor-Recurrent-Critic for UAV Target Tracking

This project implements the **Actor-Recurrent-Critic (ARC)** algorithm for UAV (Unmanned Aerial Vehicle) target tracking and path planning in 2D environments. The model combines Recurrent Neural Networks (LSTM) with Actor-Critic structures to handle sequential decision-making under partial observability.

---

## 📂 Project Structure

*   **`code/`**: Contains the complete implementation of the training, evaluation, and visualization pipelines.
    *   [`setup.py`](code/setup.py): Quick setup script to verify dependencies and create necessary output directories.
    *   [`config.py`](code/config.py): Hyperparameters and configuration settings for environments, rewards, neural networks, and training.
    *   [`environment.py`](code/environment.py): Implements the custom UAV Target Tracking simulation environment with dynamic targets and obstacles.
    *   [`networks.py`](code/networks.py): Definitions of the LSTM feature extractor, Actor network, and Critic network.
    *   [`replay_buffer.py`](code/replay_buffer.py): Experience replay buffer for episodic reinforcement learning.
    *   [`train.py`](code/train.py) / [`cli_train.py`](code/cli_train.py): Logic and CLI utilities for training ARC agents.
    *   [`test.py`](code/test.py) / [`cli_test.py`](code/cli_test.py): CLI interface to run test episodes, evaluate models, and run baselines (like APF).
    *   [`visualize.py`](code/visualize.py) / [`visualize_training.py`](code/visualize_training.py) / [`cli_visualize.py`](code/cli_visualize.py): Visualization utilities for plotting paths and generating animations/GIFs.
*   **`results/`**: Output directory for trained model weights, evaluation metrics, CSVs, and figures.
*   **`models_original/`**: The original trained 2D ARC checkpoints (300 episodes on `env_apf_urban`; `arc_best.pth` is identical to `arc_checkpoint_ep300.pth`).
*   The methodology document now lives in `docs/legacy/` (it predates the corrections; see `experiments/FINDINGS.md`).

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.7 or higher
*   PyTorch, NumPy, Matplotlib, Shapely, tqdm, TensorBoard

To check your environment and verify dependencies:

```bash
cd code
python setup.py
```

If dependencies are missing, install them via pip:
```bash
pip install torch numpy matplotlib shapely tqdm tensorboard
```

---

## 🚀 How to Run

### 1. Training
To train an ARC agent, run the command-line training script specifying the target environment and number of episodes:

```bash
python code/cli_train.py --env env7 --episodes 300
```

Available environments (`env1` to `env8`):
*   `env1` to `env3`: Stationary target tracking (Barrier-free, Simple obstacle, Complex obstacles).
*   `env4` to `env6`: Moving target tracking (Barrier-free, Simple obstacle, Complex obstacles).
*   `env7` & `env8`: Sparse and dense obstacle layouts for testing/validation.

To monitor training in real-time, launch TensorBoard:
```bash
tensorboard --logdir=runs/
```

### 2. Evaluation & Testing
To evaluate a trained model:

```bash
# Test on a specific environment
python code/cli_test.py --env env7

# Run 100 evaluation trials to measure success rates
python code/cli_test.py --env env7 --evaluate --runs 100

# Run all test environments (env1 - env6)
python code/cli_test.py --all
```

### 3. Paper experiments
The shared APF/ARC layout of the paper is the `corridor_gauntlet` environment in `code/config.py` (an exact copy of the APF Corridor Gauntlet). Training, paired evaluation against APF, and every table in the paper are produced by the scripts in `../experiments/` (see `../experiments/README.md`).
