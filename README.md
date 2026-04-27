# Multi-Objective RL for Nuclear Fusion Optimization

Developed at Neo Hackathon 2025 (https://devpost.com/software/fusion-lab) 

Problem Statement: 
The climate crisis demands revolutionary solutions. Fusion energy could provide unlimited, clean power from seawater, with zero carbon emissions and no long-lived radioactive waste. But there is a catch: the biggest barrier to commercial fusion isn't building the reactor, it's controlling 100-million-degree plasma that can collapse in milliseconds. Traditional control systems struggle with plasma's chaotic, nonlinear behavior. We realized that reinforcement learning, could be the key to mastering plasma dynamics. This project uses RL for optimizing plasma shape control in nuclear fusion reactors using shape constraints and self-fixing mechanisms. 

Solution Overview: 
The Soft-Actor-Critic RL agent effectively learns from plasma simulation environment, leveraging TORAX which is Google Deepmind's tokamac physics engine. The SAC RL agent is compared to a PID agent (Proprotional-Integral-Derivative) industry standard and a Random agent in the gymnasium environment and performs 8x better than both agents in fusion power output (Q). 

AI Integration: 
Utilized GymTorax environment (https://arxiv.org/abs/2510.11283) along with Modal to train the agents on GPUs. Ran into difficulties trying to compress and parallelize the data for training as TORAX is too slow for online RL training. Ended up training the data one-time on TORAX and using that data for offline GPU RL. 

Architecture/Design Decisions: 
The system is structured as a modular pipeline consisting of a Next.js frontend, a Python-based API server, and a reinforcement learning optimization layer. The frontend provides an interactive interface for exploring simulation outputs, while the backend handles requests and interfaces with the RL agents and environment. Data flows from user inputs through the API into the optimization layer, where policies are evaluated and updated before results are returned for visualization.

A key design decision was to separate the frontend and backend to allow independent iteration on the user interface and the optimization logic. This also makes it easier to scale or swap components, such as replacing the RL algorithm or extending the simulation environment. Streamlit is used as a complementary visualization tool for rapid experimentation and quick inspection of agent behavior and performance metrics.

The system emphasizes modularity, with core functionality organized into reusable components (e.g., agents, utilities, and simulation environments). This structure supports experimentation with different optimization strategies and constraints without tightly coupling components.

AI Help: 
Worked thoroughly through GymTorax paper and sketching diagrams and then developed strategy with help of Claude. Used Cursor creating Next.js frontend and simulation local webpages. Tools helped speed up the process but were most helpful towards end after drawing on whiteboard the architecture for the agent flow. 

Testing/Error Handling: 
While the GymTorax environment had already defined the physics function for the environment, getting more precise results requires taking into account the vertical displacement of the plasma which affects its stability. So eventually, had to derive a computation of vertical position. Also, the agents face multiple competing objectives, maximizing fusion power and maintinaing low disruption risk, in environment so needed to ensure they find a Pareto frontier where both goals are satisfied. 


## 🚀 Quick Start

### Installation
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Run the Main Demo
```bash
# Interactive mode with shape guard visualization
python main.py
```

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser to:
- Adjust parameters (shape penalty, damping, episodes)
- Run simulations interactively
- View real-time visualizations and animations
- Analyze step-by-step data

### Animation
```bash
# Standalone animation with custom speed
python animate_shape.py 0.5  # Slow motion (0.5x speed)
python animate_shape.py 2.0  # Fast forward (2x speed)
python animate_shape.py 1.0 200  # Normal speed, 200 steps
```

## 📁 Project Structure

```
nuclear/
├── agent.py                    # Agent implementations (Random, SAC wrapper)
├── main.py                     # Main evaluation script with shape guard
├── app.py                      # Streamlit web interface
├── animate_shape.py            # Animated trajectory visualization
├── sac_shape_guard.py          # SAC agent with shape constraints
├── test_multi_env.py           # Environment benchmarking
├── modal_run.py                # Modal deployment script
│
├── eval/                       # Evaluation scripts
│   ├── rand_eval.py           # Random agent evaluation
│   └── sac_eval.py            # SAC agent evaluation
│
├── train/                      # Training scripts
│   └── sac_train.py           # SAC training pipeline
│
├── optimization-for-constraints/
│   └── shape_guard.py         # Shape constraint definitions
│
├── logs/                       # Trained model checkpoints
│   └── sac_torax_*.zip        # SAC models at various training steps
│
├── vm/                         # Cloud deployment configs
│   └── tpu_cluster.yaml       # TPU cluster configuration
│
└── results/                    # Generated visualizations (auto-created)
```

## 🎯 Features

### Shape Guard System
The shape guard monitors and corrects plasma shape violations:
- **β_N (Normalized Beta)**: Plasma pressure constraint
- **q_min (Minimum Safety Factor)**: Stability constraint
- **q95 (Edge Safety Factor)**: Edge stability constraint

### Self-Fixing Mechanism
- Detects constraint violations in real-time
- Applies corrective actions to reduce severity
- Tracks and visualizes recovery progress

### Visualization Tools
1. **Interactive Terminal Mode**: Step-by-step constraint monitoring
2. **Web Interface**: Full-featured Streamlit dashboard
3. **Animated Trajectories**: Real-time 2D/3D shape evolution
4. **Comprehensive Plots**: Multi-panel analysis with severity tracking

## 🔧 Usage Examples

### Training SAC Agent
```bash
python train/sac_train.py
```

### Evaluating Agents
```bash
# Random agent
python eval/rand_eval.py

# SAC agent
python eval/sac_eval.py
```

### Running on Modal (Cloud)
```bash
modal run modal_run.py
```

## 📊 What You'll See

### Terminal Output
```
Step 1:
  Shape: β_N=2.345, q_min=1.234, q95=4.567
  Status: 🟢 SAFE
  Severity: 0.000
  Reward: 0.150 → 0.150 (no penalty)

Step 2:
  Shape: β_N=3.456, q_min=0.890, q95=5.678
  Status: 🔴 VIOLATION
  Severity: 2.345 ↑ (was 0.000)
  Reward: 0.120 → -0.125 (penalty: -0.245)

Step 3:
  Status: 🟠 SELF-FIXING!
  Severity: 1.800 ↓ (was 2.345)
  ⭐ Corrective action! Severity reduced
```

### Generated Visualizations
- `random_baseline.png` - Baseline performance without shape guard
- `shape_self_fixing_with_shape_guard.png` - Multi-panel analysis showing:
  - Shape parameters over time
  - Severity reduction timeline
  - 2D/3D trajectory plots
  - Recovery statistics

## ⚙️ Configuration

### Shape Guard Parameters
Adjust in your scripts:
```python
agent = RandomAgent(
    action_space=env.action_space,
    shape_penalty=0.1,        # Penalty coefficient (0.0 = off)
    damp_on_violation=True,   # Reduce action magnitude on violations
    damp_factor=0.5,          # Damping strength (0.0-1.0)
)
```

### Constraint Tuning
Edit `optimization-for-constraints/shape_guard.py` to adjust:
- `BETA_N_MIN`, `BETA_N_MAX`
- `QMIN_MIN`
- `Q95_MIN`, `Q95_MAX`

## 🐛 Troubleshooting

**Rewards too negative?**
- Reduce `shape_penalty` (try 0.01 or 0.05)
- Check if initial state violates constraints

**No corrective actions shown?**
- Verify violations are decreasing in severity
- Check corrective logic in `agent.py`

**Streamlit app issues?**
- Ensure dependencies installed: `pip install streamlit pandas`
- Check that `agent.py` and `shape_guard.py` are accessible

## 📚 Dependencies

Core requirements:
- `gymtorax>=0.1.1` - Fusion reactor environment
- `torax==1.0.3` - Physics simulation
- `matplotlib>=3.8.0` - Visualization
- `numpy>=1.26.0` - Numerical computing
- `streamlit>=1.28.0` - Web interface
- `ray>=2.51.1` - Distributed computing
- `modal>=1.2.1` - Cloud deployment

See `pyproject.toml` for complete list.



