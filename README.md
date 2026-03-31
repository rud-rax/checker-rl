# 6x6 Checkers with Actor-Critic Reinforcement Learning

A custom implementation of 6x6 Checkers using PettingZoo's AEC API, trained with Actor-Critic self-play reinforcement learning using PyTorch.

## 📁 Project Structure
```
.
├── model/                          # Model checkpoints and training outputs
│   └── V1/                        # Version 1 models
│       ├── checkpoints/           # Saved agent weights
│       ├── training_state.pkl     # Training metrics history
│       └── training_curves.png    # Training visualization
├── src/                           # Source code
│   ├── mycheckersenv.py          # Custom 6x6 Checkers environment
│   ├── myagent.py                # Actor-Critic agent implementation
│   └── myrunner.py               # Training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎮 Environment Setup (`mycheckersenv.py`)

### **6x6 Checkers Environment**

Built using PettingZoo's AEC (Agent Environment Cycle) API for turn-based games.

**Key Features:**
- **Board Size:** 6x6 grid
- **Players:** 2 (player_0 and player_1)
- **Piece Representation:**
  - `0` = Empty square
  - `1` / `-1` = Regular pieces (player_0 / player_1)
  - `2` / `-2` = King pieces (player_0 / player_1)

**Game Rules:**
- Diagonal movement only
- Mandatory jumping (must capture if possible)
- King promotion at opposite end
- Multiple jumps in one turn
- Win by capturing all pieces or blocking all moves
- Maximum 500 steps per game (prevents infinite games)

**Action Space:**
- `Discrete(1296)` - Encoded as: `from_row * 216 + from_col * 36 + to_row * 6 + to_col`
- Action masking ensures only valid moves are selected

**Observation Space:**
- `Dict` containing:
  - `observation`: 6x6 board state
  - `action_mask`: Binary mask of valid actions (1296 length)

**Reward Function (Shaped):**
- **Capture opponent piece:** +0.5
- **King promotion:** +0.3
- **Step penalty:** -0.01 (encourages efficiency)
- **Win:** +1.0
- **Loss:** -1.0
- **Draw (max steps):** 0.0

---

## 🤖 Agent Architecture (`myagent.py`)

### **Actor-Critic Neural Networks**

**Actor Network (Policy):**
```
Input (36) → FC(128) → ReLU → FC(128) → ReLU → FC(1296) → Logits
```
- **Input:** Flattened 6x6 board state (36 features)
- **Output:** Action logits (1296 possible moves)
- **Purpose:** Learns which moves to make

**Critic Network (Value Function):**
```
Input (36) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → State Value
```
- **Input:** Flattened 6x6 board state (36 features)
- **Output:** Single value (expected cumulative reward)
- **Purpose:** Evaluates how good a state is

**Training Algorithm:**
- **Policy Gradient** with advantage estimates
- **Gradient Clipping:** max_norm=1.0 (prevents instability)
- **Advantage Normalization:** Reduces variance
- **Optimizers:** Adam for both networks

**Key Methods:**
- `select_action(state, action_mask, training)` - Choose action with masking
- `get_value(state)` - Estimate state value
- `update(states, actions, advantages, returns)` - Update both networks
- `save(filepath)` / `load(filepath)` - Checkpoint management

---

## 🏃 Training Setup (`myrunner.py`)

### **Configuration**

All training parameters are defined at the top of `myrunner.py`:
```python
# MODEL CONFIG
MODEL_NAME = "V1/"                    # Version name for organizing models
SAVE_MODEL_DIR = "model/" + MODEL_NAME

ACTOR_LEARNING_RATE = 0.0001         # Learning rate for actor network
CRITIC_LEARNING_RATE = 0.0001        # Learning rate for critic network
GAMMA = 0.99                          # Discount factor

EPISODES = 10000                      # Total episodes to train
SAVE_INTERVAL = 500                   # Save checkpoint every N episodes
LOG_INTERVAL = 500                    # Print stats every N episodes

RESUME_FROM = None                    # Start fresh training
# RESUME_FROM = "model/V1/checkpoints/agent_final.pth"  # Resume from checkpoint
```

### **Self-Play Training**

- Agent plays **both sides** (player_0 and player_1)
- Learns by playing against itself
- Improves strategy over time

### **Training Process:**

1. **Episode Loop:**
   - Reset environment
   - Play full game (both players use same agent)
   - Collect experiences (states, actions, rewards, values)

2. **Update Step:**
   - Compute advantages (TD error)
   - Compute returns (discounted rewards)
   - Update actor network (policy gradient)
   - Update critic network (value function MSE)

3. **Checkpointing:**
   - Save agent weights every `SAVE_INTERVAL` episodes
   - Save training metrics (rewards, losses, lengths)
   - Generate training curves plot

---

## 🚀 Usage

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch==2.8.0`
- `pettingzoo`
- `gymnasium`
- `numpy`
- `matplotlib`

### **2. Start Training (Fresh)**
```bash
python src/myrunner.py
```

### **3. Resume Training**

Edit `myrunner.py`:
```python
# Change this line:
RESUME_FROM = "model/V1/checkpoints/agent_episode_5000.pth"
```

Then run:
```bash
python src/myrunner.py
```


**Training Curves:**
- `model/V1/training_curves.png` - 4 plots:
  - Episode Rewards
  - Episode Lengths
  - Actor Loss
  - Critic Loss

### **5. Change Model Version**

To train a new version:
```python
MODEL_NAME = "V2/"  # Creates model/V2/ directory
EPISODES = 20000    # Train longer
ACTOR_LEARNING_RATE = 0.00005  # Different learning rate
```

---

## 🎯 Model Files

**Checkpoint Structure:**

Each `.pth` file contains:
- Actor network weights
- Critic network weights
- Optimizer states (for resuming)
- Episode number
- Training metrics

**Files Generated:**
```
model/V1/
├── checkpoints/
│   ├── agent_episode_500.pth
│   ├── agent_episode_1000.pth
│   ├── ...
│   └── agent_final.pth
├── training_state.pkl       # Full training history
└── training_curves.png      # Visualization
```

---

## 🔧 Hyperparameter Tuning

**Learning Rates:**
- Start: `0.0001`
- If unstable: decrease to `0.00005` or `0.00001`
- If too slow: increase to `0.0005`

**Gamma (Discount Factor):**
- Default: `0.99` (values future rewards highly)
- Short-term focus: `0.9`
- Long-term focus: `0.995`

**Episode Limit:**
- Current: 500 steps max per game
- Adjust in `mycheckersenv.py`: `self.max_steps = 500`

---

## 📝 Implementation Details

**Actor-Critic Algorithm:**
1. Select action using policy (actor)
2. Observe reward and next state
3. Compute advantage: `A = r + γ*V(s') - V(s)`
4. Update actor: Maximize `log π(a|s) * A`
5. Update critic: Minimize `(V(s) - G)²`

**Advantage Calculation:**
- Uses TD (Temporal Difference) error
- Normalized for stability
- Reduces variance in policy gradient

**Action Masking:**
- Prevents invalid moves
- Sets invalid action logits to `-1e8`
- Ensures agent only samples legal moves


---

## 📚 References

- **PettingZoo Documentation:** https://pettingzoo.farama.org/
- **Actor-Critic Methods:** Sutton & Barto, RL Book Chapter 13
- **Policy Gradient:** https://spinningup.openai.com/en/latest/algorithms/vpg.html

---

## 👤 Author

rud-rax

