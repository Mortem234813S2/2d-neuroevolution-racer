# 🏎️ 2D Neuroevolution Racecar Engine

A high-performance, lightweight 2D autonomous racing simulator built from scratch using Python, `pygame-ce`, and `NumPy`. The engine combines Feedforward Neural Networks with a Genetic Algorithm (GA) to train autonomous vehicles to navigate complex track layouts in real time.

---

## ✨ Features

* **Neuroevolution Mechanics**: Implements multi-parent uniform crossover ($50/50$ weight blending), adaptive mutation scaling, elitism, and smart forward-bias initialization to eliminate early trial-and-error stagnation.
* **Continuous Raycast Sensors**: 5-directional LIDAR raycasting ($[-60^\circ, -30^\circ, 0^\circ, 30^\circ, 60^\circ]$) for distance detection against track borders.
* **Dual Operating Modes**:
  * **Solo AI Training**: Accelerated, hands-free training loop featuring active 3-second checkpoint culling.
  * **Human vs. AI Racing**: Interactive WASD manual driving mode with custom off-road grass friction, dynamic lap timers, and fair-launch countdowns.
* **Brain State Persistence**: Save and reload trained network weights (`.npz` format) to preserve top-performing models across sessions.
* **Dynamic Telemetry & Visuals**: Real-time tire skid mark overlays, fading trajectory ghost trails, orthogonal checkpoint gates, and lap record tracking.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/2d-neuroevolution-racer.git](https://github.com/YOUR_USERNAME/2d-neuroevolution-racer.git)
   cd 2d-neuroevolution-racer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the simulator**:
   ```bash
   python final_race_sim.py
   ```

---

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **`1` / `2`** | Select Mode in Main Menu |
| **`W` / `A` / `S` / `D`** | Drive Human Vehicle (Human vs. AI Mode) |
| **`F`** | Toggle 300 FPS Fast-Forward Mode |
| **`P`** | Save Current Champion Weights (`best_brain.npz`) |
| **`L`** | Load Saved Weights from File |
| **`R`** | Force New Generation Reset |
| **`ESC`** | Return to Main Menu |

---

## 🧠 Neural Architecture

```text
Inputs (6)               Hidden Layer (12)            Outputs (2)
[Ray Distance 1] ───┐                               ┌─── [Throttle (-1.0 to 1.0)]
[Ray Distance 2] ───┼─── (Fully Connected) ───┤
[Ray Distance 3] ───┼─────── [Tanh] ──────────┼─── [Steering (-1.0 to 1.0)]
[Ray Distance 4] ───┤                         │
[Ray Distance 5] ───┤                         │
[Current Speed]  ───┘                         ┘
```