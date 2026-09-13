```markdown
# Spring-Mass-Damper: SMC vs. PID Control

A lightweight simulation comparing **Boundary-Layer Sliding Mode Control (SMC)** and **PID with Anti-Windup** under severe plant parameter mismatch and actuator saturation limits.

---

## Quickstart

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install numpy matplotlib

# 3. Run simulation
python main.py

```

---

## Comparison: Exact vs. Mismatched Parameters

Both experiments run with physical force limits of **±15 N** on a real plant ($m=2.0$, $c=1.0$, $k=8.0$).

| Experiment 1: Exact Parameters | Experiment 2: Severe Mismatch (~60% Error) |
| --- | --- |
|  |  |
| *Controllers tuned on true physical model.* | *Controllers tuned on $\hat{m}=0.8$, $\hat{c}=0.3$, $\hat{k}=3.0$.* |

---

## Key Takeaways

* **PID (with Clamping Anti-Windup):** Degrades significantly under parameter inaccuracy; tuning derived from a lightweight nominal model causes sluggish rise times and slow error recovery.
* **SMC (with Boundary Layer):** Retains invariant exponential settling dynamics even with a ~60% parameter estimation error, recovering onto the sliding manifold immediately after leaving actuator saturation.

---

## File Structure

* `springMassDamperModel.py` — State-space formulation of the 2nd-order mechanical plant.
* `pidController.py` — Classical PID controller with anti-windup clamping.
* `slidingModeController.py` — Boundary-layer SMC implementation ($u = u_{\text{eq}} + u_{\text{sw}}$).
* `simulator.py` — Discrete simulation loop and plotting routines.

