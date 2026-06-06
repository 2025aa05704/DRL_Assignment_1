"""Builder script that generates the two assignment Jupyter notebooks.

Run with:  python3 build_notebooks.py
It writes:
  * Team 178 - MAB.ipynb
  * Team 178 - DP.ipynb

Only the Python standard library (json) is required to build the notebooks.
The notebooks themselves require numpy/pandas/matplotlib at run time (available
in the virtual lab).
"""
import json


def code(src):
    """Wrap raw source text into a Jupyter *code* cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True) + [""]
        if src.strip("\n") else [""],
    }


def md(src):
    """Wrap raw source text into a Jupyter *markdown* cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True) + [""]
        if src.strip("\n") else [""],
    }


def notebook(cells):
    """Assemble a full nbformat-4 notebook dict."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =====================================================================
#  PART 1 - MULTI-ARMED BANDIT
# =====================================================================
mab_cells = [
    md(r"""
# Deep Reinforcement Learning - Lab Assignment 1
## Part #1 - Multi-Armed Bandit (MAB)
### Adaptive Treatment Recommendation System using Multi-Armed Bandit Learning

**Team / Group Number (G): 178**

Each medicine is modelled as an *arm* of a Multi-Armed Bandit. The system learns from
patient outcomes over time and progressively identifies the optimal medicine.

Strategies implemented:
1. **Greedy / Immediate Exploitation** (Task 2)
2. **Epsilon-Greedy / Controlled Clinical Trial** (Task 3)
3. **UCB1 / Confidence-Based** (Task 4)
4. **Comparative Analysis** (Task 5)
"""),
    md(r"""
### Execution metadata (Virtual Lab requirement)
The cell below prints the **execution timestamp** and **Virtual Machine / Host ID**.
A timestamped screenshot from the virtual lab must accompany the final submission.
"""),
    code(r"""
# Print execution timestamp and Virtual Machine ID at the top of the notebook
import datetime          # standard library for the current date/time
import socket            # used to read the machine/host name (acts as VM ID)
import platform          # extra environment details for traceability

# Current wall-clock time of execution (match this with the virtual-lab screenshot)
print('Execution Timestamp :', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# Hostname acts as the Virtual Machine identifier inside the virtual lab
print('Virtual Machine ID  :', socket.gethostname())
# Platform info for full reproducibility context
print('Platform            :', platform.platform())
print('Python Version      :', platform.python_version())
"""),
    md(r"""
## Task 1: Dataset Design (1 Mark)

The synthetic patient-treatment environment is derived from the group number `G = 178`
so the dataset is unique and reproducible.

**Derivation for G = 178**
- Medicines: `K = (G mod 3) + 5 = 1 + 5 = 6`
- Hidden success probability: `P_i = 0.4 + ((G + i) mod 6) * 0.07`
- Severity: `Severity = (patient_id mod 5) + 1`  (1 = mild ... 5 = critical)
- Utility (reward): `utility = clinical_outcome * (1 - Severity/10)`
"""),
    code(r"""
# ---- Core imports ----
import random                       # python RNG (seeded for reproducibility)
import numpy as np                  # vectorised maths and RNG
import pandas as pd                 # tabular dataset handling
import matplotlib.pyplot as plt     # plotting cumulative-reward curves

# ---- Reproducibility: seed both RNGs with the group number ----
G = 178                             # our group number
random.seed(G)                      # seed python's random module
np.random.seed(G)                   # seed numpy's global RNG

# ---- Environment parameters derived from G ----
K = (G % 3) + 5                     # number of medicines (arms)
# Hidden success probability for each medicine i in {0,...,K-1}
true_probs = np.array([0.4 + ((G + i) % 6) * 0.07 for i in range(K)])

N_PATIENTS = 1000                   # total patients (iterations) to simulate

print('Group number G     :', G)
print('Total medicines K  :', K)
print('Hidden success probabilities (P_i):')
for i, p in enumerate(true_probs):
    print(f'  Medicine {i}: P = {p:.2f}')
print(f'\nTrue best medicine : {int(np.argmax(true_probs))} (P = {true_probs.max():.2f})')
"""),
    code(r"""
# ---- Build the static part of the dataset (patient_id + severity_score) ----
# assigned_medicine / clinical_outcome / utility_score are populated *dynamically*
# by each algorithm at run time, so the base table only holds the fixed fields.
patient_ids = np.arange(N_PATIENTS)             # 0 .. 999
severity_scores = (patient_ids % 5) + 1         # severity in range 1..5

dataset = pd.DataFrame({
    'patient_id': patient_ids,                  # patient index
    'severity_score': severity_scores,          # disease severity (1-5)
    'assigned_medicine': np.nan,                # filled during an algorithm run
    'clinical_outcome': np.nan,                 # filled during an algorithm run
    'utility_score': np.nan                     # filled during an algorithm run
})

print('First 10 dataset rows:')
print(dataset.head(10).to_string(index=False))
"""),
    code(r"""
def administer_treatment(medicine, severity):
    # Simulate giving `medicine` to a patient of the given `severity`.
    # Returns:
    #   clinical_outcome (int)  : 1 if patient recovers else 0  (Bernoulli with P_i)
    #   utility_score   (float) : reward = clinical_outcome * (1 - severity/10)
    # clinical_outcome is used to UPDATE the bandit estimates;
    # utility_score is used to ACCUMULATE the cumulative reward.
    clinical_outcome = 1 if np.random.random() < true_probs[medicine] else 0
    utility_score = clinical_outcome * (1 - severity / 10)
    return clinical_outcome, utility_score
"""),
    md(r"""
## Task 2: Immediate Exploitation Strategy (1 Mark)

**Policy:** *"Once a treatment appears best, keep prescribing only that treatment."*

- **Warm-up:** test each medicine exactly **10 times** (`10 * K` patients).
- **Exploit:** afterwards always pick the medicine with the highest estimated recovery
  rate (`argmax Q`).
"""),
    code(r"""
def run_greedy(initial_pulls=10):
    # Greedy / immediate-exploitation strategy over N_PATIENTS patients.
    # Each medicine is tried `initial_pulls` times (warm-up); then the best-so-far
    # medicine is always selected.
    np.random.seed(G)                       # re-seed: every strategy faces same outcomes
    Q = np.zeros(K)                         # estimated recovery rate per medicine
    N = np.zeros(K)                         # number of times each medicine was used
    df = dataset.copy()                     # private copy of the dataset to populate
    cumulative = np.zeros(N_PATIENTS)       # cumulative reward after each patient
    best_hist = np.zeros(N_PATIENTS, int)   # running best-estimated arm after each patient
    total = 0.0

    for t in range(N_PATIENTS):
        severity = int(df.at[t, 'severity_score'])
        if t < initial_pulls * K:           # warm-up: round-robin over all medicines
            arm = t % K
        else:                               # exploit: pick best estimated medicine
            arm = int(np.argmax(Q))
        outcome, utility = administer_treatment(arm, severity)
        N[arm] += 1                         # update usage count
        Q[arm] += (outcome - Q[arm]) / N[arm]   # incremental sample-average update
        total += utility                    # accumulate utility reward
        df.at[t, 'assigned_medicine'] = arm
        df.at[t, 'clinical_outcome'] = outcome
        df.at[t, 'utility_score'] = utility
        cumulative[t] = total
        best_hist[t] = int(np.argmax(Q))    # which medicine currently looks best
    return {'name': 'Greedy', 'df': df, 'Q': Q, 'N': N,
            'cumulative': cumulative, 'best_hist': best_hist, 'total': total}

greedy_res = run_greedy(initial_pulls=10)
print('Greedy estimated recovery rates Q:', np.round(greedy_res['Q'], 3))
print('Greedy pulls per medicine        :', greedy_res['N'].astype(int))
print('Greedy chosen best medicine      :', int(np.argmax(greedy_res['Q'])))
print(f"Greedy cumulative reward (1000)  : {greedy_res['total']:.2f}")
print('\nFirst 10 populated rows:')
print(greedy_res['df'].head(10).to_string(index=False))
"""),
    md(r"""
## Task 3: Controlled Clinical Trial - Epsilon-Greedy (1.5 Marks)

**Policy:** mostly give the current best treatment, but with probability `epsilon`
explore a random treatment to discover hidden opportunities.

Main run uses **epsilon = 0.10**; we also analyse **0.01** and **0.50**.
"""),
    code(r"""
def run_epsilon_greedy(epsilon):
    # Epsilon-greedy strategy: explore a random medicine with probability `epsilon`,
    # otherwise exploit the medicine with the highest estimated recovery rate.
    np.random.seed(G)                       # same outcome stream for a fair comparison
    Q = np.zeros(K)
    N = np.zeros(K)
    df = dataset.copy()
    cumulative = np.zeros(N_PATIENTS)
    best_hist = np.zeros(N_PATIENTS, int)
    total = 0.0

    for t in range(N_PATIENTS):
        severity = int(df.at[t, 'severity_score'])
        if np.random.random() < epsilon:    # explore: random medicine
            arm = np.random.randint(K)
        else:                               # exploit: best estimated medicine
            arm = int(np.argmax(Q))
        outcome, utility = administer_treatment(arm, severity)
        N[arm] += 1
        Q[arm] += (outcome - Q[arm]) / N[arm]
        total += utility
        df.at[t, 'assigned_medicine'] = arm
        df.at[t, 'clinical_outcome'] = outcome
        df.at[t, 'utility_score'] = utility
        cumulative[t] = total
        best_hist[t] = int(np.argmax(Q))
    return {'name': f'Eps-Greedy ({epsilon:.0%})', 'df': df, 'Q': Q, 'N': N,
            'cumulative': cumulative, 'best_hist': best_hist, 'total': total}

eps10_res = run_epsilon_greedy(0.10)        # main run: 10% exploration
print('Epsilon = 10%')
print('  Estimated Q       :', np.round(eps10_res['Q'], 3))
print('  Pulls per medicine:', eps10_res['N'].astype(int))
print(f"  Cumulative reward : {eps10_res['total']:.2f}")
print('  First 10 populated rows:')
print(eps10_res['df'].head(10).to_string(index=False))
"""),
    code(r"""
# ---- Sensitivity analysis: epsilon = 1%, 10%, 50% ----
eps01_res = run_epsilon_greedy(0.01)        # very little exploration
eps50_res = run_epsilon_greedy(0.50)        # heavy exploration

print('Effect of exploration rate on final cumulative reward:')
for res in (eps01_res, eps10_res, eps50_res):
    best = int(np.argmax(res['Q']))
    print(f"  {res['name']:>18s} -> reward = {res['total']:7.2f} | best medicine = {best}")

print('\nObservation:')
print('  * 1%  -> too little exploration; risks locking onto a sub-optimal medicine.')
print('  * 10% -> balanced; quickly finds the best medicine and mostly exploits it.')
print('  * 50% -> too much exploration; wastes ~half the patients on inferior medicines.')
"""),
    md(r"""
## Task 4: Confidence-Based Strategy - UCB1 (1 Mark)

Select the arm maximising the UCB1 index:

$$ a_t = \arg\max_i \left( Q_i + \sqrt{\frac{2\ln t}{N_i}} \right) $$

Each medicine is tried once first so every `N_i > 0` before the bonus is used.
"""),
    code(r"""
def run_ucb1():
    # UCB1 confidence-based strategy. Medicines with fewer observations receive a
    # larger exploration bonus that shrinks as more evidence is collected.
    np.random.seed(G)
    Q = np.zeros(K)
    N = np.zeros(K)
    df = dataset.copy()
    cumulative = np.zeros(N_PATIENTS)
    best_hist = np.zeros(N_PATIENTS, int)
    total = 0.0

    for t in range(N_PATIENTS):
        severity = int(df.at[t, 'severity_score'])
        if t < K:                           # initialise: try each medicine once
            arm = t
        else:                               # pick the arm with the largest UCB1 index
            ucb_values = Q + np.sqrt(2 * np.log(t + 1) / N)
            arm = int(np.argmax(ucb_values))
        outcome, utility = administer_treatment(arm, severity)
        N[arm] += 1
        Q[arm] += (outcome - Q[arm]) / N[arm]
        total += utility
        df.at[t, 'assigned_medicine'] = arm
        df.at[t, 'clinical_outcome'] = outcome
        df.at[t, 'utility_score'] = utility
        cumulative[t] = total
        best_hist[t] = int(np.argmax(Q))
    return {'name': 'UCB1', 'df': df, 'Q': Q, 'N': N,
            'cumulative': cumulative, 'best_hist': best_hist, 'total': total}

ucb_res = run_ucb1()
print('UCB1 estimated recovery rates Q:', np.round(ucb_res['Q'], 3))
print('UCB1 pulls per medicine        :', ucb_res['N'].astype(int))
print('UCB1 chosen best medicine      :', int(np.argmax(ucb_res['Q'])))
print(f"UCB1 cumulative reward (1000)  : {ucb_res['total']:.2f}")
print('\nFirst 10 populated rows:')
print(ucb_res['df'].head(10).to_string(index=False))
"""),
    md(r"""
## Task 5: Comparative Analysis (0.5 Mark)

Plot **Cumulative Reward vs. Number of Patients** for every strategy.
"""),
    code(r"""
# Compare every strategy on one cumulative-reward plot
strategies = [greedy_res, eps01_res, eps10_res, eps50_res, ucb_res]

plt.figure(figsize=(10, 6))
x = np.arange(1, N_PATIENTS + 1)            # patient index on the x-axis
for res in strategies:                      # one curve per strategy
    plt.plot(x, res['cumulative'], label=f"{res['name']} ({res['total']:.1f})")
plt.xlabel('Number of Patients')
plt.ylabel('Cumulative Reward (utility)')
plt.title('Cumulative Reward vs. Number of Patients (Group 178)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('Final cumulative reward ranking:')
for res in sorted(strategies, key=lambda r: r['total'], reverse=True):
    print(f"  {res['name']:>18s} : {res['total']:7.2f}")
"""),
    md(r"""
### Quantitative answers computed from the run

The cell below derives the answers to Questions 1-3 **directly from this run's data** so
the stated conclusions always match the numbers above.

- **Highest cumulative reward** = strategy with the largest final `total`.
- **Fastest convergence** = earliest patient index after which the running best-estimated
  medicine no longer changes (i.e. it has locked onto its final choice).
- **Most stable** = smallest standard deviation of the per-patient reward over the last
  200 patients (fewest fluctuations).
"""),
    code(r"""
true_best = int(np.argmax(true_probs))      # the genuinely optimal medicine (=1)

def convergence_point(res):
    # Earliest patient index after which the running best-estimated arm stays constant.
    bh = res['best_hist']
    final = bh[-1]
    conv = 0
    for t in range(N_PATIENTS - 1, -1, -1):
        if bh[t] == final:
            conv = t
        else:
            break
    return conv

def stability(res):
    # Std-dev of per-patient reward over the last 200 patients (lower = more stable).
    increments = np.diff(np.concatenate([[0.0], res['cumulative']]))
    return float(np.std(increments[-200:]))

print('Strategy            | final reward | best medicine | converged@ | stability(std)')
for res in strategies:
    print(f"  {res['name']:>16s} | {res['total']:11.2f} | "
          f"{int(np.argmax(res['Q'])):^13d} | {convergence_point(res):^10d} | "
          f"{stability(res):.4f}")

highest = max(strategies, key=lambda r: r['total'])
fastest = min(strategies, key=convergence_point)
most_stable = min(strategies, key=stability)
print('\nQ1 Highest cumulative reward :', highest['name'])
print('Q2 Fastest convergence       :', fastest['name'])
print('Q3 Most stable performance   :', most_stable['name'])
print('   True best medicine        :', true_best,
      '(P =', round(float(true_probs[true_best]), 2), ')')
"""),
    md(r"""
### Analysis Questions & Comparative Summary

Questions 1-3 are answered by the computed cell above (so they always match this run).
The discussion below interprets those results for **G = 178**.

**1. Highest cumulative reward at 1000 patients?**
See `Q1` above - for this run it is **Greedy (~520)**. With a clean round-robin warm-up
(10 pulls each) the true best medicine (medicine 1, P = 0.75) stands out clearly, so Greedy
locks onto it and then spends *every* remaining patient exploiting it - wasting nothing on
exploration. Epsilon-Greedy (10%) is a close second; Epsilon-Greedy (50%) is lowest because
it keeps prescribing random (often inferior) medicines.

**2. Fastest convergence?**
See `Q2` above - here it is **Epsilon-Greedy (50%)**, with **UCB1** essentially tied.
Heavy/structured exploration samples every arm very early, so the running best-estimated
medicine settles quickly. (Convergence = earliest patient after which the estimated-best
arm no longer changes.)

**3. Most stable performance?**
See `Q3` above - **Greedy** is the most stable: once it commits to a single medicine the
only remaining variation is that medicine's intrinsic Bernoulli/severity noise, giving the
smoothest curve. Epsilon-Greedy (50%) fluctuates the most due to constant random switching.

**4. Safest approach for real-world hospital deployment?**
**UCB1.** It is deterministic (no random prescriptions on real patients), gives principled
extra chances to under-tested treatments, converges quickly, and offers a confidence-based
justification for every choice - all valuable when patient safety matters. (Pure Greedy
"won" on reward here only because of a lucky, clean warm-up; on a different seed a poor
warm-up can permanently lock it onto a sub-optimal medicine, which is unacceptable
clinically.)

**Comparative summary.**
On this dataset Greedy achieves the highest cumulative reward and the most stable curve
because the warm-up cleanly identifies medicine 1 and it then exploits without waste.
Epsilon-Greedy trades a little reward for robustness: 10% performs well, 1% risks getting
stuck, and 50% wastes too many patients on inferior medicines. UCB1 converges early and
evaluates every treatment fairly; although its raw reward is slightly lower here, its
determinism and built-in confidence guarantees make it the recommended, safest choice for
a real clinical setting.
"""),
]


# =====================================================================
#  PART 2 - DYNAMIC PROGRAMMING (DRONE RESCUE)
# =====================================================================
dp_cells = [
    md(r"""
# Deep Reinforcement Learning - Lab Assignment 1
## Part #2 - Dynamic Programming (DP)
### Autonomous Drone Rescue using Value Iteration

**Team / Group Number (G): 178**

We model an autonomous rescue drone as a finite **Markov Decision Process (MDP)** and
solve it with **Value Iteration** to obtain the optimal value function `V*(s)` and the
optimal policy `pi*(s)`.

**Configuration derived from G = 178 (last digit 8):**
- Last digit 8 is in 5-9  -> **6x6 grid**, **3 rescue targets, 2 charging, 4 danger, 3 blocked**
- Last digit 8 is even     -> **maximum battery = 10**
- Last digit 8 is in 5-9  -> **wind probability = 30%**
- Max step limit (6x6)     -> **75 steps** (used for episode simulation)
"""),
    code(r"""
# Print execution timestamp and Virtual Machine ID at the top of the notebook
import datetime
import socket
import platform

print('Execution Timestamp :', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Virtual Machine ID  :', socket.gethostname())
print('Platform            :', platform.platform())
print('Python Version      :', platform.python_version())
"""),
    md(r"""
## 1. Custom Drone Rescue Environment (1 Mark)

### Grid configuration & symbol placement
Placement is generated deterministically from `G` (so it is reproducible and unique).
The **start S is fixed at the top-left corner (0,0)**. Remaining special cells are drawn
from a `G`-seeded shuffle of all other cells.

| Symbol | Meaning |
|--------|---------|
| S | Start position (0,0) |
| F | Free / safe cell |
| D | Dangerous zone |
| R | Rescue target |
| C | Charging station |
| W | Wind zone (stochastic movement) |
| X | Blocked cell / obstacle |

> The assignment specifies counts for rescue/charging/danger/blocked cells but not for
> wind zones, so we add **2 wind zones** (documented assumption) to exercise the
> stochastic-transition logic.
>
> The `+5` "reach charging station" reward is treated as a **one-time discovery bonus per
> station** (the battery still refills on every visit). Without this, repeatedly stepping
> onto a charger would form an infinite positive-reward loop and the optimal policy would
> farm the charger forever instead of rescuing - so a per-station "discovered" flag is
> added to the state (documented assumption, analogous to a rescue target paying out once).
"""),
    code(r"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

G = 178                                  # group number
random.seed(G)
np.random.seed(G)

# ---- Parameters derived from the last digit of G (=8) ----
last_digit = G % 10
N = 6 if last_digit >= 5 else 5          # grid size (6x6 for digit 8)
MAX_BATTERY = 10 if last_digit % 2 == 0 else 15   # 10 (even last digit)
WIND_P = 0.30 if last_digit >= 5 else 0.20        # wind disturbance probability
MAX_STEPS = 75 if N == 6 else 50         # episode step cap (for simulation)

if last_digit >= 5:
    n_targets, n_charge, n_danger, n_block = 3, 2, 4, 3
else:
    n_targets, n_charge, n_danger, n_block = 2, 1, 3, 2
n_wind = 2                                # documented assumption

# ---- Deterministic placement of special cells (seeded by G) ----
placer = random.Random(G)                # dedicated RNG so placement is reproducible
free_cells = [(r, c) for r in range(N) for c in range(N) if (r, c) != (0, 0)]
placer.shuffle(free_cells)

rescue_targets = free_cells[0:n_targets]
charging_cells = free_cells[n_targets:n_targets + n_charge]
danger_cells   = free_cells[n_targets + n_charge:n_targets + n_charge + n_danger]
blocked_cells  = free_cells[n_targets + n_charge + n_danger:
                            n_targets + n_charge + n_danger + n_block]
wind_cells     = free_cells[n_targets + n_charge + n_danger + n_block:
                            n_targets + n_charge + n_danger + n_block + n_wind]

# Sets for O(1) membership tests
charging_set = set(charging_cells)
danger_set   = set(danger_cells)
blocked_set  = set(blocked_cells)
wind_set     = set(wind_cells)
# Map each rescue-target cell to its index in the rescued-status tuple
target_index = {cell: i for i, cell in enumerate(rescue_targets)}
# Map each charging cell to its index in the charged-status tuple (one-time +5 bonus)
charge_index = {cell: i for i, cell in enumerate(charging_cells)}

START = (0, 0)
GAMMA = 0.95                              # discount factor
THETA = 1e-3                              # value-iteration stopping threshold

print(f'Grid size        : {N} x {N}')
print(f'Max battery      : {MAX_BATTERY}')
print(f'Wind probability : {WIND_P}')
print(f'Max steps/episode: {MAX_STEPS}')
print(f'Rescue targets   : {rescue_targets}')
print(f'Charging cells   : {charging_cells}')
print(f'Danger cells     : {danger_cells}')
print(f'Blocked cells    : {blocked_cells}')
print(f'Wind cells       : {wind_cells}')
"""),
    code(r"""
# ---- Action set and movement vectors ----
MOVES = ['Up', 'Down', 'Left', 'Right']           # the four movement actions
ACTIONS = MOVES + ['Hover']                        # full action space
DELTA = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}

def base_symbol(r, c):
    # Return the static map symbol for a cell (ignores the drone and rescue status).
    if (r, c) == START:        return 'S'
    if (r, c) in blocked_set:  return 'X'
    if (r, c) in charging_set: return 'C'
    if (r, c) in danger_set:   return 'D'
    if (r, c) in wind_set:     return 'W'
    if (r, c) in target_index: return 'R'
    return 'F'

def move_cell(r, c, direction):
    # Apply a movement direction. The drone stays put if the target cell is
    # off-grid or blocked (the move still consumes battery, handled by the caller).
    dr, dc = DELTA[direction]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < N and 0 <= nc < N):   # off-grid -> stay
        return (r, c)
    if (nr, nc) in blocked_set:             # blocked -> stay
        return (r, c)
    return (nr, nc)

def valid_actions(state):
    # Return all valid actions from a state. Hover is always valid; a movement is
    # valid only if the adjacent cell is on the grid (moving into a blocked cell is
    # allowed but keeps the drone in place).
    r, c, b, resc, chg = state
    acts = ['Hover']
    for d in MOVES:
        dr, dc = DELTA[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            acts.append(d)
    return acts

# Print the symbolic grid layout
print('Initial grid layout:')
for r in range(N):
    print(' '.join(base_symbol(r, c) for c in range(N)))
"""),
    md(r"""
### State representation & transition dynamics

**State** `s = (row, col, battery, rescued, charged)` where `rescued` is a tuple of
booleans (one per rescue target) and `charged` is a tuple of booleans (one per charging
station, marking whether its one-time `+5` bonus has already been paid). This captures the
drone position, remaining battery, which targets are still pending, and which chargers are
already discovered - the information needed for the Markov property.

**Rewards** (mutually exclusive per step, matching the assignment table):
rescue target `+20`, danger zone `-10`, charging station `+5` (first visit only),
regular movement `-1`, battery exhausted `-20` (added when battery hits 0).

**Battery:** every action costs 1 unit; entering a charging station refills to full;
hovering on a charging station adds `+2` (capped at max); battery `0` ends the episode.

**Blocked / off-grid moves:** if a movement would leave the grid or enter a blocked cell
the drone stays put and the step counts as a regular move (`-1`, battery `-1`); it does
**not** re-collect the current cell's entry reward.

**Wind:** when the drone is on a wind cell and chooses a movement, with probability
`WIND_P` the realised direction is replaced by a uniform random one of the four moves
(so the intended direction keeps probability `1 - WIND_P + WIND_P/4`).
"""),
    code(r"""
def transitions(state, action):
    # Return the list of (probability, next_state, reward, done) tuples for taking
    # `action` in `state`. Assumes `state` is non-terminal.
    r, c, b, resc, chg = state
    out = defaultdict(float)              # accumulate probability over identical outcomes

    def add(p, nstate, reward, done):
        out[(nstate, reward, done)] += p

    # ---------- HOVER ----------
    if action == 'Hover':
        if (r, c) in charging_set:        # hovering on a charger adds +2 battery
            nb = min(MAX_BATTERY, b + 2)
            add(1.0, (r, c, nb, resc, chg), 0.0, False)
        else:                             # hovering elsewhere costs 1 battery like a move
            nb = b - 1
            done = (nb == 0)
            reward = -1 + (-20 if done else 0)
            add(1.0, (r, c, nb, resc, chg), reward, done)
        return [(p, ns, rew, dn) for (ns, rew, dn), p in out.items()]

    # ---------- MOVEMENT (with optional wind disturbance) ----------
    if (r, c) in wind_set:                # build the (possibly randomised) direction dist.
        dist = {}
        for d in MOVES:
            p = WIND_P / 4 + ((1 - WIND_P) if d == action else 0.0)
            dist[d] = dist.get(d, 0.0) + p
    else:
        dist = {action: 1.0}

    for d, p in dist.items():
        nr, nc = move_cell(r, c, d)       # resolve movement (handles walls/blocked)
        nb = b - 1                        # movement consumes 1 battery
        resc2 = list(resc)
        chg2 = list(chg)
        cell = (nr, nc)
        # Mutually-exclusive event reward for the entered cell
        if (nr, nc) == (r, c):
            # Move blocked / off-grid: the drone stays put. Per the assignment this only
            # consumes battery as a regular move - it must NOT re-trigger the current
            # cell's entry reward (e.g. re-charging by bumping a wall while on a charger).
            reward = -1
        elif cell in charging_set:
            nb = MAX_BATTERY             # charging always refills battery to full
            ci = charge_index[cell]
            if not chg[ci]:              # +5 incentive paid only the FIRST time a station
                reward = 5               # is reached (one-time, like a rescue target);
                chg2[ci] = True          # mark this charger as already discovered
            else:
                reward = 0               # revisiting a known charger: refill only, no bonus
        elif cell in danger_set:
            reward = -10                 # danger penalty (does NOT terminate)
        elif cell in target_index and not resc[target_index[cell]]:
            reward = 20                  # rescue reward
            resc2[target_index[cell]] = True   # target now rescued (cell becomes free)
        else:
            reward = -1                  # regular movement cost
        done = False
        if nb == 0:                      # battery exhausted -> terminal with penalty
            reward += -20
            done = True
        resc2 = tuple(resc2)
        chg2 = tuple(chg2)
        if all(resc2):                   # all targets rescued -> terminal (success)
            done = True
        add(p, (nr, nc, nb, resc2, chg2), reward, done)

    return [(p, ns, rew, dn) for (ns, rew, dn), p in out.items()]
"""),
    code(r"""
class DroneRescueEnv:
    # Gym-style environment exposing reset(), step(action) and render().
    def __init__(self):
        self.reset()

    def reset(self):
        # Start at the top-left corner with a full battery, no targets rescued and
        # no charging stations discovered yet.
        self.r, self.c = START
        self.b = MAX_BATTERY
        self.resc = tuple([False] * n_targets)
        self.chg = tuple([False] * n_charge)
        self.steps = 0
        return self.state()

    def state(self):
        return (self.r, self.c, self.b, self.resc, self.chg)

    def step(self, action):
        # Sample one outcome from the transition distribution of the current state.
        trans = transitions(self.state(), action)
        rnd = random.random()
        acc = 0.0
        chosen = trans[-1]
        for p, ns, rew, done in trans:
            acc += p
            if rnd <= acc:
                chosen = (p, ns, rew, done)
                break
        _, ns, reward, done = chosen
        self.r, self.c, self.b, self.resc, self.chg = ns
        self.steps += 1
        if self.steps >= MAX_STEPS:       # enforce the episode step cap
            done = True
        return ns, reward, done, {}

    def render(self):
        # Print the grid with the drone marked as 'A'; rescued targets shown as 'F'.
        print(f'Battery={self.b}  Rescued={self.resc}  Steps={self.steps}')
        for r in range(N):
            row = []
            for c in range(N):
                if (r, c) == (self.r, self.c):
                    row.append('A')                       # drone (agent)
                elif (r, c) in target_index and self.resc[target_index[(r, c)]]:
                    row.append('F')                       # already-rescued target
                else:
                    row.append(base_symbol(r, c))
            print(' '.join(row))
        print()

# Quick sanity demo of the environment dynamics
env = DroneRescueEnv()
env.reset()
print('Initial environment state:')
env.render()
for a in ['Right', 'Down', 'Hover']:
    s, rew, done, _ = env.step(a)
    print(f'Action={a:5s} -> reward={rew}, done={done}')
env.render()
"""),
    md(r"""
## 2. Dynamic Programming Solution - Value Iteration (2 Marks)

We enumerate all reachable states, then apply Value Iteration with discount
`gamma = 0.95` and stopping threshold `theta = 1e-3`, reporting the number of iterations,
runtime, and the final delta.
"""),
    code(r"""
def enumerate_states():
    # Enumerate every non-terminal state: a non-blocked position, battery in 1..MAX,
    # any rescued-status combination that is not "all rescued" (terminal), and any
    # charging-discovered combination (which chargers have already paid their +5 bonus).
    states = []
    rescued_combos = []
    for mask in range(2 ** n_targets):    # all subsets of rescued targets
        combo = tuple(bool(mask & (1 << i)) for i in range(n_targets))
        rescued_combos.append(combo)
    charged_combos = []
    for mask in range(2 ** n_charge):     # all subsets of already-discovered chargers
        combo = tuple(bool(mask & (1 << i)) for i in range(n_charge))
        charged_combos.append(combo)
    for r in range(N):
        for c in range(N):
            if (r, c) in blocked_set:     # drone can never occupy a blocked cell
                continue
            for b in range(1, MAX_BATTERY + 1):   # battery 0 is terminal -> excluded
                for combo in rescued_combos:
                    if all(combo):        # all-rescued is terminal -> excluded
                        continue
                    for ccombo in charged_combos:
                        states.append((r, c, b, combo, ccombo))
    return states

all_states = enumerate_states()
print(f'Number of non-terminal states: {len(all_states)}')
"""),
    code(r"""
def value_iteration(gamma=GAMMA, theta=THETA):
    # Standard Value Iteration. V for terminal states stays 0 (default dict value).
    V = defaultdict(float)
    iterations = 0
    start_time = time.time()
    final_delta = 0.0
    while True:
        delta = 0.0
        for s in all_states:
            v_old = V[s]
            best = float('-inf')
            for a in valid_actions(s):           # Bellman optimality backup
                q = 0.0
                for p, ns, reward, done in transitions(s, a):
                    q += p * (reward + (0.0 if done else gamma * V[ns]))
                best = max(best, q)
            V[s] = best
            delta = max(delta, abs(v_old - best))
        iterations += 1
        final_delta = delta
        if delta < theta:                        # converged
            break
    runtime = time.time() - start_time
    return V, iterations, runtime, final_delta

V_star, n_iter, runtime, last_delta = value_iteration()
print(f'Converged in       : {n_iter} iterations')
print(f'Runtime            : {runtime:.3f} seconds')
print(f'Final delta/error  : {last_delta:.6f} (threshold = {THETA})')
"""),
    code(r"""
def greedy_policy(V, gamma=GAMMA):
    # Derive the optimal policy: the action maximising the one-step Bellman backup.
    policy = {}
    for s in all_states:
        best_a, best_q = None, float('-inf')
        for a in valid_actions(s):
            q = 0.0
            for p, ns, reward, done in transitions(s, a):
                q += p * (reward + (0.0 if done else gamma * V[ns]))
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy

pi_star = greedy_policy(V_star)
# Show the optimal action from the start state: full battery, nothing rescued,
# no chargers discovered yet.
start_state = (0, 0, MAX_BATTERY, tuple([False] * n_targets), tuple([False] * n_charge))
print('Optimal action at start state', start_state, '->', pi_star[start_state])
print(f'V*(start) = {V_star[start_state]:.3f}')
"""),
    md(r"""
## 3. Policy Visualisation (1 Mark)

For a fixed slice (full battery, no targets rescued yet) we draw the optimal action at
each cell as an arrow (movements) or a dot (hover), overlaid on the labelled grid.
"""),
    code(r"""
# Arrow vectors for plotting movement directions (note: image y-axis points down)
ARROW = {'Up': (0, 0.3), 'Down': (0, -0.3), 'Left': (-0.3, 0), 'Right': (0.3, 0)}
fixed_resc = tuple([False] * n_targets)
fixed_chg = tuple([False] * n_charge)
fixed_batt = MAX_BATTERY

fig, ax = plt.subplots(figsize=(7, 7))
for r in range(N):
    for c in range(N):
        sym = base_symbol(r, c)
        # Colour-code the special cells for readability
        color = {'X': '#444444', 'D': '#ff9999', 'C': '#99ccff',
                 'R': '#99ff99', 'W': '#ffe699', 'S': '#dddddd'}.get(sym, 'white')
        ax.add_patch(plt.Rectangle((c - 0.5, N - 1 - r - 0.5), 1, 1,
                                   facecolor=color, edgecolor='black'))
        ax.text(c, N - 1 - r + 0.32, sym, ha='center', va='center',
                fontsize=10, fontweight='bold')
        if sym == 'X':                      # no policy arrow on blocked cells
            continue
        s = (r, c, fixed_batt, fixed_resc, fixed_chg)
        a = pi_star.get(s)
        if a == 'Hover':
            ax.plot(c, N - 1 - r, 'ko', markersize=5)   # dot = hover
        elif a in ARROW:
            dx, dy = ARROW[a]
            ax.arrow(c, N - 1 - r, dx, dy, head_width=0.12,
                     head_length=0.12, fc='blue', ec='blue')

ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(-0.5, N - 0.5)
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_title('Optimal Policy (full battery, no targets rescued)')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
"""),
    md(r"""
## 4. State-Value Analysis (1 Mark)

We fix the rescue status (none rescued) and battery (full) and vary only the drone
position, plotting a heatmap of `V*(s)`.
"""),
    code(r"""
value_grid = np.full((N, N), np.nan)        # NaN for blocked cells (shown blank)
for r in range(N):
    for c in range(N):
        if (r, c) in blocked_set:
            continue
        value_grid[r, c] = V_star[(r, c, fixed_batt, fixed_resc, fixed_chg)]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(value_grid, cmap='viridis')
plt.colorbar(im, ax=ax, label='V*(s)')
for r in range(N):                          # annotate each cell with symbol + value
    for c in range(N):
        sym = base_symbol(r, c)
        txt = f'{sym}\n{value_grid[r, c]:.1f}' if not np.isnan(value_grid[r, c]) else 'X'
        ax.text(c, r, txt, ha='center', va='center', color='white', fontsize=8)
ax.set_title('State-Value Heatmap V*(s)  (full battery, none rescued)')
ax.set_xticks(range(N))
ax.set_yticks(range(N))
plt.tight_layout()
plt.show()

print('Observation: cells closer to rescue targets (R) have higher V*, while cells near')
print('danger zones (D) and far from any target have lower V*. Charging cells (C) keep')
print('value high by protecting against battery exhaustion, and blocked cells are absent.')
"""),
    md(r"""
## 5. DP Scalability Discussion (1 Mark)

**Curse of Dimensionality.** The state space is the product of all state variables:
`positions x battery levels x 2^(#targets) x 2^(#chargers)`. For our 6x6 grid this is
roughly `33 x 10 x 2^3 x 2^2` states. Each variable we add multiplies the size.

**How the state space grows:**
- **10x10 grid:** positions jump from ~33 to ~100, roughly **3x** more states (and the
  reachable region grows), increasing both memory and per-sweep cost.
- **More rescue targets:** the rescued-status component is `2^(#targets)`, so each extra
  target **doubles** the state space (exponential growth).
- **Dynamic weather:** adding a wind/weather variable with `w` settings multiplies the
  state space by `w` and makes transitions time-dependent, breaking the stationary
  assumption DP relies on.

**Is DP sufficient?** DP gives an exact optimal policy and is ideal here because the MDP
is small and fully known. But it requires (a) enumerating every state and (b) a known
transition model. As dimensions grow, enumeration becomes infeasible (memory + time
explode) and the model is often unknown in the real world.

**How Deep RL helps.** Methods like DQN / PPO replace the explicit value table with a
neural-network function approximator that generalises across similar states, learn
directly from sampled experience (no full model needed), and handle continuous or
high-dimensional states - exactly the regime where tabular DP fails.

**Real-world relation.** A real rescue drone faces continuous positions, noisy sensors,
unknown dynamics and changing weather; tabular DP cannot scale to this, so Deep RL (or
model-based RL with learned models) is used in practice, with DP-style value iteration
serving as the theoretical foundation and a baseline on small instances.
"""),
]


with open('Team 178 - MAB.ipynb', 'w') as f:
    json.dump(notebook(mab_cells), f, indent=1)
with open('Team 178 - DP.ipynb', 'w') as f:
    json.dump(notebook(dp_cells), f, indent=1)
print('Notebooks written successfully.')
