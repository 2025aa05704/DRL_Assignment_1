# DRL Lab Assignment 1 — Study Notes

A self-study guide so you can build this assignment (MAB + Dynamic Programming) on your
own. Read top-to-bottom; each section maps to a part of the assignment.

---

## 0. Big picture: what is Reinforcement Learning (RL)?

An **agent** interacts with an **environment**. At each step it observes a **state**,
takes an **action**, and receives a **reward**. The goal is to learn a **policy** (a rule
for choosing actions) that **maximises total reward over time**.

- **Multi-Armed Bandit (MAB)** = the *simplest* RL problem: there is only one state
  (or the state does not matter for the decision). You repeatedly choose among `K`
  actions ("arms") and only learn about the arm you pulled.
- **Dynamic Programming (DP)** = a *planning* method for the *full* RL problem (many
  states) **when you already know the environment's rules** (transition probabilities and
  rewards). It computes the optimal policy exactly.

This assignment is two halves: Part 1 = MAB, Part 2 = DP on a grid-world MDP.

---

## PART 1 — Multi-Armed Bandit (MAB)

### 1.1 Core idea
Imagine `K` slot machines (arms). Each arm `i` pays reward with an unknown probability
`Pᵢ`. You want to find and play the best arm as much as possible, but you only discover an
arm's quality by trying it. This is the **exploration vs. exploitation dilemma**:
- **Exploit** = play the arm that currently looks best (earn reward now).
- **Explore** = try other arms (gain information that may pay off later).

In the assignment, each **medicine = an arm**, each **patient = one pull**, and recovery
(0/1) is the random payout.

### 1.2 Action-value estimation
We estimate each arm's value `Q(i)` = average reward seen so far for arm `i`.

- Sample average: `Q(i) = (sum of rewards from arm i) / (times arm i was chosen)`.
- **Incremental update** (no need to store all rewards) — memorise this:

  ```
  N(i) ← N(i) + 1
  Q(i) ← Q(i) + (reward − Q(i)) / N(i)
  ```

  This is "new estimate = old estimate + step_size × (target − old estimate)", a pattern
  you will see again and again in RL.

> Assignment subtlety: update `Q` using **clinical_outcome (0/1)**, but accumulate the
> **utility_score** (`outcome × (1 − severity/10)`) for the "cumulative reward" you report.

### 1.3 The three strategies you must implement

**(a) Greedy / pure exploitation (Task 2)**
- Try each arm a few times to initialise (the assignment says 10 times each), then
  **always** pick `argmax Q`.
- Risk: if the warm-up was unlucky, it can lock onto a sub-optimal arm forever (no more
  exploration). Good to *demonstrate* this failure mode.

**(b) ε-greedy (Task 3)**
- With probability `ε` pick a **random** arm (explore); otherwise pick `argmax Q`
  (exploit).
- `ε = 0.1` is a common balance. Study the effect:
  - `ε` too small (1%): barely explores → may get stuck early.
  - `ε` too large (50%): explores so much it wastes pulls on bad arms → low total reward.

**(c) UCB1 — Upper Confidence Bound (Task 4)**
- Pick the arm maximising an **optimism-under-uncertainty** score:

  ```
  UCB(i) = Q(i) + sqrt( 2 · ln(t) / N(i) )
  ```

  where `t` = current step, `N(i)` = times arm `i` was used.
- The square-root term is an **exploration bonus**: large when an arm has few samples,
  shrinking as evidence grows. This automatically balances explore/exploit without a
  random coin flip → usually fast and stable. Try each arm once first so `N(i) > 0`.

### 1.4 How to judge strategies (Task 5)
- **Cumulative reward**: sum of rewards up to patient `t`; plot vs. `t`.
- **Regret** (good to know): reward you *lost* by not always playing the best arm.
- **Convergence speed**: how soon the strategy's "current best arm" stops changing.
- **Stability**: how smooth the reward curve is (low variance late = stable).

### 1.5 What to actually study for Part 1
- Sutton & Barto, *Reinforcement Learning: An Introduction* — **Chapter 2** (entire).
- Concepts: exploration vs exploitation, action values, incremental averaging, ε-greedy,
  UCB, (optional) optimistic initial values, gradient bandits.
- Python: drawing a Bernoulli outcome with `np.random.random() < p`, seeding RNGs with
  `random.seed()` / `np.random.seed()` for reproducibility, `pandas` DataFrame basics,
  `matplotlib` line plots.

---

## PART 2 — Dynamic Programming on an MDP (Drone Rescue)

### 2.1 Markov Decision Process (MDP) — the vocabulary
An MDP is defined by `(S, A, P, R, γ)`:
- **S** — set of **states**. Here a state = `(row, col, battery, rescued-status-of-each-target)`.
- **A** — **actions**: Up, Down, Left, Right, Hover.
- **P(s′ | s, a)** — **transition probability** of landing in `s′` after action `a` in `s`.
  Most moves are deterministic, but **wind cells** make movement stochastic.
- **R(s, a, s′)** — **reward** (rescue +20, danger −10, charging +5, move −1, battery
  dead −20).
- **γ (gamma)** — **discount factor** (0–1). Future rewards are worth `γ^k`. We used 0.95.

**Markov property**: the next state depends only on the *current* state and action, not on
the full history. That is why battery and rescue-status must be *inside* the state — the
decision depends on them.

### 2.2 Policy and value functions
- **Policy `π(s)`** — which action to take in each state.
- **State-value `V^π(s)`** — expected total discounted reward starting from `s` under `π`.
- **Action-value `Q^π(s, a)`** — same, but if you first take action `a`.

### 2.3 The Bellman equations (the heart of DP)
**Bellman expectation** (value of following policy `π`):

```
V^π(s) = Σ_a π(a|s) Σ_s′ P(s′|s,a) [ R + γ V^π(s′) ]
```

**Bellman optimality** (value of acting optimally):

```
V*(s) = max_a Σ_s′ P(s′|s,a) [ R + γ V*(s′) ]
```

The optimal policy is then "act greedily w.r.t. `V*`":
`π*(s) = argmax_a Σ_s′ P(s′|s,a) [ R + γ V*(s′) ]`.

### 2.4 Two DP algorithms (pick one — we used Value Iteration)

**Value Iteration**
1. Initialise `V(s) = 0` for all states.
2. Repeat: for every state, set `V(s) ← max_a Σ P·[R + γV(s′)]` (a "Bellman backup").
3. Track `delta` = the largest change in any `V(s)` this sweep.
4. **Stop when `delta < θ`** (here `θ = 1e-3`). Report iterations, runtime, final delta.
5. Extract `π*` by taking the greedy action in each state.

**Policy Iteration** (alternative)
- Alternate **policy evaluation** (solve `V^π` for the current policy) and **policy
  improvement** (make the policy greedy w.r.t. `V^π`) until the policy stops changing.

Both converge to the same optimum; Value Iteration is simpler to code.

### 2.5 Modelling the environment (what `step()` must do)
For a chosen action, compute the next state and reward by applying, in order:
1. **Wind**: if on a wind cell and moving, with prob `WIND_P` replace the direction with a
   uniformly random one of the four moves.
2. **Movement**: off-grid or into a **blocked** cell → stay put (still costs 1 battery).
3. **Battery**: every action −1; entering a **charging** cell refills to max; **hover on
   charger** adds +2.
4. **Cell effects/reward**: rescue (+20, target removed), danger (−10, not terminal),
   charging (+5), else regular move (−1).
5. **Termination**: battery hits 0 (−20), all targets rescued, or step cap reached.

`reset()` returns the start state (top-left, full battery, nothing rescued).
`render()` prints the grid with the drone marked and rescued targets cleared.

### 2.6 Analysis you must produce
- **Policy visualisation**: arrows showing the best action per cell (for a fixed battery &
  rescue-status slice).
- **State-value heatmap**: fix battery/rescue-status, vary position, colour by `V*`.
  Explain: high near rescue targets/chargers, low near danger/far cells.
- **Scalability / curse of dimensionality**: the state count =
  `positions × battery-levels × 2^(#targets)`. Each new variable *multiplies* the size;
  each extra rescue target *doubles* it. So DP becomes infeasible on big/continuous/
  dynamic problems → motivates **Deep RL** (neural nets approximate `V`/`Q` and learn from
  samples without a known model).

### 2.7 What to actually study for Part 2
- Sutton & Barto — **Chapter 3** (MDPs, returns, value functions, Bellman equations) and
  **Chapter 4** (Dynamic Programming: policy evaluation, policy iteration, value
  iteration).
- Concepts: state design, discount factor, stochastic transitions, convergence threshold,
  curse of dimensionality.
- Python: nested loops over a state space, dictionaries for `V` and `π`, `matplotlib`
  `imshow` for heatmaps and `arrow` for policy plots, timing with `time.time()`.

---

## 3. Suggested study order (fast path)

1. **Watch lectures CS1–CS5** (RL basics, bandits, MDPs, DP) and the webinars.
2. **MAB**: Sutton & Barto Ch.2 → implement greedy, ε-greedy, UCB on a toy 3-arm bandit.
3. **MDP/DP**: Sutton & Barto Ch.3–4 → implement value iteration on a tiny 3×3 grid first,
   then extend to the drone problem with battery + rescue status in the state.
4. **Plotting**: practice cumulative-reward line plots and a value heatmap.
5. **Reproducibility**: always seed RNGs; print group-derived parameters.

## 4. Key formulas cheat-sheet

```
Incremental value update : Q ← Q + (r − Q)/N
ε-greedy                 : random arm w.p. ε, else argmax Q
UCB1                     : argmax_i [ Q(i) + sqrt(2·ln t / N(i)) ]
Return (discounted)      : G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ...
Bellman optimality       : V*(s) = max_a Σ_s′ P(s′|s,a)[ R + γ V*(s′) ]
Value Iteration stop     : stop when max_s |V_new(s) − V_old(s)| < θ
```

## 5. Recommended resources
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — free PDF online.
  Chapters 2, 3, 4 cover everything in this assignment.
- David Silver's RL lecture series (YouTube) — Lectures 1–3 (MDPs & DP), and the bandits
  material.
- Your course lectures CS1–CS5 and webinar demos (primary, exam-aligned).

---

*Tip:* Once you understand the formulas above, re-read the two notebooks
(`Team 178 - MAB.ipynb`, `Team 178 - DP.ipynb`) cell by cell — every function has comments
mapping it back to these concepts.
