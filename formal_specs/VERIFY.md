# Verifying the Lean proof of Theorem 1

This folder contains a Lean 4 + Mathlib formalization of the 3DM reduction in the
proof of Theorem 1 (`main.v3.tex`, Section III).

**The key idea:** Lean's kernel checks every proof step, so you do not need to
read the tactic proofs. Your job is to confirm two things:

1. **The definitions** say exactly what the paper says.
2. **The main theorem statement** says what Theorem 1 claims.

If a definition is wrong, Lean will happily prove a correct theorem about the
wrong model. That is the only way this can go wrong.

Files:

- `FormalSpecs/CPMM.lean`: the swap model and the analytic lemmas (fresh-path
  gain, the 1/10 profit cap, the q > 1/6 window, and "reusing a pool makes a
  trade unprofitable").
- `FormalSpecs/Reduction.lean`: the trade sequence model, the constructed
  instance, and the main theorem `reduction_correct`.

Line numbers below were correct when this file was written. If they have moved,
search for the name instead.

## Step 1: Get the files and build (about 15 minutes the first time)

1. Install Lean's version manager, `elan`, from <https://lean-lang.org/install>.
2. Get the Overleaf project: either `git clone` it with the Overleaf Git link
   (Overleaf menu → Integrations → Git), or download it as a ZIP from the
   Overleaf menu and unzip it. The Lean files are in `formal_specs/`.
3. In a terminal:

```bash
cd formal_specs
ulimit -n 8192
lake exe cache get
lake build
```

It should end with `Build completed successfully`. Warnings about style or the
"module system" are cosmetic. Any `error` means stop and report it.

## Step 2: Check nothing was skipped

```bash
grep -nE "sorry|admit|native_decide|axiom " FormalSpecs/*.lean
```

This must print nothing. Each of those words would mean a step was skipped or
assumed rather than proved.

## Step 3: Check the axioms

Create `Check.lean` inside `formal_specs/` containing:

```lean
import FormalSpecs
#print axioms FormalSpecs.Reduction.reduction_correct
```

Then run:

```bash
lake env lean Check.lean
```

The output must list exactly `[propext, Classical.choice, Quot.sound]`. These
are Lean's standard axioms; anything else is unacceptable. Delete `Check.lean`
afterwards.

## Step 4: Compare each definition with the paper (the important part)

Open each location and confirm it matches the paper exactly.

| Lean location | Paper | What to check |
|---|---|---|
| `CPMM.lean:19` `swapOut` | Eq. (1) | Output = `rout·α·q / (rin + α·q)`, with α = 1 − γ. `swapOut_spec` (line 23) proves it satisfies `(x + α q)(y − Δy) = xy`. |
| `CPMM.lean:30–36` `α₀`, `G`, `φ` | Proof of Theorem 1 | α₀ = 997/1000, gas G = 67/30, and φ(q) = 4q/(1+q) for an unused pool. |
| `Reduction.lean:34–44` `Pool`, `p₁ p₂ p₃` | Proof of Theorem 1 | One pool per element; X-pools trade A→B, Y-pools B→C, Z-pools C→A; triple (x,y,z) uses pools x, y, z. |
| `Reduction.lean:46–51` `State`, `init` | Proof of Theorem 1 | Every pool starts at reserves (α₀, 4), and the A-balance starts at 1/3. B and C balances are zero, so they are not stored. |
| `Reduction.lean:54–62` `o₁ o₂ o₃`, `profit` | π_m, Eq. (2) | Each leg's output feeds the next leg (C4). Profit = q₃ − q₀ − G, since all prices are 1 USD. |
| `Reduction.lean:67` `StepOK` | C1, C2, C3 | Path in catalog; 0 < q ≤ wallet (C2); 0 < each output < that pool's output reserve, strict (C3); profit > 0 (C1). |
| `Reduction.lean:75` `step` | Eq. (5) | Input reserve += amount in; output reserve −= amount out; wallet = wallet − q + q₃. Gas is not taken from the wallet. |
| `Reduction.lean:83–88` `Feasible`, `totalProfit` | Eq. (4), Π_L | Each trade must pass `StepOK` in the state left by the previous one, and Π_L is the sum of the profits. |
| `Reduction.lean:93–96` `Disj`, `IsMatching` | 3DM | Distinct triples share no x, no y and no z. |
| `Reduction.lean:356` `reduction_correct` | Theorem 1 | "A matching of size ≥ M exists ⇔ a feasible sequence with Π_L ≥ M/10 exists." |
| `Reduction.lean:384` `instance_size` | Proof of Theorem 1 | Number of pools = \|X\| + \|Y\| + \|Z\|. |

## Step 5: Confirm the modelling choices match what the paper says

Each of these is deliberate. Check that the paper states or implies it, and flag
any that it does not:

- C3 is strict (output < reserve).
- Gas is subtracted from reported profit, not from the wallet.
- The pool keeps the whole input, fee included.
- All prices are 1 USD and fixed during an episode.
- Catalog paths all start at token A.
- C6, the time limit, is left out.
- Trade amounts are real numbers.

## Step 6: Understand what is not proved in Lean

- **NP-completeness of 3DM** is cited (Karp 1972), not proved.
- **That the construction runs in polynomial time** is argued on paper, beyond
  the pool count.
- **Hardness holds only for a supplied path catalog**, not for every possible
  cycle among the pools.

The paper must not claim more than this.

## Step 7: Sanity test (recommended)

1. In `CPMM.lean:33`, temporarily change `G` from `67 / 30` to `2`.
2. Run `lake build` again. It **must fail**: with gas at 2, reusing a pool is
   no longer unprofitable.
3. Change it back and rebuild to confirm it passes again.

This shows the proof really depends on the paper's numbers and is not
vacuously true.

## Report back

- Any definition that does not match the paper.
- Any modelling choice the paper does not state.
