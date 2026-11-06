import Mathlib

/-!
# CPMM lemmas for Theorem 1 (3DM reduction)

Analytic lemmas behind the NP-hardness reduction in Section III of
`main.v3.tex`. The swap model is the fee-adjusted constant-product rule of
Eq. (1): with fee multiplier `α = 1 - γ`, an input `q` against reserves
`(rin, rout)` yields `out` with `(rin + α q)(rout - out) = rin * rout`, and the
pool keeps the full input.

The reduction instance uses `α = 997/1000`, fresh directional reserves
`(α, 4)`, gas `G = 67/30` per transaction, and starting balance `1/3`.
-/

namespace FormalSpecs.CPMM

/-- Fee-adjusted CPMM output for input `q` against reserves `(rin, rout)`. -/
noncomputable def swapOut (α rin rout q : ℝ) : ℝ :=
  rout * (α * q) / (rin + α * q)

/-- The output satisfies the fee-adjusted constant-product rule, Eq. (1). -/
theorem swapOut_spec (α rin rout q : ℝ) (h : rin + α * q ≠ 0) :
    (rin + α * q) * (rout - swapOut α rin rout q) = rin * rout := by
  unfold swapOut
  field_simp
  ring

/-- Fee multiplier used in the reduction. -/
noncomputable def α₀ : ℝ := 997 / 1000

/-- Gas cost per transaction in the reduction. -/
noncomputable def G : ℝ := 67 / 30

/-- Output map of an unused reduction pool with reserves `(α₀, 4)`. -/
noncomputable def φ (q : ℝ) : ℝ := 4 * q / (1 + q)

/-- An unused pool implements `φ`. -/
theorem swapOut_fresh (q : ℝ) (hq : 0 ≤ q) : swapOut α₀ α₀ 4 q = φ q := by
  unfold swapOut φ α₀
  have h1 : (1 : ℝ) + q ≠ 0 := by positivity
  have h2 : (997 / 1000 : ℝ) + 997 / 1000 * q ≠ 0 := by positivity
  rw [div_eq_div_iff h2 h1]
  ring

/-- Composing `φ` with a Möbius map `a q / (1 + b q)`. -/
theorem φ_mobius (a b q : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hq : 0 ≤ q) :
    φ (a * q / (1 + b * q)) = 4 * a * q / (1 + (a + b) * q) := by
  unfold φ
  have h1 : (1 : ℝ) + b * q ≠ 0 := by positivity
  have h2 : (1 : ℝ) + (a + b) * q ≠ 0 := by positivity
  have h3 : (1 : ℝ) + a * q / (1 + b * q) = (1 + (a + b) * q) / (1 + b * q) := by
    field_simp
    ring
  rw [h3]
  field_simp

theorem φ_eq (q : ℝ) : φ q = 4 * q / (1 + 1 * q) := by
  unfold φ; ring_nf

/-- An unused triangular path returns `F(q) = 64 q / (1 + 21 q)`. -/
theorem φ3 (q : ℝ) (hq : 0 ≤ q) : φ (φ (φ q)) = 64 * q / (1 + 21 * q) := by
  have e1 : φ q = 4 * q / (1 + 1 * q) := φ_eq q
  have e2 : φ (φ q) = 16 * q / (1 + 5 * q) := by
    rw [e1, φ_mobius 4 1 q (by norm_num) (by norm_num) hq]; ring_nf
  rw [e2]
  have := φ_mobius 16 5 q (by norm_num) (by norm_num) hq
  rw [this]; ring_nf

/-- The best gross gain of an unused path is `7/3`, attained at `q = 1/3`. -/
theorem gain_le (q : ℝ) (hq : 0 ≤ q) : 64 * q / (1 + 21 * q) - q ≤ 7 / 3 := by
  have h : (0 : ℝ) < 1 + 21 * q := by positivity
  rw [sub_le_iff_le_add, div_le_iff₀ h]
  nlinarith [sq_nonneg (q - 1 / 3)]

theorem gain_at_third : 64 * (1 / 3 : ℝ) / (1 + 21 * (1 / 3)) - 1 / 3 = 7 / 3 := by
  norm_num

/-- Hence an unused path earns at most `1/10` after gas. -/
theorem profit_le (q : ℝ) (hq : 0 ≤ q) :
    φ (φ (φ q)) - q - G ≤ 1 / 10 := by
  rw [φ3 q hq]; unfold G; linarith [gain_le q hq]

/-- The construction's transaction (`q = 1/3`) has outputs `1, 2, 8/3` and
after-gas profit exactly `1/10`. -/
theorem third_outputs :
    φ (1 / 3) = 1 ∧ φ (φ (1 / 3)) = 2 ∧ φ (φ (φ (1 / 3))) = 8 / 3 ∧
      φ (φ (φ (1 / 3))) - 1 / 3 - G = 1 / 10 := by
  unfold φ G; norm_num

/-- A profitable unused path uses an input strictly between `1/6` and `1`. -/
theorem profitable_window (q : ℝ) (hq : 0 ≤ q) (hprof : G < φ (φ (φ q)) - q) :
    1 / 6 < q ∧ q < 1 := by
  rw [φ3 q hq] at hprof
  unfold G at hprof
  have h : (0 : ℝ) < 1 + 21 * q := by positivity
  have key : 67 / 30 * (1 + 21 * q) < 64 * q - q * (1 + 21 * q) := by
    have := (lt_div_iff₀ h).mp (by linarith : (67 / 30 : ℝ) + q < 64 * q / (1 + 21 * q))
    nlinarith
  constructor
  · by_contra hle
    rw [not_lt] at hle
    nlinarith [mul_nonneg hq (sub_nonneg.mpr hle)]
  · by_contra hge
    rw [not_lt] at hge
    nlinarith [mul_nonneg (sub_nonneg.mpr hge) hq]

/-- `φ` maps inputs above `1/6` to outputs above `1/6`. -/
theorem φ_gt_sixth (q : ℝ) (hq : 1 / 6 < q) : 1 / 6 < φ q := by
  unfold φ
  have h : (0 : ℝ) < 1 + q := by linarith
  rw [lt_div_iff₀ h]
  linarith

/-- In a profitable unused path, every leg input exceeds `1/6`. -/
theorem profitable_legs_gt_sixth (q : ℝ) (hq : 0 ≤ q) (hprof : G < φ (φ (φ q)) - q) :
    1 / 6 < q ∧ 1 / 6 < φ q ∧ 1 / 6 < φ (φ q) := by
  have hq6 := (profitable_window q hq hprof).1
  exact ⟨hq6, φ_gt_sixth q hq6, φ_gt_sixth _ (φ_gt_sixth q hq6)⟩

/-! ## Basic properties of a swap -/

theorem swapOut_nonneg (α rin rout q : ℝ) (hα : 0 ≤ α) (hin : 0 < rin) (hout : 0 ≤ rout)
    (hq : 0 ≤ q) : 0 ≤ swapOut α rin rout q := by
  unfold swapOut; positivity

theorem swapOut_le_rout (α rin rout q : ℝ) (hα : 0 ≤ α) (hin : 0 < rin) (hout : 0 ≤ rout)
    (hq : 0 ≤ q) : swapOut α rin rout q ≤ rout := by
  unfold swapOut
  have h : 0 < rin + α * q := by positivity
  rw [div_le_iff₀ h]
  nlinarith [mul_nonneg hout hin.le]

theorem φ_mono {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) : φ a ≤ φ b := by
  unfold φ
  have h1 : (0 : ℝ) < 1 + a := by linarith
  have h2 : (0 : ℝ) < 1 + b := by linarith
  rw [div_le_div_iff₀ h1 h2]
  nlinarith

theorem φ_nonneg {a : ℝ} (ha : 0 ≤ a) : 0 ≤ φ a := by
  unfold φ; positivity

/-! ## Reused pools

A pool of the reduction that has already been traded with input `s` has input
reserve at least `α₀ + s` and output reserve at most `4 / (1 + s)`. Its output
is then bounded by `g s r = 4 r / ((1 + s)(1 + s + r))`. With `s = 0` the bound
is `φ`, which covers both fresh and reused pools. -/

/-- Output bound for a pool whose first trade had input `s`. -/
noncomputable def g (s r : ℝ) : ℝ := 4 * r / ((1 + s) * (1 + s + r))

theorem g_zero (r : ℝ) : g 0 r = φ r := by
  unfold g φ; ring_nf

theorem g_nonneg {s r : ℝ} (hs : 0 ≤ s) (hr : 0 ≤ r) : 0 ≤ g s r := by
  unfold g; positivity

theorem g_mono {s a b : ℝ} (hs : 0 ≤ s) (ha : 0 ≤ a) (hab : a ≤ b) : g s a ≤ g s b := by
  unfold g
  have h1 : (0 : ℝ) < (1 + s) * (1 + s + a) := by positivity
  have h2 : (0 : ℝ) < (1 + s) * (1 + s + b) := by
    have : 0 ≤ b := le_trans ha hab
    positivity
  rw [div_le_div_iff₀ h1 h2]
  nlinarith [mul_nonneg hs (sub_nonneg.mpr hab), mul_nonneg (mul_nonneg hs hs) (sub_nonneg.mpr hab)]

/-- The paper's reuse bound: next output at most `4 r / ((1 + s)(1 + s + r))`. -/
theorem reused_out_le (s r rin rout : ℝ) (hs : 0 ≤ s) (hr : 0 ≤ r)
    (hin : α₀ + s ≤ rin) (hout : rout * (1 + s) ≤ 4) :
    swapOut α₀ rin rout r ≤ g s r := by
  unfold swapOut g
  have hα : (α₀ : ℝ) = 997 / 1000 := rfl
  have hrin : 0 < rin := by rw [hα] at hin; linarith
  have h1 : 0 < rin + α₀ * r := by rw [hα]; positivity
  have h2 : (0 : ℝ) < (1 + s) * (1 + s + r) := by positivity
  rw [div_le_div_iff₀ h1 h2]
  have hk : 0 ≤ α₀ * r * (1 + s + r) := by rw [hα]; positivity
  have e1 := mul_le_mul_of_nonneg_right hout hk
  rw [hα] at e1 hin ⊢
  nlinarith [mul_nonneg hr hs, mul_nonneg hr (sub_nonneg.mpr hin)]

/-- Any pool of the construction (fresh or reused) outputs at most `φ`. -/
theorem pool_out_le_φ (r rin rout : ℝ) (hr : 0 ≤ r) (hin : α₀ ≤ rin)
    (hout : rout ≤ 4) : swapOut α₀ rin rout r ≤ φ r := by
  have := reused_out_le 0 r rin rout le_rfl hr (by simpa using hin) (by simpa using hout)
  rwa [g_zero] at this

/-- After a fresh pool is traded with input `s`, its reserves are
`(α₀ + s, 4 / (1 + s))`, i.e. they satisfy the reuse hypotheses with equality. -/
theorem fresh_after_trade (s : ℝ) (hs : 0 ≤ s) :
    (4 - swapOut α₀ α₀ 4 s) * (1 + s) = 4 := by
  rw [swapOut_fresh s hs]; unfold φ
  have h : (1 : ℝ) + s ≠ 0 := by positivity
  field_simp
  ring

/-- Further trades keep a used pool within the reuse hypotheses. -/
theorem reuse_invariant (s r rin rout : ℝ) (hs : 0 ≤ s) (hr : 0 ≤ r)
    (hin : α₀ + s ≤ rin) (hout0 : 0 ≤ rout) (hout : rout * (1 + s) ≤ 4) :
    α₀ + s ≤ rin + r ∧ 0 ≤ rout - swapOut α₀ rin rout r ∧
      (rout - swapOut α₀ rin rout r) * (1 + s) ≤ 4 := by
  have hα : (0 : ℝ) ≤ α₀ := by unfold α₀; norm_num
  have hrin : 0 < rin := by unfold α₀ at hin; linarith
  have hnn := swapOut_nonneg α₀ rin rout r hα hrin hout0 hr
  have hle := swapOut_le_rout α₀ rin rout r hα hrin hout0 hr
  refine ⟨by linarith, by linarith, ?_⟩
  nlinarith

/-! ## Reuse makes a transaction unprofitable

If any leg of a path uses a pool whose earlier input `s` exceeded `1/6`, the
transaction's gross gain stays below the gas cost `G`. We bound each leg
position separately rather than relying on "the first leg is the worst case". -/

/-- Closed form when the reused pool is on leg 1. -/
theorem leg1_closed (s r : ℝ) (hs : 0 ≤ s) (hr : 0 ≤ r) :
    φ (φ (g s r)) = 64 * r / ((1 + s) ^ 2 + (s + 21) * r) := by
  have hu := g_nonneg hs hr
  have e : φ (φ (g s r)) = 16 * g s r / (1 + 5 * g s r) := by
    rw [φ_eq (g s r), φ_mobius 4 1 _ (by norm_num) (by norm_num) hu]; ring_nf
  rw [e]; unfold g
  have h1 : (0 : ℝ) < (1 + s) * (1 + s + r) := by positivity
  have h2 : (0 : ℝ) < (1 + s) ^ 2 + (s + 21) * r := by positivity
  field_simp
  ring

theorem leg1_unprofitable (s r : ℝ) (hs : 1 / 6 ≤ s) (hr : 0 ≤ r) :
    φ (φ (g s r)) - r < G := by
  have hs0 : 0 ≤ s := by linarith
  rw [leg1_closed s r hs0 hr]; unfold G
  have h2 : (0 : ℝ) < (1 + s) ^ 2 + (s + 21) * r := by positivity
  rw [sub_lt_iff_lt_add, div_lt_iff₀ h2]
  have ht : 0 ≤ s - 1 / 6 := by linarith
  nlinarith [sq_nonneg (r - 287 / 762), mul_nonneg ht hr, mul_nonneg ht ht,
    mul_nonneg (mul_nonneg ht hr) hr, mul_nonneg (mul_nonneg ht ht) hr]

/-- Closed form when the reused pool is on leg 2. -/
theorem leg2_closed (s r : ℝ) (hs : 0 ≤ s) (hr : 0 ≤ r) :
    φ (g s (φ r)) = 64 * r / ((1 + s) * ((1 + s) * (1 + r) + 4 * r) + 16 * r) := by
  unfold φ g
  have h0 : (0 : ℝ) < 1 + r := by positivity
  have h1 : (0 : ℝ) < (1 + s) * ((1 + s) * (1 + r) + 4 * r) := by positivity
  have h2 : (0 : ℝ) < (1 + s) * ((1 + s) * (1 + r) + 4 * r) + 16 * r := by positivity
  have h3 : (0 : ℝ) < (1 + s) * (1 + s + 4 * r / (1 + r)) := by positivity
  field_simp
  ring

theorem leg2_unprofitable (s r : ℝ) (hs : 1 / 6 ≤ s) (hr : 0 ≤ r) :
    φ (g s (φ r)) - r < G := by
  have hs0 : 0 ≤ s := by linarith
  rw [leg2_closed s r hs0 hr]; unfold G
  have h2 : (0 : ℝ) < (1 + s) * ((1 + s) * (1 + r) + 4 * r) + 16 * r := by positivity
  rw [sub_lt_iff_lt_add, div_lt_iff₀ h2]
  have ht : 0 ≤ s - 1 / 6 := by linarith
  nlinarith [sq_nonneg (r - 3 / 8), mul_nonneg ht hr, mul_nonneg ht ht,
    mul_nonneg (mul_nonneg ht hr) hr, mul_nonneg (mul_nonneg ht ht) hr,
    mul_nonneg (mul_nonneg ht ht) (mul_nonneg hr hr)]

/-- Closed form when the reused pool is on leg 3. -/
theorem leg3_closed (s r : ℝ) (hs : 0 ≤ s) (hr : 0 ≤ r) :
    g s (φ (φ r)) = 64 * r / ((1 + s) * ((1 + s) * (1 + 5 * r) + 16 * r)) := by
  have e : φ (φ r) = 16 * r / (1 + 5 * r) := by
    rw [φ_eq r, φ_mobius 4 1 r (by norm_num) (by norm_num) hr]; ring_nf
  rw [e]; unfold g
  have h0 : (0 : ℝ) < 1 + 5 * r := by positivity
  have h1 : (0 : ℝ) < (1 + s) * ((1 + s) * (1 + 5 * r) + 16 * r) := by positivity
  have h3 : (0 : ℝ) < (1 + s) * (1 + s + 16 * r / (1 + 5 * r)) := by positivity
  field_simp
  ring

theorem leg3_unprofitable (s r : ℝ) (hs : 1 / 6 ≤ s) (hr : 0 ≤ r) :
    g s (φ (φ r)) - r < G := by
  have hs0 : 0 ≤ s := by linarith
  rw [leg3_closed s r hs0 hr]; unfold G
  have h2 : (0 : ℝ) < (1 + s) * ((1 + s) * (1 + 5 * r) + 16 * r) := by positivity
  rw [sub_lt_iff_lt_add, div_lt_iff₀ h2]
  have ht : 0 ≤ s - 1 / 6 := by linarith
  nlinarith [sq_nonneg (r - 1 / 3), mul_nonneg ht hr, mul_nonneg ht ht,
    mul_nonneg (mul_nonneg ht hr) hr, mul_nonneg (mul_nonneg ht ht) hr,
    mul_nonneg (mul_nonneg ht ht) (mul_nonneg hr hr)]

/-! ## One transaction over three pools -/

/-- A pool state that can arise in the construction: input reserve at least
`α₀`, output reserve in `[0, 4]`. Fresh pools are `(α₀, 4)`. -/
def PoolOK (rin rout : ℝ) : Prop := α₀ ≤ rin ∧ 0 ≤ rout ∧ rout ≤ 4

/-- A pool whose first trade had input `s`. -/
def Reused (s rin rout : ℝ) : Prop := α₀ + s ≤ rin ∧ 0 ≤ rout ∧ rout * (1 + s) ≤ 4

/-- Final output of a three-leg path with input `r`. -/
noncomputable def pathOut (rin₁ rout₁ rin₂ rout₂ rin₃ rout₃ r : ℝ) : ℝ :=
  swapOut α₀ rin₃ rout₃ (swapOut α₀ rin₂ rout₂ (swapOut α₀ rin₁ rout₁ r))

theorem PoolOK.out_nonneg {rin rout : ℝ} (h : PoolOK rin rout) {r : ℝ} (hr : 0 ≤ r) :
    0 ≤ swapOut α₀ rin rout r := by
  have hα : (0 : ℝ) ≤ α₀ := by unfold α₀; norm_num
  have hrin : 0 < rin := by have := h.1; unfold α₀ at this; linarith
  exact swapOut_nonneg α₀ rin rout r hα hrin h.2.1 hr

/-- **Reuse lemma.** If any leg of a path uses a pool whose earlier input was at
least `1/6`, the transaction's gross gain is below the gas cost, so C1 fails. -/
theorem reuse_unprofitable (s r rin₁ rout₁ rin₂ rout₂ rin₃ rout₃ : ℝ)
    (hs : 1 / 6 ≤ s) (hr : 0 ≤ r)
    (h₁ : PoolOK rin₁ rout₁) (h₂ : PoolOK rin₂ rout₂) (h₃ : PoolOK rin₃ rout₃)
    (hreuse : Reused s rin₁ rout₁ ∨ Reused s rin₂ rout₂ ∨ Reused s rin₃ rout₃) :
    pathOut rin₁ rout₁ rin₂ rout₂ rin₃ rout₃ r - r < G := by
  have hs0 : 0 ≤ s := by linarith
  unfold pathOut
  set o₁ := swapOut α₀ rin₁ rout₁ r with ho₁
  set o₂ := swapOut α₀ rin₂ rout₂ o₁ with ho₂
  have n₁ : 0 ≤ o₁ := h₁.out_nonneg hr
  have n₂ : 0 ≤ o₂ := h₂.out_nonneg n₁
  have hφ₁ : o₁ ≤ φ r := pool_out_le_φ r rin₁ rout₁ hr h₁.1 h₁.2.2
  have hφ₂ : o₂ ≤ φ o₁ := pool_out_le_φ o₁ rin₂ rout₂ n₁ h₂.1 h₂.2.2
  have hφ₃ : swapOut α₀ rin₃ rout₃ o₂ ≤ φ o₂ := pool_out_le_φ o₂ rin₃ rout₃ n₂ h₃.1 h₃.2.2
  rcases hreuse with hR | hR | hR
  · -- reused pool on leg 1
    have b₁ : o₁ ≤ g s r := reused_out_le s r rin₁ rout₁ hs0 hr hR.1 hR.2.2
    have b₂ : o₂ ≤ φ (g s r) := le_trans hφ₂ (φ_mono n₁ b₁)
    have b₃ : swapOut α₀ rin₃ rout₃ o₂ ≤ φ (φ (g s r)) := le_trans hφ₃ (φ_mono n₂ b₂)
    linarith [leg1_unprofitable s r hs hr]
  · -- reused pool on leg 2
    have b₂ : o₂ ≤ g s (φ r) :=
      le_trans (reused_out_le s o₁ rin₂ rout₂ hs0 n₁ hR.1 hR.2.2) (g_mono hs0 n₁ hφ₁)
    have b₃ : swapOut α₀ rin₃ rout₃ o₂ ≤ φ (g s (φ r)) := le_trans hφ₃ (φ_mono n₂ b₂)
    linarith [leg2_unprofitable s r hs hr]
  · -- reused pool on leg 3
    have b₂ : o₂ ≤ φ (φ r) := le_trans hφ₂ (φ_mono n₁ hφ₁)
    have b₃ : swapOut α₀ rin₃ rout₃ o₂ ≤ g s (φ (φ r)) :=
      le_trans (reused_out_le s o₂ rin₃ rout₃ hs0 n₂ hR.1 hR.2.2) (g_mono hs0 n₂ b₂)
    linarith [leg3_unprofitable s r hs hr]

/-- A reused pool is still a valid construction pool. -/
theorem Reused.poolOK {s rin rout : ℝ} (hs : 0 ≤ s) (h : Reused s rin rout) :
    PoolOK rin rout := by
  refine ⟨by linarith [h.1], h.2.1, ?_⟩
  nlinarith [h.2.1, h.2.2]

/-- A fresh pool is a valid construction pool. -/
theorem fresh_poolOK : PoolOK α₀ 4 := by
  unfold PoolOK; norm_num

end FormalSpecs.CPMM
