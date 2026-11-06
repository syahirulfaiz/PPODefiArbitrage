import FormalSpecs.CPMM

/-!
# Theorem 1: the 3DM reduction

Formalizes the reduction in the proof of Theorem 1 (`main.v3.tex`, Section III).

Given disjoint sets `X, Y, Z` (here: three types) and triples `E`, the
construction has three tokens `A, B, C` and one directed pool per element:
`X`-pools trade `A → B`, `Y`-pools `B → C`, `Z`-pools `C → A`. Every pool starts
with directional reserves `(α₀, 4)` and fee multiplier `α₀ = 997/1000`. The
supplied catalog contains exactly the paths of the triples in `E`, each starting
at `A`. All token prices are one USD, the initial `A`-balance is `1/3`, and every
executed transaction costs `G = 67/30` in gas, which is deducted from reported
profit and not from the wallet (Eq. (5)).

The main result `reduction_correct` states: `E` contains a matching of size at
least `M` iff some finite sequence of transactions satisfying C1–C5 and the
reserve/wallet transition earns cumulative after-gas profit at least `M/10`.

Constraints C4 and C5 hold by construction of the catalog paths. The time
budget C6 is excluded from the complexity input, as in the paper.
-/

namespace FormalSpecs.Reduction

set_option linter.unusedSectionVars false

open FormalSpecs.CPMM

variable {X Y Z : Type*} [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

/-- One pool per element of `X ⊔ Y ⊔ Z`. -/
abbrev Pool (X Y Z : Type*) := X ⊕ Y ⊕ Z

/-- A 3DM triple; in the construction it is the catalog path `A → B → C → A`. -/
abbrev Triple (X Y Z : Type*) := X × Y × Z

/-- The pools used by the three legs of a triple's path. -/
def p₁ (e : Triple X Y Z) : Pool X Y Z := Sum.inl e.1
def p₂ (e : Triple X Y Z) : Pool X Y Z := Sum.inr (Sum.inl e.2.1)
def p₃ (e : Triple X Y Z) : Pool X Y Z := Sum.inr (Sum.inr e.2.2)

/-- Market state: directional reserves `(input, output)` of every pool and the
`A`-balance of the trading wallet (`B` and `C` balances stay zero). -/
structure State (X Y Z : Type*) where
  res : Pool X Y Z → ℝ × ℝ
  wal : ℝ

/-- Initial state of the constructed instance. -/
noncomputable def init : State X Y Z := ⟨fun _ => (α₀, 4), 1 / 3⟩

/-- Leg outputs of the transaction on path `e` with first-swap input `q`. -/
noncomputable def o₁ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : ℝ :=
  swapOut α₀ (st.res (p₁ e)).1 (st.res (p₁ e)).2 q
noncomputable def o₂ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : ℝ :=
  swapOut α₀ (st.res (p₂ e)).1 (st.res (p₂ e)).2 (o₁ st e q)
noncomputable def o₃ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : ℝ :=
  swapOut α₀ (st.res (p₃ e)).1 (st.res (p₃ e)).2 (o₂ st e q)

/-- After-gas profit `π = (q₃ - q₀) · 1 USD - G`. -/
noncomputable def profit (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : ℝ :=
  o₃ st e q - q - G

/-- A transaction is admissible: catalog path, C2 (wallet), C3 (each output
positive and below the pool's output reserve) and C1 (positive profit). -/
def StepOK (E : Finset (Triple X Y Z)) (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : Prop :=
  e ∈ E ∧ 0 < q ∧ q ≤ st.wal ∧
  0 < o₁ st e q ∧ o₁ st e q < (st.res (p₁ e)).2 ∧
  0 < o₂ st e q ∧ o₂ st e q < (st.res (p₂ e)).2 ∧
  0 < o₃ st e q ∧ o₃ st e q < (st.res (p₃ e)).2 ∧
  0 < profit st e q

/-- Reserve and wallet transition, Eq. (5). -/
noncomputable def step (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) : State X Y Z where
  res := Function.update (Function.update (Function.update st.res
      (p₁ e) ((st.res (p₁ e)).1 + q, (st.res (p₁ e)).2 - o₁ st e q))
      (p₂ e) ((st.res (p₂ e)).1 + o₁ st e q, (st.res (p₂ e)).2 - o₂ st e q))
      (p₃ e) ((st.res (p₃ e)).1 + o₂ st e q, (st.res (p₃ e)).2 - o₃ st e q)
  wal := st.wal - q + o₃ st e q

/-- A finite sequence of transactions that are each admissible in turn. -/
def Feasible (E : Finset (Triple X Y Z)) : State X Y Z → List (Triple X Y Z × ℝ) → Prop
  | _, [] => True
  | st, (e, q) :: l => StepOK E st e q ∧ Feasible E (step st e q) l

/-- Cumulative after-gas profit `Π_L` of a sequence. -/
noncomputable def totalProfit : State X Y Z → List (Triple X Y Z × ℝ) → ℝ
  | _, [] => 0
  | st, (e, q) :: l => profit st e q + totalProfit (step st e q) l

/-- Two triples share no element. -/
def Disj (a b : Triple X Y Z) : Prop := a.1 ≠ b.1 ∧ a.2.1 ≠ b.2.1 ∧ a.2.2 ≠ b.2.2

/-- A 3DM matching: distinct triples are element-disjoint. -/
def IsMatching (S : Finset (Triple X Y Z)) : Prop :=
  ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Disj a b

/-! ## State lemmas -/

/-- A pool is fresh when it still has its initial reserves. -/
def Fresh (st : State X Y Z) (p : Pool X Y Z) : Prop := st.res p = (α₀, 4)

/-- Every pool is fresh or was first traded with an input of at least `1/6`. -/
def Good (st : State X Y Z) : Prop :=
  ∀ p, Fresh st p ∨ ∃ s, 1 / 6 ≤ s ∧ Reused s (st.res p).1 (st.res p).2

omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem good_init : Good (init : State X Y Z) := fun _ => Or.inl rfl

omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem Good.poolOK {st : State X Y Z} (h : Good st) (p : Pool X Y Z) :
    PoolOK (st.res p).1 (st.res p).2 := by
  rcases h p with hf | ⟨s, hs, hR⟩
  · unfold Fresh at hf; rw [hf]; exact fresh_poolOK
  · exact hR.poolOK (by linarith)

omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem p₁_ne_p₂ (e : Triple X Y Z) : p₁ e ≠ p₂ e := by simp [p₁, p₂]
omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem p₁_ne_p₃ (e : Triple X Y Z) : p₁ e ≠ p₃ e := by simp [p₁, p₃]
omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem p₂_ne_p₃ (e : Triple X Y Z) : p₂ e ≠ p₃ e := by simp [p₂, p₃]

theorem step_res_p₁ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) :
    (step st e q).res (p₁ e) = ((st.res (p₁ e)).1 + q, (st.res (p₁ e)).2 - o₁ st e q) := by
  simp [step, Function.update_of_ne (p₁_ne_p₃ e), Function.update_of_ne (p₁_ne_p₂ e)]

theorem step_res_p₂ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) :
    (step st e q).res (p₂ e) = ((st.res (p₂ e)).1 + o₁ st e q, (st.res (p₂ e)).2 - o₂ st e q) := by
  simp [step, Function.update_of_ne (p₂_ne_p₃ e)]

theorem step_res_p₃ (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) :
    (step st e q).res (p₃ e) = ((st.res (p₃ e)).1 + o₂ st e q, (st.res (p₃ e)).2 - o₃ st e q) := by
  simp [step]

theorem step_res_other (st : State X Y Z) (e : Triple X Y Z) (q : ℝ) (p : Pool X Y Z)
    (h₁ : p ≠ p₁ e) (h₂ : p ≠ p₂ e) (h₃ : p ≠ p₃ e) : (step st e q).res p = st.res p := by
  simp [step, Function.update_of_ne h₁, Function.update_of_ne h₂, Function.update_of_ne h₃]

omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
/-- On a path of fresh pools, the legs implement `φ`. -/
theorem fresh_outputs {st : State X Y Z} {e : Triple X Y Z} {q : ℝ} (hq : 0 ≤ q)
    (f₁ : Fresh st (p₁ e)) (f₂ : Fresh st (p₂ e)) (f₃ : Fresh st (p₃ e)) :
    o₁ st e q = φ q ∧ o₂ st e q = φ (φ q) ∧ o₃ st e q = φ (φ (φ q)) := by
  unfold Fresh at f₁ f₂ f₃
  have e₁ : o₁ st e q = φ q := by unfold o₁; rw [f₁]; exact swapOut_fresh q hq
  have e₂ : o₂ st e q = φ (φ q) := by
    unfold o₂; rw [e₁, f₂]; exact swapOut_fresh _ (φ_nonneg hq)
  have e₃ : o₃ st e q = φ (φ (φ q)) := by
    unfold o₃; rw [e₂, f₃]; exact swapOut_fresh _ (φ_nonneg (φ_nonneg hq))
  exact ⟨e₁, e₂, e₃⟩

/-! ## An executed transaction uses only fresh pools -/

omit [DecidableEq X] [DecidableEq Y] [DecidableEq Z] in
theorem not_fresh_reused {st : State X Y Z} (hG : Good st) {p : Pool X Y Z}
    (h : ¬ Fresh st p) : ∃ s, 1 / 6 ≤ s ∧ Reused s (st.res p).1 (st.res p).2 := by
  rcases hG p with hf | hR
  · exact absurd hf h
  · exact hR

/-- **Key step.** Under the invariant, an admissible transaction touches only
fresh pools: otherwise `reuse_unprofitable` contradicts C1. -/
theorem executed_fresh {E : Finset (Triple X Y Z)} {st : State X Y Z} {e : Triple X Y Z}
    {q : ℝ} (hG : Good st) (h : StepOK E st e q) :
    Fresh st (p₁ e) ∧ Fresh st (p₂ e) ∧ Fresh st (p₃ e) := by
  obtain ⟨-, hq, -, -, -, -, -, -, -, hprof⟩ := h
  have ok₁ := hG.poolOK (p₁ e)
  have ok₂ := hG.poolOK (p₂ e)
  have ok₃ := hG.poolOK (p₃ e)
  have contra : ∀ s, 1 / 6 ≤ s →
      (Reused s (st.res (p₁ e)).1 (st.res (p₁ e)).2 ∨ Reused s (st.res (p₂ e)).1 (st.res (p₂ e)).2 ∨
        Reused s (st.res (p₃ e)).1 (st.res (p₃ e)).2) → False := by
    intro s hs hR
    have := reuse_unprofitable s q _ _ _ _ _ _ hs hq.le ok₁ ok₂ ok₃ hR
    unfold profit o₃ o₂ o₁ at hprof
    unfold pathOut at this
    linarith
  refine ⟨?_, ?_, ?_⟩
  · by_contra hn
    obtain ⟨s, hs, hR⟩ := not_fresh_reused hG hn
    exact contra s hs (Or.inl hR)
  · by_contra hn
    obtain ⟨s, hs, hR⟩ := not_fresh_reused hG hn
    exact contra s hs (Or.inr (Or.inl hR))
  · by_contra hn
    obtain ⟨s, hs, hR⟩ := not_fresh_reused hG hn
    exact contra s hs (Or.inr (Or.inr hR))

/-- A freshly traded pool satisfies the reuse hypotheses for its input `s`. -/
theorem traded_reused {s o : ℝ} (hs : 0 ≤ s) (ho : o = swapOut α₀ α₀ 4 s) (hlt : o < 4) :
    Reused s (α₀ + s) (4 - o) := by
  refine ⟨le_rfl, by linarith, ?_⟩
  rw [ho]; exact (fresh_after_trade s hs).le

/-- After an admissible transaction: the invariant is preserved, the three pools
of the path are no longer fresh, and its profit is at most `1/10`. -/
theorem executed_step {E : Finset (Triple X Y Z)} {st : State X Y Z} {e : Triple X Y Z}
    {q : ℝ} (hG : Good st) (h : StepOK E st e q) :
    Good (step st e q) ∧ ¬ Fresh (step st e q) (p₁ e) ∧ ¬ Fresh (step st e q) (p₂ e) ∧
      ¬ Fresh (step st e q) (p₃ e) ∧ profit st e q ≤ 1 / 10 := by
  obtain ⟨f₁, f₂, f₃⟩ := executed_fresh hG h
  obtain ⟨-, hq, -, -, hlt₁, -, hlt₂, -, hlt₃, hprof⟩ := h
  obtain ⟨e₁, e₂, e₃⟩ := fresh_outputs hq.le f₁ f₂ f₃
  have hG' : G < φ (φ (φ q)) - q := by unfold profit at hprof; rw [e₃] at hprof; linarith
  obtain ⟨s₁, s₂, s₃⟩ := profitable_legs_gt_sixth q hq.le hG'
  unfold Fresh at f₁ f₂ f₃
  rw [f₁] at hlt₁; rw [f₂] at hlt₂; rw [f₃] at hlt₃
  simp only at hlt₁ hlt₂ hlt₃
  -- the three traded pools
  have r₁ : (step st e q).res (p₁ e) = (α₀ + q, 4 - o₁ st e q) := by rw [step_res_p₁, f₁]
  have r₂ : (step st e q).res (p₂ e) = (α₀ + o₁ st e q, 4 - o₂ st e q) := by rw [step_res_p₂, f₂]
  have r₃ : (step st e q).res (p₃ e) = (α₀ + o₂ st e q, 4 - o₃ st e q) := by rw [step_res_p₃, f₃]
  have hα : (α₀ : ℝ) = 997 / 1000 := rfl
  have nf : ∀ t : ℝ, 0 < t → ∀ o : ℝ, ((α₀ + t, o) : ℝ × ℝ) ≠ (α₀, 4) := by
    intro t ht o hEq
    have := congrArg Prod.fst hEq
    simp only at this
    linarith
  have hs₁ : 1 / 6 < o₁ st e q := by rw [e₁]; exact s₂
  have hs₂ : 1 / 6 < o₂ st e q := by rw [e₂]; exact s₃
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro p
    by_cases h₁ : p = p₁ e
    · subst h₁; right
      refine ⟨q, s₁.le, ?_⟩
      rw [r₁]
      exact traded_reused hq.le (by unfold o₁; rw [f₁]) hlt₁
    by_cases h₂ : p = p₂ e
    · subst h₂; right
      refine ⟨o₁ st e q, hs₁.le, ?_⟩
      rw [r₂]
      exact traded_reused (by linarith) (by unfold o₂; rw [f₂]) hlt₂
    by_cases h₃ : p = p₃ e
    · subst h₃; right
      refine ⟨o₂ st e q, hs₂.le, ?_⟩
      rw [r₃]
      exact traded_reused (by linarith) (by unfold o₃; rw [f₃]) hlt₃
    have hp := step_res_other st e q p h₁ h₂ h₃
    rcases hG p with hf | hR
    · left; unfold Fresh; rw [hp]; exact hf
    · right; rw [hp]; exact hR
  · unfold Fresh; rw [r₁]; exact nf q hq _
  · unfold Fresh; rw [r₂]; exact nf _ (by linarith) _
  · unfold Fresh; rw [r₃]; exact nf _ (by linarith) _
  · unfold profit; rw [e₃]; exact profit_le q hq.le

/-- Pools that are fresh after an admissible step were fresh before it. -/
theorem fresh_of_step {E : Finset (Triple X Y Z)} {st : State X Y Z} {e : Triple X Y Z}
    {q : ℝ} (hG : Good st) (h : StepOK E st e q) {p : Pool X Y Z}
    (hp : Fresh (step st e q) p) : Fresh st p ∧ p ≠ p₁ e ∧ p ≠ p₂ e ∧ p ≠ p₃ e := by
  obtain ⟨-, n₁, n₂, n₃, -⟩ := executed_step hG h
  have h₁ : p ≠ p₁ e := by rintro rfl; exact n₁ hp
  have h₂ : p ≠ p₂ e := by rintro rfl; exact n₂ hp
  have h₃ : p ≠ p₃ e := by rintro rfl; exact n₃ hp
  refine ⟨?_, h₁, h₂, h₃⟩
  unfold Fresh at hp ⊢
  rwa [step_res_other st e q p h₁ h₂ h₃] at hp

/-! ## Sequence ⇒ matching -/

/-- All three pools of a triple are fresh. -/
def PathFresh (st : State X Y Z) (e : Triple X Y Z) : Prop :=
  Fresh st (p₁ e) ∧ Fresh st (p₂ e) ∧ Fresh st (p₃ e)

theorem seq_to_matching (E : Finset (Triple X Y Z)) :
    ∀ (l : List (Triple X Y Z × ℝ)) (st : State X Y Z), Good st → Feasible E st l →
      (l.map Prod.fst).Pairwise Disj ∧ (∀ e ∈ l.map Prod.fst, e ∈ E ∧ PathFresh st e) ∧
        totalProfit st l ≤ (l.length : ℝ) / 10
  | [], st, _, _ => by simp [totalProfit]
  | (e, q) :: l, st, hG, hF => by
    obtain ⟨hstep, hrest⟩ := hF
    obtain ⟨hG', -, -, -, hπ⟩ := executed_step hG hstep
    obtain ⟨ih₁, ih₂, ih₃⟩ := seq_to_matching E l (step st e q) hG' hrest
    have hfr := executed_fresh hG hstep
    refine ⟨?_, ?_, ?_⟩
    · refine List.pairwise_cons.mpr ⟨?_, ih₁⟩
      intro e' he'
      obtain ⟨-, a₁, a₂, a₃⟩ := ih₂ e' he'
      have b₁ := (fresh_of_step hG hstep a₁).2.1
      have b₂ := (fresh_of_step hG hstep a₂).2.2.1
      have b₃ := (fresh_of_step hG hstep a₃).2.2.2
      refine ⟨fun hx => b₁ ?_, fun hy => b₂ ?_, fun hz => b₃ ?_⟩
      · simp only [p₁, Sum.inl.injEq]; exact hx.symm
      · simp only [p₂, Sum.inr.injEq, Sum.inl.injEq]; exact hy.symm
      · simp only [p₃, Sum.inr.injEq]; exact hz.symm
    · intro e' he'
      simp only [List.map_cons, List.mem_cons] at he'
      rcases he' with rfl | he'
      · exact ⟨hstep.1, hfr⟩
      · obtain ⟨hE, a₁, a₂, a₃⟩ := ih₂ e' he'
        exact ⟨hE, (fresh_of_step hG hstep a₁).1, (fresh_of_step hG hstep a₂).1,
          (fresh_of_step hG hstep a₃).1⟩
    · simp only [totalProfit, List.length_cons, Nat.cast_succ]
      linarith

/-! ## Matching ⇒ sequence -/

theorem matching_to_seq (E : Finset (Triple X Y Z)) :
    ∀ (T : List (Triple X Y Z)) (st : State X Y Z), T.Pairwise Disj → (∀ e ∈ T, e ∈ E) →
      (∀ e ∈ T, PathFresh st e) → 1 / 3 ≤ st.wal →
      Feasible E st (T.map fun e => (e, (1 / 3 : ℝ))) ∧
        totalProfit st (T.map fun e => (e, (1 / 3 : ℝ))) = (T.length : ℝ) / 10
  | [], st, _, _, _, _ => by simp [Feasible, totalProfit]
  | e :: T, st, hP, hE, hfr, hw => by
    obtain ⟨hdisj, hP'⟩ := List.pairwise_cons.mp hP
    obtain ⟨f₁, f₂, f₃⟩ := hfr e List.mem_cons_self
    obtain ⟨e₁, e₂, e₃⟩ := fresh_outputs (by norm_num : (0 : ℝ) ≤ 1 / 3) f₁ f₂ f₃
    obtain ⟨t₁, t₂, t₃, tπ⟩ := third_outputs
    have hstep : StepOK E st e (1 / 3) := by
      unfold Fresh at f₁ f₂ f₃
      refine ⟨hE e List.mem_cons_self, by norm_num, hw, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [e₁, t₁]; norm_num
      · rw [e₁, t₁, f₁]; norm_num
      · rw [e₂, t₂]; norm_num
      · rw [e₂, t₂, f₂]; norm_num
      · rw [e₃, t₃]; norm_num
      · rw [e₃, t₃, f₃]; norm_num
      · unfold profit; rw [e₃, tπ]; norm_num
    have hπ : profit st e (1 / 3) = 1 / 10 := by unfold profit; rw [e₃, tπ]
    have hw' : 1 / 3 ≤ (step st e (1 / 3)).wal := by
      show 1 / 3 ≤ st.wal - 1 / 3 + o₃ st e (1 / 3)
      rw [e₃, t₃]; linarith
    have hfr' : ∀ e' ∈ T, PathFresh (step st e (1 / 3)) e' := by
      intro e' he'
      obtain ⟨d₁, d₂, d₃⟩ := hdisj e' he'
      obtain ⟨g₁, g₂, g₃⟩ := hfr e' (List.mem_cons_of_mem e he')
      unfold Fresh at g₁ g₂ g₃
      unfold PathFresh Fresh
      refine ⟨?_, ?_, ?_⟩
      · rw [step_res_other _ _ _ _ (by simp [p₁]; exact fun h => d₁ h.symm)
          (by simp [p₁, p₂]) (by simp [p₁, p₃])]; exact g₁
      · rw [step_res_other _ _ _ _ (by simp [p₁, p₂])
          (by simp [p₂]; exact fun h => d₂ h.symm) (by simp [p₂, p₃])]; exact g₂
      · rw [step_res_other _ _ _ _ (by simp [p₁, p₃]) (by simp [p₂, p₃])
          (by simp [p₃]; exact fun h => d₃ h.symm)]; exact g₃
    obtain ⟨ih₁, ih₂⟩ := matching_to_seq E T (step st e (1 / 3)) hP'
      (fun e' he' => hE e' (List.mem_cons_of_mem e he')) hfr' hw'
    refine ⟨⟨hstep, ih₁⟩, ?_⟩
    simp only [List.map_cons, totalProfit, List.length_cons, Nat.cast_succ, ih₂, hπ]
    ring

/-! ## Theorem 1 (reduction correctness) -/

instance : Std.Symm (Disj (X := X) (Y := Y) (Z := Z)) where
  symm _ _ h := ⟨Ne.symm h.1, Ne.symm h.2.1, Ne.symm h.2.2⟩

instance : Std.Irrefl (Disj (X := X) (Y := Y) (Z := Z)) where
  irrefl _ h := h.1 rfl

/-- **Theorem 1, reduction correctness.** For the constructed instance, `E`
contains a matching of size at least `M` iff some feasible sequence of
transactions (C1–C5 with the reserve and wallet transition) earns cumulative
after-gas profit at least `M/10`. -/
theorem reduction_correct (E : Finset (Triple X Y Z)) (M : ℕ) :
    (∃ S ⊆ E, IsMatching S ∧ M ≤ S.card) ↔
      ∃ l : List (Triple X Y Z × ℝ), Feasible E init l ∧ (M : ℝ) / 10 ≤ totalProfit init l := by
  constructor
  · rintro ⟨S, hSE, hM, hcard⟩
    have hP : S.toList.Pairwise Disj :=
      (Finset.nodup_toList S).pairwise_of_forall_ne fun a ha b hb hab =>
        hM a (Finset.mem_toList.mp ha) b (Finset.mem_toList.mp hb) hab
    obtain ⟨hF, hTot⟩ := matching_to_seq E S.toList init hP
      (fun e he => hSE (Finset.mem_toList.mp he))
      (fun e _ => ⟨rfl, rfl, rfl⟩) (by simp [init])
    refine ⟨_, hF, ?_⟩
    rw [hTot, Finset.length_toList]
    have : (M : ℝ) ≤ S.card := by exact_mod_cast hcard
    linarith
  · rintro ⟨l, hF, hTot⟩
    obtain ⟨hP, hmem, hle⟩ := seq_to_matching E l init good_init hF
    refine ⟨(l.map Prod.fst).toFinset, ?_, ?_, ?_⟩
    · intro e he
      exact (hmem e (List.mem_toFinset.mp he)).1
    · intro a ha b hb hab
      exact hP.forall (List.mem_toFinset.mp ha) (List.mem_toFinset.mp hb) hab
    · rw [List.toFinset_card_of_nodup hP.nodup, List.length_map]
      have : (M : ℝ) ≤ l.length := by linarith
      exact_mod_cast this

/-- The constructed instance has polynomial size: one pool per element and one
catalog path per triple, with constant-size numbers. -/
theorem instance_size [Fintype X] [Fintype Y] [Fintype Z] :
    Fintype.card (Pool X Y Z) = Fintype.card X + Fintype.card Y + Fintype.card Z := by
  simp [Pool, Fintype.card_sum, add_assoc]

end FormalSpecs.Reduction
