"""Evaluate nine methods on 23 settings and seeds 40-49 (2070 episodes).
Uses only the new best_model.zip files; no historical hashes or model manifests.
The workbook filename is retained for convenience, not historical result identity.
"""

from __future__ import annotations
import argparse
import copy
import dataclasses
import json
import math
import re
import time
import traceback
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

METHOD_ORDER_DEFAULT = [
    "Greedy",
    "Heuristic",
    "Beam_Search",
    "Grid_Search",
    "SAC",
    "TD3",
    "PPO",
    "NLP_SciPy_Minimize",
    "Bellman_Ford_NetworkX",
]
METHOD_DISPLAY = {
    "Greedy": "Greedy",
    "Heuristic": "Heuristic",
    "Beam_Search": "Beam Search",
    "Grid_Search": "Grid Search",
    "PPO": "PPO",
    "SAC": "SAC",
    "TD3": "TD3",
    "NLP_SciPy_Minimize": "NLP SciPy minimize",
    "Bellman_Ford_NetworkX": "Bellman-Ford NetworkX",
}
RL_METHODS = {"PPO", "SAC", "TD3"}
EPS = 1e-12
PROTOCOL_ID = "v2.09-legacy-symbol-jun27-174p-114r-c6-r3"
EVALUATION_PROTOCOL_ID = "v2.09-legacy-symbol-jun27-driver-primary-nine-method-c6-r1"
POOL_DATA_FILENAME = "uniswap_v2_and_sushiswap_v2_pools_data.2026-06-27.json"


@dataclass(frozen=True)
class Scenario:
    dimension: str
    ablation_value: float
    ablation_label: str
    balance_usd: int
    c3_liquidity_depth: float
    gas_multiplier: float
    c6_latency_budget_seconds: float


@dataclass
class Selection:
    path_index: Optional[int]
    amount_fraction: float
    expected_net_profit_usd: float = 0.0
    expected_gross_profit_usd: float = 0.0
    selected_by: str = ""
    note: str = ""
    error: str = ""
    path_screen_count: int = 0
    amount_evaluation_count: int = 0
    cpmm_evaluation_count: int = 0
    graph_operation_count: int = 0
    optimizer_function_evaluation_count: int = 0

    @property
    def has_trade(self) -> bool:
        return self.path_index is not None


WORKLOAD_FIELDS: Tuple[str, ...] = (
    "path_screen_count",
    "amount_evaluation_count",
    "cpmm_evaluation_count",
    "graph_operation_count",
    "optimizer_function_evaluation_count",
)


class ModelCache:

    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str, str], Any] = {}
        self._bound_env_refs: Dict[Tuple[str, str, str], Any] = {}

    def get(self, method: str, model_path: Path, device: str, sim: Any, env: Any) -> Any:
        key = (method, str(model_path.resolve()), device)
        if key not in self._models:
            try:
                from stable_baselines3 import PPO, SAC, TD3
            except ImportError as exc:
                raise RuntimeError(
                    "stable_baselines3 is required for PPO/SAC/TD3 inference. Install it or remove RL methods from --methods."
                ) from exc
            cls_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
            self._models[key] = cls_map[method].load(str(model_path), device=device)
        model = self._models[key]
        bound_ref = self._bound_env_refs.get(key)
        if method == "PPO" and (bound_ref is None or bound_ref() is not env):
            sim.configure_model_for_env(model, env, method)
            self._bound_env_refs[key] = weakref.ref(env)
        return model


MODEL_CACHE = ModelCache()


def parse_number_list(text: str, cast=float) -> List[Any]:
    text = str(text or "").strip()
    if not text:
        return []
    out: List[Any] = []
    for part in re.split("[,;\\s]+", text):
        if part.strip():
            out.append(cast(part.strip()))
    return out


def normalize_method_name(name: str) -> str:
    key = str(name).strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "Beam": "Beam_Search",
        "BeamSearch": "Beam_Search",
        "Beam_Search": "Beam_Search",
        "Grid": "Grid_Search",
        "GridSearch": "Grid_Search",
        "Grid_Search": "Grid_Search",
        "NLP": "NLP_SciPy_Minimize",
        "SciPy": "NLP_SciPy_Minimize",
        "SciPy_Minimize": "NLP_SciPy_Minimize",
        "NLP_SciPy_Minimize": "NLP_SciPy_Minimize",
        "BellmanFord": "Bellman_Ford_NetworkX",
        "Bellman_Ford": "Bellman_Ford_NetworkX",
        "Bellman_Ford_NetworkX": "Bellman_Ford_NetworkX",
        "NetworkX": "Bellman_Ford_NetworkX",
        "Greedy": "Greedy",
        "Heuristic": "Heuristic",
        "PPO": "PPO",
        "SAC": "SAC",
        "TD3": "TD3",
    }
    if key in aliases:
        return aliases[key]
    upper = key.upper()
    if upper in RL_METHODS:
        return upper
    raise ValueError(f"Unknown method {name!r}")


def format_balance_label(balance: int) -> str:
    if int(balance) % 1000 == 0:
        return f"{int(balance) // 1000}K"
    return str(balance)


def format_float_label(value: float) -> str:
    value = float(value)
    if abs(value - int(value)) < 1e-09:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    balance_values = [int(x) for x in parse_number_list(args.balance_values, cast=float)]
    latency_values = [float(x) for x in parse_number_list(args.latency_values, cast=float)]
    gas_values = [float(x) for x in parse_number_list(args.gas_values, cast=float)]
    liquidity_values = [float(x) for x in parse_number_list(args.liquidity_values, cast=float)]
    scenarios: List[Scenario] = []
    for balance in balance_values:
        scenarios.append(
            Scenario(
                dimension="balance",
                ablation_value=float(balance),
                ablation_label=format_balance_label(balance),
                balance_usd=int(balance),
                c3_liquidity_depth=1.0,
                gas_multiplier=1.0,
                c6_latency_budget_seconds=12.0,
            )
        )
    for c6 in latency_values:
        scenarios.append(
            Scenario(
                dimension="latency",
                ablation_value=float(c6),
                ablation_label=f"{format_float_label(c6)}s",
                balance_usd=2000,
                c3_liquidity_depth=1.0,
                gas_multiplier=1.0,
                c6_latency_budget_seconds=float(c6),
            )
        )
    for gas in gas_values:
        scenarios.append(
            Scenario(
                dimension="gas",
                ablation_value=float(gas),
                ablation_label=f"{format_float_label(gas)}x",
                balance_usd=2000,
                c3_liquidity_depth=1.0,
                gas_multiplier=float(gas),
                c6_latency_budget_seconds=12.0,
            )
        )
    for c3 in liquidity_values:
        scenarios.append(
            Scenario(
                dimension="liquidity",
                ablation_value=float(c3),
                ablation_label=f"{format_float_label(c3)}",
                balance_usd=2000,
                c3_liquidity_depth=float(c3),
                gas_multiplier=1.0,
                c6_latency_budget_seconds=12.0,
            )
        )
    if args.only_dimensions:
        allowed = {x.strip().lower() for x in parse_number_list(args.only_dimensions, cast=str)}
        scenarios = [s for s in scenarios if s.dimension.lower() in allowed]
    if args.max_scenarios and args.max_scenarios > 0:
        scenarios = scenarios[: int(args.max_scenarios)]
    return scenarios


def resolve_model_path(args, method):
    return Path(args.model_dir) / method.lower() / "best_model.zip"


def make_env_namespace(
    args: argparse.Namespace, scenario: Scenario, method: str
) -> argparse.Namespace:
    env_amount_min = float(args.amount_min)
    env_amount_max = float(args.amount_max)
    uses_baseline_grid = method in {
        "Greedy",
        "Heuristic",
        "Beam_Search",
        "Grid_Search",
        "Bellman_Ford_NetworkX",
    }
    env_amount_grid = (
        baseline_grid_string(args, method) if uses_baseline_grid else str(args.amount_grid)
    )
    return argparse.Namespace(
        dataset_dir=str(Path(args.dataset_dir).expanduser()),
        amount_min=env_amount_min,
        amount_max=env_amount_max,
        gas_multiplier=float(scenario.gas_multiplier),
        max_paths=int(args.max_paths),
        block_time_seconds=float(scenario.c6_latency_budget_seconds),
        reward_reference_usd=float(args.reward_reference_usd),
        action_mode=str(args.action_mode),
        amount_grid=env_amount_grid,
        protocol_id=PROTOCOL_ID,
        c1_rejection_reward=float(getattr(args, "c1_rejection_reward", -0.1)),
        c1_counterfactual_reward=False,
    )


def force_wallet_balance(env: Any, balance_usd: int) -> str:
    desired = f".{int(balance_usd)}.json"
    exact = [(name, wallet) for name, wallet in env.wallets if str(name).endswith(desired)]
    if not exact:
        pattern = re.compile(f"(?:^|\\.){int(balance_usd)}\\.json$")
        exact = [(name, wallet) for name, wallet in env.wallets if pattern.search(str(name))]
    if not exact:
        available = ", ".join((str(name) for name, _ in env.wallets[:20]))
        raise ValueError(
            f"No wallet file found for balance {balance_usd}. Available examples: {available}"
        )
    env.wallets = [exact[0]]
    env.reset_counter = 0
    env.wallet_start_tokens = sorted(
        {
            token
            for _, wallet in env.wallets
            for token, amount in wallet.items()
            if float(amount) > 0.0
        }
    )
    return str(exact[0][0])


def scale_liquidity_depth(env: Any, c3_liquidity_depth: float) -> None:
    c3 = float(c3_liquidity_depth)
    if c3 <= 0:
        raise ValueError("C3 liquidity depth must be positive")
    env.liquidity_depth_multiplier = c3
    base = copy.deepcopy(env.pool_list)
    new_pools = []
    for pool in base:
        new_pools.append(
            dataclasses.replace(
                pool, reserve0=float(pool.reserve0) * c3, reserve1=float(pool.reserve1) * c3
            )
        )
    env.pool_list = new_pools
    env.pool_template = {pool.pool_id: pool for pool in env.pool_list}
    env.current_pools = env._pool_state_from_template()


def make_driver_env(
    sim: Any,
    args: argparse.Namespace,
    scenario: Scenario,
    seed: int,
    method: str,
    clock: Optional[Any] = None,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    env_args = make_env_namespace(args, scenario, method)
    algo = method if method in RL_METHODS else None
    env = sim.make_env(env_args, seed=int(seed), wallet_mode="cycle", algo=algo)
    if clock is not None:
        env._clock = clock
    force_wallet_balance(env, scenario.balance_usd)
    scale_liquidity_depth(env, scenario.c3_liquidity_depth)
    obs, reset_info = sim.reset_env_compat(env, seed=int(seed))
    return (env, obs, reset_info)


def amount_grid(env: Any) -> List[float]:
    return [float(x) for x in getattr(env, "amount_grid", [])]


def _dedupe_sorted_fractions(values: Iterable[float], lo: float, hi: float) -> List[float]:
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = (hi, lo)
    rounded = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v) and lo - 1e-15 <= v <= hi + 1e-15:
            rounded.append(round(min(max(v, lo), hi), 12))
    return sorted(dict.fromkeys(rounded))


def build_uniform_fraction_grid(step: float, amount_max: float = 1.0) -> List[float]:
    step = float(step)
    amount_max = float(amount_max)
    if step <= 0.0 or amount_max <= 0.0:
        raise ValueError("Grid step and amount_max must be positive.")
    count = int(round(amount_max / step))
    if count <= 0 or not math.isclose(count * step, amount_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"amount_max={amount_max} must be an integer multiple of grid step {step}."
        )
    return (np.arange(1, count + 1, dtype=float) * step).tolist()


def baseline_grid_for_method_args(args: argparse.Namespace, method: str) -> List[float]:
    method_key = normalize_method_name(method)
    if method_key not in {
        "Greedy",
        "Heuristic",
        "Beam_Search",
        "Grid_Search",
        "Bellman_Ford_NetworkX",
    }:
        raise ValueError(f"{method_key} does not use the baseline amount grid.")
    grid = build_uniform_fraction_grid(
        float(args.baseline_grid_step), float(args.baseline_amount_max)
    )
    return _dedupe_sorted_fractions(
        grid, float(args.baseline_amount_min), float(args.baseline_amount_max)
    )


def baseline_grid_string(args: argparse.Namespace, method: str) -> str:
    return ",".join((f"{x:.12g}" for x in baseline_grid_for_method_args(args, method)))


def method_fraction_grid(env: Any, args: argparse.Namespace, method: str) -> List[float]:
    if method in RL_METHODS:
        return amount_grid(env)
    grid = baseline_grid_for_method_args(args, method)
    return _dedupe_sorted_fractions(grid, float(env.amount_min), float(env.amount_max))


def reserves_for_direction(
    pool_state: Mapping[str, Any], token_in: str
) -> Tuple[float, float, str]:
    if token_in == pool_state["token0"]:
        return (
            float(pool_state["reserve0"]),
            float(pool_state["reserve1"]),
            str(pool_state["token1"]),
        )
    if token_in == pool_state["token1"]:
        return (
            float(pool_state["reserve1"]),
            float(pool_state["reserve0"]),
            str(pool_state["token0"]),
        )
    return (0.0, 0.0, "")


def spot_rate_for_pool(env: Any, pool_id: str, token_in: str) -> float:
    state = env.current_pools.get(pool_id)
    if not state:
        return 0.0
    reserve_in, reserve_out, _token_out = reserves_for_direction(state, token_in)
    if reserve_in <= 0 or reserve_out <= 0:
        return 0.0
    fee = float(state.get("fee", 0.003))
    return max(0.0, (1.0 - fee) * reserve_out / reserve_in)


def path_spot_product(env: Any, path_index: int) -> float:
    path = env.triangular_paths[int(path_index)]
    tokens = [path[0], path[1], path[2]]
    pool_ids = path[3]
    product = 1.0
    for token_in, pool_id in zip(tokens, pool_ids):
        rate = spot_rate_for_pool(env, pool_id, token_in)
        if rate <= 0:
            return 0.0
        product *= rate
    return product


def path_min_normalized_liquidity(env: Any, path_index: int) -> float:
    path = env.triangular_paths[int(path_index)]
    tokens = [path[0], path[1], path[2]]
    vals = []
    for token_in, pool_id in zip(tokens, path[3]):
        state = env.current_pools.get(pool_id)
        if not state:
            continue
        reserve_in, reserve_out, token_out = reserves_for_direction(state, token_in)
        reference_in = float(env.token_amount_references.get(token_in, 0.0))
        reference_out = float(env.token_amount_references.get(token_out, 0.0)) if token_out else 0.0
        if reserve_in > 0 and reference_in > 0:
            vals.append(reserve_in / reference_in)
        if reserve_out > 0 and reference_out > 0:
            vals.append(reserve_out / reference_out)
    return float(min(vals)) if vals else 0.0


def path_heuristic_score(env: Any, path_index: int) -> float:
    product = path_spot_product(env, path_index)
    if product <= 0:
        return -float("inf")
    liq = path_min_normalized_liquidity(env, path_index)
    return (product - 1.0) * max(liq, 0.0)


def selection_deadline_expired(env: Any) -> bool:
    if bool(getattr(env, "driver_disable_selection_deadline", False)):
        return False
    elapsed = float(env.cumulative_decision_time_seconds)
    if bool(getattr(env, "_decision_clock_active", False)):
        elapsed += max(0.0, float(env._clock()) - float(env._decision_clock_started_at))
    return elapsed > float(env.block_time_seconds)


def evaluate_path_grid_vectorized(
    env: Any, path_index: int, fractions: Sequence[float]
) -> Tuple[Optional[int], float, float]:
    fraction_array = np.asarray(fractions, dtype=float)
    if fraction_array.ndim != 1 or fraction_array.size == 0:
        return (None, -float("inf"), 0.0)
    path = env.triangular_paths[int(path_index)]
    start_balance = float(env.current_balance.get(path[0], 0.0))
    coefficients = env._path_composition_coefficients(int(path_index))
    if coefficients is None or start_balance <= 0.0:
        return (None, -float("inf"), 0.0)
    coefficient_a, coefficient_b, coefficient_c = coefficients
    amounts = start_balance * fraction_array
    denominator = coefficient_b + coefficient_c * amounts
    valid = (amounts > 0.0) & (denominator > 0.0)
    outputs = np.zeros_like(amounts)
    outputs[valid] = coefficient_a * amounts[valid] / denominator[valid]
    gross = (outputs - amounts) * float(env.prices.get(path[0], 0.0))
    net = gross - float(env.current_gas.gas_cost_usd)
    net[~valid] = -np.inf
    best_position = int(np.argmax(net))
    if not np.isfinite(net[best_position]):
        return (None, -float("inf"), 0.0)
    return (best_position, float(net[best_position]), float(gross[best_position]))


def select_best_over_grid(
    env: Any, path_indices: Iterable[int], fractions: Sequence[float], method_name: str
) -> Selection:
    path_indices = [int(index) for index in path_indices]
    best_idx: Optional[int] = None
    best_frac = float(fractions[0]) if fractions else float(env.amount_min)
    best_net = -float("inf")
    best_gross = 0.0
    evaluated_paths = 0
    for path_index in path_indices:
        if selection_deadline_expired(env):
            break
        position, net, gross = evaluate_path_grid_vectorized(env, int(path_index), fractions)
        evaluated_paths += 1
        if position is not None and net > best_net:
            best_net = net
            best_gross = gross
            best_idx = int(path_index)
            best_frac = float(fractions[position])
    if best_idx is None or not np.isfinite(best_net) or best_net <= float(env.min_profit_usd):
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            expected_net_profit_usd=max(best_net, 0.0) if np.isfinite(best_net) else 0.0,
            expected_gross_profit_usd=best_gross,
            selected_by=method_name,
            note=(
                "deadline reached before a positive candidate"
                if evaluated_paths < len(path_indices)
                else "no positive candidate"
            ),
            path_screen_count=evaluated_paths,
            amount_evaluation_count=evaluated_paths * len(fractions),
            cpmm_evaluation_count=evaluated_paths * len(fractions),
        )
    return Selection(
        path_index=best_idx,
        amount_fraction=best_frac,
        expected_net_profit_usd=best_net,
        expected_gross_profit_usd=best_gross,
        selected_by=method_name,
        note=(
            "best verified candidate before deadline"
            if evaluated_paths < len(path_indices)
            else "best verified candidate"
        ),
        path_screen_count=evaluated_paths,
        amount_evaluation_count=evaluated_paths * len(fractions),
        cpmm_evaluation_count=evaluated_paths * len(fractions),
    )


def greedy_selection(env: Any, args: argparse.Namespace) -> Selection:
    fractions = method_fraction_grid(env, args, "Greedy")
    ranked_scores = sorted(
        (
            (path_spot_product(env, path_index), path_index)
            for path_index in range(len(env.triangular_paths))
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    scanned = len(ranked_scores)
    path_screen_count = scanned
    amount_evaluation_count = 0
    cpmm_evaluation_count = 0
    for rank, (spot_product, path_index) in enumerate(ranked_scores, start=1):
        if spot_product <= 0:
            continue
        sel = select_best_over_grid(env, [path_index], fractions, "Greedy")
        path_screen_count += int(sel.path_screen_count)
        amount_evaluation_count += int(sel.amount_evaluation_count)
        cpmm_evaluation_count += int(sel.cpmm_evaluation_count)
        sel.path_screen_count = path_screen_count
        sel.amount_evaluation_count = amount_evaluation_count
        sel.cpmm_evaluation_count = cpmm_evaluation_count
        sel.note = f"greedy path rank={rank}; fine sizing grid={len(fractions)}"
        if sel.has_trade:
            return sel
    return Selection(
        path_index=None,
        amount_fraction=0.0,
        selected_by="Greedy",
        note="no profitable path after greedy ranking",
        path_screen_count=path_screen_count,
        amount_evaluation_count=amount_evaluation_count,
        cpmm_evaluation_count=cpmm_evaluation_count,
    )


def grid_search_selection(env: Any, args: argparse.Namespace) -> Selection:
    fractions = method_fraction_grid(env, args, "Grid_Search")
    sel = select_best_over_grid(env, range(len(env.triangular_paths)), fractions, "Grid_Search")
    sel.note = (
        f"exhaustive uniform grid: {len(env.triangular_paths)} paths x {len(fractions)} fractions"
    )
    return sel


def heuristic_selection(env: Any, args: argparse.Namespace) -> Selection:
    scored = [(path_heuristic_score(env, i), i) for i in range(len(env.triangular_paths))]
    scored.sort(reverse=True, key=lambda x: x[0])
    top_k = max(1, int(args.heuristic_top_k))
    candidate_indices = [i for score, i in scored[:top_k] if np.isfinite(score)]
    if not candidate_indices:
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            selected_by="Heuristic",
            note="no finite heuristic path score",
            path_screen_count=len(scored),
        )
    fractions = method_fraction_grid(env, args, "Heuristic")
    sel = select_best_over_grid(env, candidate_indices, fractions, "Heuristic")
    sel.path_screen_count += len(scored)
    sel.note = (
        f"top-{len(candidate_indices)} heuristic paths verified on {len(fractions)} fractions"
    )
    return sel


def simulate_prefix_score(env: Any, path_index: int, legs: int, fraction: float) -> float:
    path = env.triangular_paths[int(path_index)]
    pool_ids = list(path[3])
    pool_state = {
        pid: copy.deepcopy(env.current_pools[pid]) for pid in pool_ids if pid in env.current_pools
    }
    tokens_expected = [path[0], path[1], path[2], path[0]]
    amount = float(env.current_balance.get(path[0], 0.0)) * float(fraction)
    if amount <= 0:
        return -float("inf")
    start_amount = amount
    try:
        for i in range(int(legs)):
            pool_id = pool_ids[i]
            token_in = tokens_expected[i]
            amount_out, token_out, ok = env._execute_swap_on_state(
                pool_id, token_in, amount, pool_state
            )
            if not ok or token_out != tokens_expected[i + 1]:
                return -float("inf")
            amount = float(amount_out)
        prefix_ratio = amount / max(start_amount, EPS)
        lookahead = 1.0
        for j in range(int(legs), 3):
            rate = spot_rate_for_pool(env, pool_ids[j], tokens_expected[j])
            if rate <= 0:
                return -float("inf")
            lookahead *= rate
        return prefix_ratio * lookahead
    except Exception:
        return -float("inf")


def beam_search_selection(env: Any, args: argparse.Namespace) -> Selection:
    initial_width = max(1, int(args.beam_width))
    probe_fraction = min(
        max(float(args.beam_probe_fraction), float(env.amount_min)), float(env.amount_max)
    )
    fractions = method_fraction_grid(env, args, "Beam_Search")
    path_count = len(env.triangular_paths)
    width = min(initial_width, path_count)
    verified: set = set()
    prefix_evals = 0
    amount_evals = 0
    cpmm_evals = 0
    widths: List[int] = []
    while width > 0 and (not selection_deadline_expired(env)):
        widths.append(width)
        candidates = list(range(path_count))
        for legs in (1, 2, 3):
            scored: List[Tuple[float, int]] = []
            for path_index in candidates:
                if selection_deadline_expired(env):
                    break
                scored.append(
                    (simulate_prefix_score(env, path_index, legs, probe_fraction), path_index)
                )
            prefix_evals += len(scored)
            scored.sort(reverse=True, key=lambda item: item[0])
            candidates = [path_index for score, path_index in scored[:width] if np.isfinite(score)]
            if not candidates or selection_deadline_expired(env):
                break
        new_candidates = [index for index in candidates if index not in verified]
        verified.update(new_candidates)
        if new_candidates and (not selection_deadline_expired(env)):
            selection = select_best_over_grid(env, new_candidates, fractions, "Beam_Search")
            amount_evals += selection.amount_evaluation_count
            cpmm_evals += selection.cpmm_evaluation_count
            if selection.has_trade:
                selection.path_screen_count = path_count
                selection.amount_evaluation_count = amount_evals
                selection.cpmm_evaluation_count = cpmm_evals
                selection.graph_operation_count = prefix_evals
                selection.note = f"deterministic beam widths={widths}; prefix_evals={prefix_evals}; verified_paths={len(verified)}; fractions={len(fractions)}"
                return selection
        if width >= path_count or len(verified) >= path_count:
            break
        width = min(path_count, width * 2)
    return Selection(
        path_index=None,
        amount_fraction=0.0,
        selected_by="Beam_Search",
        note=f"no profitable candidate; beam widths={widths}; verified_paths={len(verified)}",
        path_screen_count=path_count,
        amount_evaluation_count=amount_evals,
        cpmm_evaluation_count=cpmm_evals,
        graph_operation_count=prefix_evals,
    )


def nlp_scipy_minimize_selection(env: Any, args: argparse.Namespace) -> Selection:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            selected_by="NLP_SciPy_Minimize",
            note="scipy not available",
            error=str(exc),
        )
    scored = [(path_heuristic_score(env, i), i) for i in range(len(env.triangular_paths))]
    scored.sort(reverse=True, key=lambda x: x[0])
    path_limit = int(args.nlp_path_limit)
    candidate_paths = [
        i
        for _score, i in (scored[:path_limit] if path_limit > 0 else scored)
        if np.isfinite(_score)
    ]
    lo = float(env.amount_min)
    hi = float(env.amount_max)
    best_idx: Optional[int] = None
    best_frac = lo
    best_net = -float("inf")
    best_gross = 0.0
    optimizer_function_evaluations = 0
    verification_evaluations = 0
    paths_considered = 0
    deadline_hit = False
    for path_index in candidate_paths:
        if selection_deadline_expired(env):
            deadline_hit = True
            break
        paths_considered += 1
        path = env.triangular_paths[int(path_index)]
        start_balance = float(env.current_balance.get(path[0], 0.0))
        price = float(env.prices.get(path[0], 0.0))
        coefficients = env._path_composition_coefficients(int(path_index))
        if coefficients is None or start_balance <= EPS or price <= EPS:
            continue
        coefficient_a, coefficient_b, coefficient_c = coefficients
        gas_cost = float(env.current_gas.gas_cost_usd)
        callback_counts = {
            "objective": 0,
            "objective_jacobian": 0,
            "constraint": 0,
            "constraint_jacobian": 0,
        }

        def net_and_gradient(fraction: float) -> Tuple[float, float, float]:
            fraction = float(np.clip(fraction, lo, hi))
            amount = start_balance * fraction
            denominator = coefficient_b + coefficient_c * amount
            if amount <= 0.0 or denominator <= 0.0:
                return (-float("inf"), 0.0, -float("inf"))
            output = coefficient_a * amount / denominator
            gross = (output - amount) * price
            derivative = (
                price
                * start_balance
                * (coefficient_a * coefficient_b / (denominator * denominator) - 1.0)
            )
            return (gross - gas_cost, derivative, gross)

        def objective(x: Sequence[float]) -> float:
            nonlocal deadline_hit
            callback_counts["objective"] += 1
            if selection_deadline_expired(env):
                deadline_hit = True
                return 1e18
            net, _derivative, _gross = net_and_gradient(float(np.asarray(x).reshape(-1)[0]))
            return -net if np.isfinite(net) else 1e18

        def objective_jacobian(x: Sequence[float]) -> np.ndarray:
            nonlocal deadline_hit
            callback_counts["objective_jacobian"] += 1
            if selection_deadline_expired(env):
                deadline_hit = True
                return np.asarray([0.0], dtype=float)
            _net, derivative, _gross = net_and_gradient(float(np.asarray(x).reshape(-1)[0]))
            return np.asarray([-derivative], dtype=float)

        def profit_constraint(x: Sequence[float]) -> float:
            nonlocal deadline_hit
            callback_counts["constraint"] += 1
            if selection_deadline_expired(env):
                deadline_hit = True
                return -1.0
            net, _derivative, _gross = net_and_gradient(float(np.asarray(x).reshape(-1)[0]))
            return net - float(env.min_profit_usd)

        def profit_constraint_jacobian(x: Sequence[float]) -> np.ndarray:
            nonlocal deadline_hit
            callback_counts["constraint_jacobian"] += 1
            if selection_deadline_expired(env):
                deadline_hit = True
                return np.asarray([0.0], dtype=float)
            _net, derivative, _gross = net_and_gradient(float(np.asarray(x).reshape(-1)[0]))
            return np.asarray([derivative], dtype=float)

        stationary_amount = (
            (math.sqrt(coefficient_a * coefficient_b) - coefficient_b) / coefficient_c
            if coefficient_a > coefficient_b and coefficient_c > EPS
            else start_balance * lo
        )
        stationary_fraction = stationary_amount / max(start_balance, EPS)
        fractions_to_check = _dedupe_sorted_fractions([lo, hi, stationary_fraction], lo, hi)
        feasible_starts = [
            fraction
            for fraction in fractions_to_check
            if net_and_gradient(fraction)[0] > float(env.min_profit_usd)
        ]
        if feasible_starts and (not deadline_hit):
            try:
                result = minimize(
                    objective,
                    x0=np.asarray([max(feasible_starts, key=lambda f: net_and_gradient(f)[0])]),
                    jac=objective_jacobian,
                    method=str(args.nlp_method),
                    bounds=[(lo, hi)],
                    constraints=(
                        {
                            "type": "ineq",
                            "fun": profit_constraint,
                            "jac": profit_constraint_jacobian,
                        },
                    ),
                    options={
                        "maxiter": int(args.nlp_maxiter),
                        "ftol": float(args.nlp_ftol),
                        "disp": False,
                    },
                )
                if getattr(result, "x", None) is not None:
                    fractions_to_check.append(float(result.x[0]))
            except Exception:
                pass
        optimizer_function_evaluations += sum(callback_counts.values())
        verified_fractions = _dedupe_sorted_fractions(fractions_to_check, lo, hi)
        for frac in verified_fractions:
            if selection_deadline_expired(env):
                deadline_hit = True
                break
            verification_evaluations += 1
            net, _derivative, gross = net_and_gradient(frac)
            if net > best_net:
                best_net = net
                best_gross = gross
                best_idx = int(path_index)
                best_frac = float(frac)
        if deadline_hit:
            break
    total_evaluations = optimizer_function_evaluations + verification_evaluations
    selection = Selection(
        path_index=best_idx,
        amount_fraction=best_frac if best_idx is not None else 0.0,
        expected_net_profit_usd=(
            best_net
            if best_idx is not None and np.isfinite(best_net)
            else max(best_net, 0.0) if np.isfinite(best_net) else 0.0
        ),
        expected_gross_profit_usd=best_gross,
        selected_by="NLP_SciPy_Minimize",
        note=(
            "SLSQP continuous sizing using exact CPMM composition and analytic derivative"
            if best_idx is not None and best_net > float(env.min_profit_usd)
            else "no positive NLP candidate before deadline"
        ),
        path_screen_count=paths_considered,
        amount_evaluation_count=total_evaluations,
        cpmm_evaluation_count=total_evaluations,
        optimizer_function_evaluation_count=optimizer_function_evaluations,
    )
    if best_idx is None or not np.isfinite(best_net) or best_net <= float(env.min_profit_usd):
        selection.path_index = None
        selection.amount_fraction = 0.0
    return selection


def bellman_ford_networkx_selection(env: Any, args: argparse.Namespace) -> Selection:
    try:
        import networkx as nx
    except ImportError as exc:
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            selected_by="Bellman_Ford_NetworkX",
            note="networkx not available",
            error=str(exc),
        )
    graph = nx.DiGraph()
    edge_best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    directed_pool_edge_screens = 0
    for pool in env.pool_list:
        state = env.current_pools[pool.pool_id]
        for token_in in (pool.token0, pool.token1):
            directed_pool_edge_screens += 1
            reserve_in, reserve_out, token_out = reserves_for_direction(state, token_in)
            if reserve_in <= 0 or reserve_out <= 0 or (not token_out):
                continue
            rate = spot_rate_for_pool(env, pool.pool_id, token_in)
            if rate <= 0:
                continue
            key = (token_in, token_out)
            if key not in edge_best or rate > edge_best[key][0]:
                edge_best[key] = (rate, pool.pool_id)
    for (u, v), (rate, pool_id) in edge_best.items():
        graph.add_edge(u, v, weight=-math.log(max(rate, EPS)), rate=rate, pool_id=pool_id)
    try:
        has_negative_cycle = bool(nx.negative_edge_cycle(graph, weight="weight"))
    except Exception:
        has_negative_cycle = False
    graph_operations = directed_pool_edge_screens + int(graph.number_of_edges())
    if not has_negative_cycle:
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            selected_by="Bellman_Ford_NetworkX",
            note="no negative log-rate cycle detected",
            graph_operation_count=graph_operations,
        )
    cycle_nodes: List[str] = []
    source_searches = 0
    if hasattr(nx, "find_negative_cycle"):
        for source in list(graph.nodes):
            source_searches += 1
            try:
                cyc = nx.find_negative_cycle(graph, source, weight="weight")
                if cyc:
                    cycle_nodes = [str(x) for x in cyc]
                    break
            except Exception:
                continue
    graph_operations += source_searches
    path_scores = {
        index: path_spot_product(env, index) for index in range(len(env.triangular_paths))
    }
    cycle_candidates: List[int] = []
    if len(cycle_nodes) >= 4:
        triples = set()
        for i in range(len(cycle_nodes) - 3):
            if cycle_nodes[i] == cycle_nodes[i + 3]:
                triples.add((cycle_nodes[i], cycle_nodes[i + 1], cycle_nodes[i + 2]))
        for idx, path in enumerate(env.triangular_paths):
            if (path[0], path[1], path[2]) in triples and path_scores[idx] > 1.0:
                cycle_candidates.append(idx)
    all_positive_candidates = sorted(
        [index for index, score in path_scores.items() if score > 1.0],
        key=lambda index: path_scores[index],
        reverse=True,
    )
    if not all_positive_candidates:
        return Selection(
            path_index=None,
            amount_fraction=0.0,
            selected_by="Bellman_Ford_NetworkX",
            note="negative cycle exists but no triangular candidate verified",
            path_screen_count=len(env.triangular_paths),
            graph_operation_count=graph_operations,
        )
    cycle_candidates = sorted(
        dict.fromkeys(cycle_candidates), key=lambda index: path_scores[index], reverse=True
    )
    candidates = cycle_candidates or all_positive_candidates
    if int(args.bellman_path_limit) > 0:
        candidates = candidates[: int(args.bellman_path_limit)]
        all_positive_candidates = all_positive_candidates[: int(args.bellman_path_limit)]
    fractions = method_fraction_grid(env, args, "Bellman_Ford_NetworkX")
    sel = select_best_over_grid(env, candidates, fractions, "Bellman_Ford_NetworkX")
    fallback_used = False
    if not sel.has_trade and bool(cycle_candidates) and (not selection_deadline_expired(env)):
        primary_set = set(candidates)
        fallback_candidates = [
            index for index in all_positive_candidates if index not in primary_set
        ]
        if fallback_candidates:
            fallback = select_best_over_grid(
                env, fallback_candidates, fractions, "Bellman_Ford_NetworkX"
            )
            fallback.path_screen_count += sel.path_screen_count
            fallback.amount_evaluation_count += sel.amount_evaluation_count
            fallback.cpmm_evaluation_count += sel.cpmm_evaluation_count
            sel = fallback
            fallback_used = True
    sel.path_screen_count += len(env.triangular_paths)
    sel.graph_operation_count += graph_operations
    sel.note = f"NetworkX Bellman-Ford marginal negative-cycle screening plus CPMM/gas verification; primary_candidates={len(candidates)}; fallback_all_positive_cycles={fallback_used}; fractions={len(fractions)}"
    return sel


def classical_selector(method: str) -> Any:
    selectors = {
        "Greedy": greedy_selection,
        "Heuristic": heuristic_selection,
        "Beam_Search": beam_search_selection,
        "Grid_Search": grid_search_selection,
        "NLP_SciPy_Minimize": nlp_scipy_minimize_selection,
        "Bellman_Ford_NetworkX": bellman_ford_networkx_selection,
    }
    if method not in selectors:
        raise ValueError(f"No classical selector is registered for {method!r}.")
    return selectors[method]


def rl_selection_and_action(sim, env, obs, args, method):
    model = MODEL_CACHE.get(method, resolve_model_path(args, method), "cpu", sim, env)
    started = time.perf_counter()
    action, _ = model.predict(obs, deterministic=True)
    elapsed = time.perf_counter() - started
    path_index, fraction, _ = env._decode_action(action)
    return (
        Selection(path_index=path_index, amount_fraction=fraction, selected_by=method),
        action,
        elapsed,
    )


def warm_method_runtime(
    sim: Any, env: Any, obs: np.ndarray, args: argparse.Namespace, method: str
) -> None:
    if method in RL_METHODS:
        model_path = resolve_model_path(args, method)
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"{method} model zip not found for warm-up.")
        model = MODEL_CACHE.get(method, Path(model_path), str(args.device), sim, env)
        model.predict(obs, deterministic=bool(args.deterministic_rl))
        return
    if method == "NLP_SciPy_Minimize":
        __import__("scipy.optimize")
    elif method == "Bellman_Ford_NetworkX":
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edge("warm_a", "warm_b", weight=1.0)
        nx.single_source_bellman_ford_path_length(graph, "warm_a", weight="weight")
    if env.triangular_paths:
        evaluate_path_grid_vectorized(env, 0, np.linspace(0.01, 1.0, 16, dtype=float))


def classify_arbitrage_type(env: Any, path: Any) -> Tuple[str, str]:
    if not path:
        return ("no_trade", "")
    try:
        pool_ids = list(path[3])
    except Exception:
        return ("unknown", "")
    dexes = []
    for pool_id in pool_ids:
        pool = env.pool_template.get(pool_id)
        dexes.append(str(getattr(pool, "dex_type", "unknown")) if pool is not None else "unknown")
    unique = sorted(set(dexes))
    if len(unique) <= 1:
        return ("single_dex", "|".join(dexes))
    return ("cross_dex", "|".join(dexes))


def cached_common_stop_status(env: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    state_key = env._timed_common_stop_state_key()
    if state_key != env._cached_common_stop_state_key:
        reason, best = env._timed_common_stop_check()
        env._cached_common_stop_reason = str(reason)
        env._cached_common_stop_best = copy.deepcopy(best)
        env._cached_common_stop_state_key = state_key
    return (str(env._cached_common_stop_reason), copy.deepcopy(env._cached_common_stop_best))


def run_episode(
    sim: Any, args: argparse.Namespace, scenario: Scenario, seed: int, method: str
) -> Dict[str, Any]:
    episode_start = time.perf_counter()
    env: Optional[Any] = None
    attempts = 0
    timed_attempts = 0
    successful_transactions = 0
    rejected_transactions = 0
    no_candidate_decisions = 0
    single_dex_transactions = 0
    cross_dex_transactions = 0
    unknown_dex_transactions = 0
    committed_transactions: List[Dict[str, Any]] = []
    total_selection_time = 0.0
    total_complete_decision_time = 0.0
    previous_cumulative_decision_time = 0.0
    commits_within_c6 = 0
    c6_expired_before_commit = False
    c6_deadline_reached_after_commit = False
    workload_totals = {field: 0 for field in WORKLOAD_FIELDS}
    stop_reason = ""
    error_type = ""
    error_message = ""
    error_text = ""
    try:
        frozen_clock = None
        env, obs, _reset_info = make_driver_env(
            sim, args, scenario, int(seed), method, clock=frozen_clock
        )
        env.driver_disable_selection_deadline = bool(False)
        warm_method_runtime(sim, env, obs, args, method)
        obs, _reset_info = sim.reset_env_compat(env, seed=int(seed))
        while True:
            if not False and env.cumulative_decision_time_seconds > float(
                scenario.c6_latency_budget_seconds
            ):
                stop_reason = "C6_wall_clock_budget_exhausted"
                break
            if str(env._cached_common_stop_reason):
                stop_reason = str(env._cached_common_stop_reason)
                break
            attempts += 1
            env.resume_decision_clock()
            if method in RL_METHODS:
                selection, action, selection_time = rl_selection_and_action(
                    sim, env, obs, args, method
                )
                if action is None:
                    raise RuntimeError(selection.error or f"{method} produced no action.")
            else:
                selection_started = time.perf_counter()
                selection = classical_selector(method)(env, args)
                selection_time = time.perf_counter() - selection_started
                if selection.has_trade:
                    action = env._action_from_path_amount(
                        int(selection.path_index), float(selection.amount_fraction)
                    )
                else:
                    action = None
            total_selection_time += float(selection_time)
            for field in WORKLOAD_FIELDS:
                workload_totals[field] += int(getattr(selection, field))
            if selection.error:
                raise RuntimeError(f"{method} candidate selection failed: {selection.error}")
            if action is None:
                env._finish_selection_timing()
                natural_stop_reason, _best = cached_common_stop_status(env)
                cumulative_time = float(env.cumulative_decision_time_seconds)
                complete_attempt_time = max(
                    0.0, cumulative_time - previous_cumulative_decision_time
                )
                total_complete_decision_time += max(0.0, complete_attempt_time)
                previous_cumulative_decision_time = cumulative_time
                timed_attempts += 1
                if not False and env.cumulative_decision_time_seconds > float(
                    scenario.c6_latency_budget_seconds
                ):
                    stop_reason = "C6_wall_clock_budget_exhausted"
                    break
                elif natural_stop_reason:
                    stop_reason = str(natural_stop_reason)
                    break
                else:
                    no_candidate_decisions += 1
                    continue
            obs, _reward, done, info = sim.step_env_compat(env, action)
            commits_within_c6 += int(bool(info.get("c6_commit_within_budget", False)))
            c6_expired_before_commit = bool(
                c6_expired_before_commit or info.get("c6_expired_before_commit", False)
            )
            c6_deadline_reached_after_commit = bool(
                c6_deadline_reached_after_commit or info.get("c6_expired_after_commit", False)
            )
            if bool(info.get("submitted_onchain", False)):
                raise RuntimeError(
                    "Controlled simulator attempted to report an on-chain submission."
                )
            cumulative_time = float(
                info.get("cumulative_decision_time_seconds", env.cumulative_decision_time_seconds)
            )
            complete_attempt_time = max(0.0, cumulative_time - previous_cumulative_decision_time)
            total_complete_decision_time += complete_attempt_time
            previous_cumulative_decision_time = cumulative_time
            timed_attempts += 1
            if bool(info.get("success", False)):
                successful_transactions += 1
                arbitrage_type, dex_sequence = classify_arbitrage_type(env, info.get("path"))
                if arbitrage_type == "single_dex":
                    single_dex_transactions += 1
                elif arbitrage_type == "cross_dex":
                    cross_dex_transactions += 1
                else:
                    unknown_dex_transactions += 1
                executed_path = info.get("path")
                if not executed_path or len(executed_path) < 4:
                    raise RuntimeError(
                        "A committed transaction did not expose its structural path."
                    )
                token_symbols = [str(token) for token in executed_path[:3]]
                pool_ids = [str(pool) for pool in executed_path[3]]
                committed_transactions.append(
                    {
                        "transaction_order": int(successful_transactions),
                        "path_index": int(info.get("path_index", -1)),
                        "token_route": str(
                            info.get("path_tokens") or env.format_path_tokens(executed_path)
                        ),
                        "token_symbols": token_symbols + [token_symbols[0]],
                        "pool_ids": pool_ids,
                        "dex_sequence": str(dex_sequence),
                        "arbitrage_type": str(arbitrage_type),
                        "amount_fraction": float(info.get("amount_fraction", 0.0)),
                        "input_amount_token": float(info.get("amount_in", 0.0)),
                        "gross_profit_usd": float(info.get("gross_profit_usd", 0.0)),
                        "gas_cost_usd": float(info.get("gas_charged_usd", 0.0)),
                        "after_gas_profit_usd": float(info.get("net_profit_usd", 0.0)),
                        "learning_reward": float(_reward),
                        "selection_time_seconds": float(selection_time),
                        "complete_attempt_time_seconds": float(complete_attempt_time),
                        "commit_time_upper_bound_seconds": float(cumulative_time),
                        "c6_budget_seconds": float(scenario.c6_latency_budget_seconds),
                        "operational_deadline_enforced": bool(not False),
                        "c6_commit_within_budget": bool(info.get("c6_commit_within_budget", False)),
                    }
                )
            rejected_transactions = int(env.failed_actions)
            if bool(done):
                stop_reason = str(info.get("stop_reason") or info.get("reason") or "terminated")
                break
    except Exception as exc:
        stop_reason = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        error_text = traceback.format_exc(limit=8)
    finally:
        episode_wall_time = time.perf_counter() - episode_start
    episode_valid = not bool(error_text)
    if env is not None:
        wallet_file = str(getattr(env, "wallet_name", ""))
        dataset_file = str(getattr(env, "pool_dataset_file", ""))
        profit_usd = float(env.total_profit_usd) if episode_valid else float("nan")
        gross_profit_usd = float(env.total_gross_usd) if episode_valid else float("nan")
        gas_charged_usd = float(env.total_gas_usd) if episode_valid else float("nan")
        gas_cost_usd = float(env.current_gas.gas_cost_usd)
        normalization_clipping_count = int(env.normalization_clipping_count)
        normalization_clipping_by_feature = json.dumps(
            dict(sorted(env.normalization_clipping_by_feature.items())), sort_keys=True
        )
        loaded_pool_count = int(len(env.pool_list))
        participating_pool_count = int(
            len({pool_id for path in env.triangular_paths for pool_id in path[3]})
        )
        structural_path_count = int(len(env.triangular_paths))
        dataset_mode = str(getattr(env, "dataset_mode", "real_snapshot"))
        gas_price_gwei = float(env.current_gas.gas_price_gwei)
        gas_used = float(env.current_gas.gas_used)
        gas_source_row_index = int(env.current_gas.source_row_index)
        rejected_transactions = int(env.failed_actions)
        try:
            env.close()
        except Exception:
            pass
    else:
        wallet_file = ""
        dataset_file = str(Path(args.dataset_dir).expanduser() / POOL_DATA_FILENAME)
        profit_usd = float("nan")
        gross_profit_usd = float("nan")
        gas_charged_usd = float("nan")
        gas_cost_usd = float("nan")
        normalization_clipping_count = 0
        normalization_clipping_by_feature = "{}"
        loaded_pool_count = 0
        participating_pool_count = 0
        structural_path_count = 0
        dataset_mode = ""
        gas_price_gwei = float("nan")
        gas_used = float("nan")
        gas_source_row_index = -1
    realized_routes = single_dex_transactions + cross_dex_transactions + unknown_dex_transactions
    if single_dex_transactions and cross_dex_transactions:
        route_class = "mixed"
    elif single_dex_transactions:
        route_class = "single_dex"
    elif cross_dex_transactions:
        route_class = "cross_dex"
    elif successful_transactions:
        route_class = "unknown"
    else:
        route_class = "no_trade"
    roi_pct = (
        profit_usd / max(float(scenario.balance_usd), EPS) * 100.0
        if math.isfinite(profit_usd)
        else float("nan")
    )
    gas_share_pct = (
        100.0 * gas_charged_usd / gross_profit_usd
        if math.isfinite(gas_charged_usd)
        and math.isfinite(gross_profit_usd)
        and (gross_profit_usd > EPS)
        else float("nan")
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "dimension": scenario.dimension,
        "ablation_value": scenario.ablation_value,
        "ablation_label": scenario.ablation_label,
        "balance_usd": scenario.balance_usd,
        "c3_liquidity_depth": scenario.c3_liquidity_depth,
        "gas_multiplier": scenario.gas_multiplier,
        "c6_latency_budget_seconds": scenario.c6_latency_budget_seconds,
        "seed": int(seed),
        "method": method,
        "method_display": METHOD_DISPLAY.get(method, method),
        "episode_id": f"{scenario.dimension}:{scenario.ablation_label}:{method}:{seed}",
        "episode_valid": episode_valid,
        "wallet_file": wallet_file,
        "success": bool(episode_valid and successful_transactions > 0),
        "submitted_onchain": False,
        "simulated_execution": bool(successful_transactions > 0),
        "dataset_file": dataset_file,
        "profit_usd": profit_usd,
        "gross_profit_usd": gross_profit_usd,
        "gas_cost_usd": gas_cost_usd,
        "gas_charged_usd": gas_charged_usd,
        "roi_pct": roi_pct,
        "gas_cost_share_pct": gas_share_pct,
        "local_decision_time_seconds": (
            total_complete_decision_time / timed_attempts if timed_attempts else 0.0
        ),
        "mean_complete_decision_time_seconds": (
            total_complete_decision_time / timed_attempts if timed_attempts else 0.0
        ),
        "mean_selection_time_seconds": total_selection_time / attempts if attempts else 0.0,
        "total_selection_time_seconds": float(total_selection_time),
        "total_complete_decision_time_seconds": (
            float(env.cumulative_decision_time_seconds)
            if env is not None
            else float(previous_cumulative_decision_time)
        ),
        "common_stop_check_time_seconds": (
            float(env.cumulative_common_stop_check_time_seconds)
            if env is not None
            else float("nan")
        ),
        "common_stop_check_phase_seconds": (
            json.dumps(dict(sorted(env.common_stop_check_phase_seconds.items())), sort_keys=True)
            if env is not None
            else "{}"
        ),
        "episode_wall_time_including_setup_seconds": float(episode_wall_time),
        "c6_episode_completed_without_expiry": bool(
            episode_valid and (not False) and (not str(stop_reason).startswith("C6_"))
        ),
        "c6_commits_within_budget": int(commits_within_c6),
        "c6_expired_before_commit": bool(c6_expired_before_commit),
        "c6_deadline_reached_after_commit": bool(c6_deadline_reached_after_commit),
        "arbitrage_type": route_class,
        "single_dex_transactions": int(single_dex_transactions),
        "cross_dex_transactions": int(cross_dex_transactions),
        "unknown_dex_transactions": int(unknown_dex_transactions),
        "cross_dex_share_pct": (
            100.0 * cross_dex_transactions / realized_routes if realized_routes else float("nan")
        ),
        "executed_transactions": int(successful_transactions),
        "executed_transactions_json": json.dumps(
            committed_transactions, separators=(",", ":"), sort_keys=True, allow_nan=False
        ),
        "decision_attempts": int(attempts),
        "screened_rejections": int(rejected_transactions),
        "no_candidate_decisions": int(no_candidate_decisions),
        **{f"workload_{field}": int(workload_totals[field]) for field in WORKLOAD_FIELDS},
        "stop_reason": stop_reason,
        "terminal_condition": stop_reason,
        "normalization_clipping_count": normalization_clipping_count,
        "normalization_clipping_by_feature": normalization_clipping_by_feature,
        "loaded_pool_count": loaded_pool_count,
        "participating_pool_count": participating_pool_count,
        "structural_path_count": structural_path_count,
        "dataset_mode": dataset_mode,
        "gas_price_gwei": gas_price_gwei,
        "gas_used": gas_used,
        "gas_source_row_index": gas_source_row_index,
        "error_type": error_type,
        "error_message": error_message,
        "error": error_text,
    }


# ---- Full fixed evaluation: 23 settings x 10 seeds x 9 methods ----
from types import SimpleNamespace
import pandas as pd
from chart import pairwise_counts


def default_args(dataset_dir="DATASET", model_dir="models"):
    return SimpleNamespace(
        dataset_dir=str(dataset_dir),
        model_dir=str(model_dir),
        device="cpu",
        action_mode="auto",
        amount_min=0.0,
        amount_max=1.0,
        amount_grid="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1",
        baseline_amount_min=0.0,
        baseline_amount_max=1.0,
        baseline_grid_step=0.001,
        reward_reference_usd=10000.0,
        c1_rejection_reward=0.0,
        max_paths=5000,
        balance_values="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000",
        latency_values="4,6,8,10,12",
        gas_values="1,2,4,8",
        liquidity_values="0.25,0.5,0.75,1",
        only_dimensions="",
        max_scenarios=0,
        heuristic_top_k=20,
        beam_width=40,
        beam_probe_fraction=0.01,
        nlp_path_limit=0,
        nlp_method="SLSQP",
        nlp_maxiter=80,
        nlp_ftol=1e-9,
        bellman_path_limit=0,
        deterministic_rl=True,
    )


def main():
    import train as sim
    from chart import summarize

    parser = argparse.ArgumentParser(
        description="Test nine methods on the fixed 2070-episode design."
    )
    parser.add_argument("--dataset-dir", default="DATASET")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--workbook", default="PPR01_V209_EVALUATION_job9851603.xlsx")
    cli = parser.parse_args()
    args = default_args(cli.dataset_dir, cli.model_dir)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for method in ("PPO", "SAC", "TD3"):
        if not resolve_model_path(args, method).is_file():
            parser.error(f"Missing {resolve_model_path(args, method)}; run train.py first.")
    rows, transactions = [], []
    scenarios = build_scenarios(args)
    for scenario in scenarios:
        for seed in range(40, 50):
            for method in METHOD_ORDER_DEFAULT:
                row = run_episode(sim, args, scenario, seed, method)
                if not row["episode_valid"]:
                    # Do not silently publish a broken method as zero profit.
                    raise RuntimeError(f"{method}, {scenario.dimension}, {seed}: {row['error']}")
                trace = json.loads(row["executed_transactions_json"])
                times = [t["complete_attempt_time_seconds"] for t in trace]
                row["successful_atomic_complete_decision_time_seconds"] = (
                    float(np.mean(times)) if times else np.nan
                )
                row.update(
                    selected_before_test=method in RL_METHODS,
                    test_metrics_used_for_model_selection=False,
                    deadline_free_executed=False,
                    oracle_actor_executed=False,
                    common_stop_check_consumes_c6=True,
                )
                rows.append(row)
                transactions.extend(
                    {"episode_id": row["episode_id"], "method": method, **t} for t in trace
                )
        print(
            f"Completed {scenario.dimension}={scenario.ablation_value}: {len(rows)}/2070",
            flush=True,
        )
    raw = pd.DataFrame(rows)
    keys = ["dimension", "ablation_value", "seed", "method"]
    if len(raw) != 2070 or raw.duplicated(keys).any():
        raise ValueError("Incomplete or duplicate evaluation grid.")
    summary = summarize(raw)
    overall = raw.groupby("method", as_index=False).agg(
        episodes=("profit_usd", "size"),
        mean_profit_usd=("profit_usd", "mean"),
        std_profit_usd=("profit_usd", "std"),
        transactions=("executed_transactions", "sum"),
    )
    tables = {
        "raw_runs": raw,
        "summary_by_value_method": summary,
        "overall_by_method": overall,
        "ppo_pairwise_counts": pairwise_counts(raw),
        "scenario_plan": pd.DataFrame([dataclasses.asdict(s) for s in scenarios]),
        "config": pd.DataFrame(
            [
                {
                    "learner_seed": 20,
                    "training_environment_seed": 20,
                    "validation_seeds": "30-39",
                    "test_seeds": "40-49",
                    "run_identity": "new_streamlined_run_not_historical_job9851603",
                    "simulator_only": True,
                    "ambiguous_ohm_routes": 8,
                }
            ]
        ),
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    transaction_frame = pd.DataFrame(transactions)
    transaction_frame.to_csv(output / "transactions.csv", index=False)
    workbook_raw = raw.copy()
    # Excel has a per-cell text limit; full traces always remain in raw_runs.csv.
    long_trace = workbook_raw.executed_transactions_json.str.len() > 32767
    workbook_raw.loc[long_trace, "executed_transactions_json"] = (
        "See raw_runs.csv / transactions.csv"
    )
    tables["raw_runs"] = workbook_raw
    with pd.ExcelWriter(output / cli.workbook, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"Saved {output / cli.workbook}", flush=True)


if __name__ == "__main__":
    main()
