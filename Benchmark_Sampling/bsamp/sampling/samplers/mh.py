"""Metropolis-Hastings adaptive sampler over allocation space.

Searches the allocation space a = [n_1, ..., n_H] (sum = B) for the
minimum-variance allocation.

Energy:    E(a) = sum_h (W_h^2 * S_h^2 / n_h)  -- stratified variance estimator
Proposal:  move 1 unit from a stratum with n_h > 1 to another (symmetric)
Accept:    min(1, exp(-dE / T))  -- Metropolis criterion
Annealing: T_{t+1} = T_t * anneal_rate

Two layers of randomness:
  - Layer 1 (Allocation): MH searches over allocation vector a (optimisation target)
  - Layer 2 (Realisation): given a, items are drawn within strata (conditional sampling)
"""

from __future__ import annotations

import math
import random
from typing import Any

from bsamp.sampling.samplers.base import BaseSampler
from bsamp.sampling.types import BenchmarkItem, ItemRealization, StratumStats


class MetropolisHastingsSampler(BaseSampler):
    """Metropolis-Hastings sampler over strata allocation space.

    Searches for the optimal integer allocation vector `a = [n_1, ..., n_H]`
    that minimizes the stratified variance estimator.
    """

    def __init__(self, rng_seed: int | None = None, initial_temperature: float = 1.0, anneal_rate: float = 0.99):
        self.rng = random.Random(rng_seed)
        self.temperature = initial_temperature
        self.anneal_rate = anneal_rate

        self.current_allocation: dict[str, int] | None = None
        self._sorted_labels: list[str] = []
        self.step_count = 0
        self.n_accepted = 0
        self.energy_trace: list[float] = []

    def get_state(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "anneal_rate": self.anneal_rate,
            "current_allocation": self.current_allocation.copy() if self.current_allocation else None,
            "sorted_labels": list(self._sorted_labels),
            "step_count": self.step_count,
            "n_accepted": self.n_accepted,
            "energy_trace": self.energy_trace.copy(),
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.temperature = state["temperature"]
        self.anneal_rate = state["anneal_rate"]
        self.current_allocation = state["current_allocation"].copy() if state["current_allocation"] else None
        self._sorted_labels = list(state.get("sorted_labels", []))
        self.step_count = state["step_count"]
        self.n_accepted = state["n_accepted"]
        self.energy_trace = state["energy_trace"].copy()

        rng_state = state["rng_state"]
        if isinstance(rng_state, list):
            rng_state = tuple(rng_state)
            if len(rng_state) > 1 and isinstance(rng_state[1], list):
                inner = list(rng_state)
                inner[1] = tuple(inner[1])
                rng_state = tuple(inner)
        self.rng.setstate(rng_state)

    def _propose(self, current: dict[str, int]) -> dict[str, int]:
        """Move 1 unit from a random donor stratum to a random recipient stratum."""
        proposed = current.copy()
        strata_names = list(proposed.keys())
        
        # Pick donor stratum (must have n_h > 1 to donate)
        donors = [h for h in strata_names if proposed[h] > 1]
        if not donors:
            return proposed  # Cannot propose a move if all have 1
            
        donor = self.rng.choice(donors)
        
        # Pick recipient stratum (different from donor)
        recipients = [h for h in strata_names if h != donor]
        if not recipients:
            return proposed  # Only one stratum
            
        recipient = self.rng.choice(recipients)
        
        proposed[donor] -= 1
        proposed[recipient] += 1
        return proposed

    def _energy(self, allocation: dict[str, int], strata_stats: dict[str, StratumStats], total_population: int) -> float:
        """Calculate the energy of an allocation.
        
        Energy is defined as the stratified variance:
        E(a) = Var(F_hat) = sum_h (W_h^2 * s_h^2 / n_h)
        Lower energy is better.
        """
        if total_population == 0:
            return 0.0
            
        var = 0.0
        for h, n_h in allocation.items():
            if n_h <= 0:
                continue
            stats = strata_stats.get(h)
            if not stats:
                continue

            w_h = stats.population_size / total_population
            s_h_sq = stats.sample_variance if stats.sample_size > 1 else 1.0

            var += (w_h ** 2) * (s_h_sq / n_h)

        return var

    def _accept(self, energy_current: float, energy_proposed: float) -> bool:
        delta = energy_proposed - energy_current
        if delta < 0:
            return True  # Always accept lower energy
            
        # If temperature is close to zero, essentially greedy
        if self.temperature < 1e-8:
            return False
            
        # Otherwise accept with probability exp(-delta / T)
        try:
            prob = math.exp(-delta / self.temperature)
        except OverflowError:
            prob = 0.0  # delta / temperature is very large
            
        return self.rng.random() < prob

    def _initial_allocation(self, strata: dict[str, list[BenchmarkItem]], budget: int) -> dict[str, int]:
        """Proportional allocation as the starting point."""
        total_items = sum(len(items) for items in strata.values())
        if total_items == 0:
            return {h: 0 for h in strata}
            
        allocation = {}
        remaining_budget = budget
        
        # Sort strata to make it deterministic
        sorted_strata = sorted(strata.keys())
        
        # Give at least 1 to each stratum if possible
        for h in sorted_strata:
            allocation[h] = 1
            remaining_budget -= 1
            
        if remaining_budget < 0:
            # Budget is less than number of strata, just pick randomly
            allocation = {h: 0 for h in strata}
            chosen = self.rng.sample(sorted_strata, budget)
            for h in chosen:
                allocation[h] = 1
            return allocation

        # Distribute the rest proportionally
        for h in sorted_strata:
            pop = len(strata[h])
            # Desired proportional count beyond the 1 already given
            prop = int(budget * (pop / total_items)) - 1
            prop = max(0, prop)
            add = min(prop, remaining_budget)
            allocation[h] += add
            remaining_budget -= add
            
        # Distribute any leftovers round-robin
        while remaining_budget > 0:
            for h in sorted_strata:
                if remaining_budget <= 0:
                    break
                allocation[h] += 1
                remaining_budget -= 1
                
        return allocation

    def select(
        self,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        budget: int,
    ) -> ItemRealization:

        if self.current_allocation is None or sum(self.current_allocation.values()) != budget:
            self.current_allocation = self._initial_allocation(strata, budget)

        if not self._sorted_labels:
            self._sorted_labels = sorted(strata.keys())

        total_population = sum(len(items) for items in strata.values())

        proposed_allocation = self._propose(self.current_allocation)

        e_current = self._energy(self.current_allocation, strata_stats, total_population)
        e_proposed = self._energy(proposed_allocation, strata_stats, total_population)

        if self._accept(e_current, e_proposed):
            self.current_allocation = proposed_allocation
            self.n_accepted += 1
            self.energy_trace.append(e_proposed)
        else:
            self.energy_trace.append(e_current)

        self.step_count += 1
        self.temperature *= self.anneal_rate

        rng_state_before = self.rng.getstate()

        realized_items: list[str] = []
        for h in self._sorted_labels:
            n_h = self.current_allocation.get(h, 0)
            if n_h > 0 and h in strata:
                k = min(n_h, len(strata[h]))
                drawn = self.rng.sample(strata[h], k)
                realized_items.extend(item.item_id for item in drawn)

        allocation_list = [self.current_allocation.get(h, 0) for h in self._sorted_labels]

        return ItemRealization(
            allocation=allocation_list,
            realized_items=realized_items,
            rng_state_before=rng_state_before,
        )
