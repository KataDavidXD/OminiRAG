from __future__ import annotations

from bsamp.sampling.budget import BudgetController


class TestBudgetController:
    def test_initial_state(self):
        bc = BudgetController(total=100)
        assert bc.remaining == 100
        assert bc.fraction_used == 0.0
        assert not bc.exhausted

    def test_consume(self):
        bc = BudgetController(total=50)
        bc.consume(20)
        assert bc.used == 20
        assert bc.remaining == 30

    def test_exhaustion(self):
        bc = BudgetController(total=10)
        bc.consume(10)
        assert bc.exhausted
        assert bc.remaining == 0

    def test_overspend_clamps_remaining(self):
        bc = BudgetController(total=5)
        bc.consume(8)
        assert bc.remaining == 0
        assert bc.exhausted

    def test_can_afford(self):
        bc = BudgetController(total=20, used=15)
        assert bc.can_afford(5)
        assert not bc.can_afford(6)

    def test_clamp(self):
        bc = BudgetController(total=30, used=25)
        assert bc.clamp(10) == 5
        assert bc.clamp(3) == 3
