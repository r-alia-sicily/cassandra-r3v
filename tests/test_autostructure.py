from __future__ import annotations

import unittest

from cassandra_r3v.autostructure import compute_recommendations, estimate_pairwise_candidates
from cassandra_r3v.config import AutoStructureConfig
from cassandra_r3v.rules import Antecedent, SemanticRule, compose_rules, pairwise_candidates, unary_backbone


class AutoStructureTests(unittest.TestCase):
    def test_full24_pair_count(self):
        self.assertEqual(estimate_pairwise_candidates(120, 24), 6900)

    def test_pairwise_generator_excludes_same_variable(self):
        candidates = pairwise_candidates(unary_backbone(("a", "b")))
        self.assertEqual(len(candidates), 25)
        self.assertTrue(all(len(set(rule.features)) == 2 for rule in candidates))

    def test_higher_order_composition(self):
        pair = SemanticRule((Antecedent("a", "LOW"), Antecedent("b", "HIGH")))
        unary = SemanticRule((Antecedent("c", "MEDIUM"),))
        result = compose_rules(pair, unary)
        self.assertEqual(result.order, 3)

    def test_repeated_variable_composition_is_rejected(self):
        first = SemanticRule((Antecedent("a", "LOW"),))
        second = SemanticRule((Antecedent("a", "HIGH"),))
        self.assertIsNone(compose_rules(first, second))

    def test_budget_recommendations_are_actionable(self):
        suggestions = compute_recommendations(AutoStructureConfig())
        self.assertGreaterEqual(len(suggestions), 4)
        self.assertTrue(any("support" in item for item in suggestions))


if __name__ == "__main__":
    unittest.main()
