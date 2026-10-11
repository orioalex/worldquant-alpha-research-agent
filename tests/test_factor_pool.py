import json
import unittest
from collections import Counter

import alpha_research_pipeline as pipeline


class FactorPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        with open("alpha_pipeline_ideas.json", encoding="utf-8") as handle:
            self.library = json.load(handle)
        self.fields = pipeline.load_available_fields("wqb_data_fields_summary.json")

    def test_generates_bounded_two_and_three_factor_pools(self) -> None:
        candidates = pipeline.generate_seed_candidates(
            library=self.library,
            family_filter=set(),
            available_fields=self.fields,
            factor_pool_max_atoms=50,
            factor_pool_max_pairs=7,
            factor_pool_max_triples=5,
            random_seed=17,
        )
        pairs = [item for item in candidates if item.stage == "factor_pool_2"]
        triples = [item for item in candidates if item.stage == "factor_pool_3"]
        self.assertEqual(len(pairs), 7)
        self.assertEqual(len(triples), 5)
        self.assertTrue(all(item.metadata["factor_count"] == 2 for item in pairs))
        self.assertTrue(all(item.metadata["factor_count"] == 3 for item in triples))
        self.assertTrue(all("trade_when" not in item.expression for item in pairs + triples))

    def test_synthetic_family_filter_selects_only_requested_pool(self) -> None:
        candidates = pipeline.generate_seed_candidates(
            library=self.library,
            family_filter={"factor_pool_3"},
            available_fields=self.fields,
            factor_pool_max_atoms=50,
            factor_pool_max_pairs=4,
            factor_pool_max_triples=4,
            random_seed=3,
        )
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(item.family == "factor_pool_3" for item in candidates))

    def test_atom_reuse_and_social_buzz_share_are_bounded(self) -> None:
        candidates = pipeline.generate_seed_candidates(
            library=self.library,
            family_filter=set(),
            available_fields=self.fields,
            factor_pool_max_atoms=80,
            factor_pool_max_pairs=40,
            factor_pool_max_triples=40,
            factor_pool_max_atom_reuse=2,
            factor_pool_social_buzz_share=0.25,
            random_seed=29,
        )
        pool = [item for item in candidates if item.metadata.get("factor_pool")]
        usage = Counter(
            signature
            for item in pool
            for signature in item.metadata.get("source_atom_signatures", [])
        )
        self.assertTrue(usage)
        self.assertLessEqual(max(usage.values()), 2)
        social_candidates = [
            item
            for item in pool
            if "social_buzz" in item.metadata.get("source_families", [])
        ]
        self.assertLessEqual(len(social_candidates), len(pool) * 0.5)

    def test_pareto_objectives_penalize_failed_correlation(self) -> None:
        base = {
            "status": "ok",
            "metrics": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.2, "drawdown": 0.1},
            "check_raw": {"is": {"checks": [
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "LOW_FITNESS", "result": "PASS"},
                {"name": "PROD_CORRELATION", "result": "PASS"},
            ]}},
        }
        blocked = json.loads(json.dumps(base))
        blocked["check_raw"]["is"]["checks"][-1]["result"] = "FAIL"
        self.assertGreater(
            pipeline.pareto_metrics(base)["correlation"],
            pipeline.pareto_metrics(blocked)["correlation"],
        )
        self.assertGreater(pipeline.pareto_score(base), pipeline.pareto_score(blocked))

    def test_evolutionary_search_creates_mutations_and_crossovers(self) -> None:
        records = [
            {
                "status": "ok",
                "alpha_id": "A1",
                "candidate_key": "k1",
                "idea_name": "seed1",
                "family": "price_momentum",
                "expression": "rank((ts_mean(close, 5))) + rank((ts_mean(volume, 20)))",
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "metadata": {"factor_count": 2, "source_families": ["price_momentum", "volume_flow"]},
                "metrics": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.2, "drawdown": 0.1},
                "check_raw": {"is": {"checks": []}},
            },
            {
                "status": "ok",
                "alpha_id": "A2",
                "candidate_key": "k2",
                "idea_name": "seed2",
                "family": "quality",
                "expression": "rank((ts_mean(returns, 13)))",
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "metadata": {"factor_count": 1, "source_families": ["quality"]},
                "metrics": {"sharpe": 1.3, "fitness": 1.0, "turnover": 0.2, "drawdown": 0.1},
                "check_raw": {"is": {"checks": []}},
            },
        ]
        candidates = pipeline.build_evolutionary_candidates(
            records,
            family_filter=set(),
            max_candidates=12,
            random_seed=4,
        )
        self.assertTrue(candidates)
        self.assertTrue(any(item.stage == "evolve_window" for item in candidates))
        self.assertTrue(any(item.stage == "evolve_crossover" for item in candidates))
        self.assertTrue(all("trade_when" not in item.expression for item in candidates))


if __name__ == "__main__":
    unittest.main()
