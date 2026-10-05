import unittest

import alpha_research_pipeline as pipeline
from alpha_agent.config import ModelConfig
from alpha_agent.planner import OpenAIJsonPlanner


class ExpressionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = {"close", "vwap", "scl12_buzz", "roic"}

    def test_accepts_normalized_multifactor_expression(self) -> None:
        result = pipeline.validate_llm_expression(
            "0.4 * rank(ts_mean(roic, 8)) + 0.35 * rank(ts_mean(scl12_buzz, 5)) - 0.25 * rank(ts_delta(close, 10))",
            available_fields=self.fields,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["fields"], ["close", "roic", "scl12_buzz"])

    def test_rejects_unknown_field_and_operator(self) -> None:
        result = pipeline.validate_llm_expression(
            "magic_operator(close, invented_factor)",
            available_fields=self.fields,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("unknown operators" in error for error in result["errors"]))
        self.assertTrue(any("unknown fields" in error for error in result["errors"]))

    def test_rejects_too_many_factors(self) -> None:
        result = pipeline.validate_llm_expression(
            "rank(close) + rank(vwap) + rank(roic) + rank(scl12_buzz)",
            available_fields=self.fields,
            max_factors=3,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("maximum is 3" in error for error in result["errors"]))


class PlannerExpressionParsingTests(unittest.TestCase):
    def test_parses_expression_action(self) -> None:
        planner = OpenAIJsonPlanner(ModelConfig(provider="openai"))
        action = planner._parse_action(
            payload={
                "action": "propose_expression",
                "batch_size": 1,
                "focus_family": "multi_factor_combo",
                "expression": "rank(close) + rank(vwap)",
                "rationale": "Combine price and liquidity signals.",
            },
            remaining_budget=5,
        )
        self.assertEqual(action.action, "propose_expression")
        self.assertEqual(action.expression, "rank(close) + rank(vwap)")


if __name__ == "__main__":
    unittest.main()
