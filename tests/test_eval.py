import unittest

from evals.run_eval import load_cases
from src.llm.schema import Category


class EvaluationCasesTests(unittest.TestCase):
    def test_evaluation_contains_eight_valid_category_cases(self) -> None:
        cases = load_cases()

        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case["id"] for case in cases}), 8)
        for case in cases:
            self.assertIn(case["expected_category"], {category.value for category in Category})

    def test_ambiguous_cases_use_other(self) -> None:
        cases = {case["id"]: case for case in load_cases()}

        self.assertEqual(cases["vague-account-problem"]["expected_category"], "other")
        self.assertEqual(cases["unclear-message"]["expected_category"], "other")
