import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app


class TriageEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_stub_returns_valid_closed_output(self) -> None:
        with patch.dict(os.environ, {"LLM_STUB": "1"}):
            response = self.client.post(
                "/triage", json={"text": "I was charged twice."}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"category", "urgency", "confidence", "reason"},
        )
        self.assertIn(
            response.json()["category"], {"billing", "bug", "feature", "other"}
        )
        self.assertIn(response.json()["urgency"], {"low", "normal", "high"})
        self.assertGreaterEqual(response.json()["confidence"], 0.0)
        self.assertLessEqual(response.json()["confidence"], 1.0)

    def test_missing_text_returns_400_naming_text(self) -> None:
        response = self.client.post("/triage", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["field"], "text")
        self.assertIn("text", response.json()["message"])

    def test_wrong_text_type_returns_400_naming_text(self) -> None:
        response = self.client.post("/triage", json={"text": 42})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["field"], "text")
        self.assertIn("text", response.json()["message"])

    def test_malformed_json_returns_400_naming_text(self) -> None:
        response = self.client.post(
            "/triage",
            content='{"text":',
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["field"], "text")
        self.assertIn("text", response.json()["message"])

    def test_text_over_limit_returns_400_naming_text(self) -> None:
        response = self.client.post("/triage", json={"text": "x" * 2001})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["field"], "text")
        self.assertIn("text", response.json()["message"])

    def test_stub_disabled_returns_explicit_503(self) -> None:
        with patch.dict(os.environ, {"LLM_STUB": "0"}):
            response = self.client.post(
                "/triage", json={"text": "I was charged twice."}
            )

        self.assertEqual(response.status_code, 503)
