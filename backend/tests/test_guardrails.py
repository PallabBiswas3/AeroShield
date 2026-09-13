import unittest
from unittest.mock import patch

from app.agents.orchestrator import _get_llm, generate_enforcement_mandate, legal_drafter_node


class _UnsafeResponse:
    content = '{"statute_violated":"Section 21", "legal_notice_draft":"Cease and desist: facility has violated the law", "dispatch_priority":"URGENT", "case_summary":"Violation at facility"}'


class _UnsafeLLM:
    def invoke(self, _messages):
        return _UnsafeResponse()


class GuardrailTests(unittest.TestCase):
    def test_llm_is_opt_in_for_reliable_deployment(self):
        with patch.dict("os.environ", {"AEROSHIELD_ENABLE_LLM": "false"}), \
             patch("app.agents.orchestrator.GROQ_API_KEY", "configured-key"):
            self.assertIsNone(_get_llm())

    def test_deterministic_fallback_never_declares_violation(self):
        with patch("app.agents.orchestrator._get_llm", return_value=None):
            result = generate_enforcement_mandate(1, 150.0, "Example source", [])
        combined = " ".join(str(value) for value in result.values()).lower()
        self.assertNotIn("cease and desist", combined)
        self.assertIn("not a finding", combined)

    def test_unsafe_llm_output_is_replaced(self):
        state = {"primary_violator": "Example", "escalation_level": "HIGH", "enforcement_brief": "Verify", "applicable_statutes": [], "aqi_value": 120.0}
        with patch("app.agents.orchestrator._get_llm", return_value=_UnsafeLLM()):
            result = legal_drafter_node(state)
        self.assertNotIn("cease and desist", result["legal_notice_draft"].lower())
        self.assertEqual(result["dispatch_priority"], "HIGH")


if __name__ == "__main__":
    unittest.main()
