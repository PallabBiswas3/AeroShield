import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class DeploymentTests(unittest.TestCase):
    def test_health_reports_configuration_without_secret_values(self):
        with patch.dict(os.environ, {"NASA_FIRMS_MAP_KEY": "never-return-this", "GROQ_API_KEY": "also-secret"}):
            response = TestClient(app).get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["integrations"]["nasa_firms_configured"])
        self.assertNotIn("never-return-this", response.text)
        self.assertNotIn("also-secret", response.text)


if __name__ == "__main__":
    unittest.main()
