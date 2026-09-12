import unittest

from app.ml.intervention import rank_interventions


class InterventionTests(unittest.TestCase):
    def test_ranking_uses_conservative_probability(self):
        sources = [
            {"source_id": 1, "name": "Factory", "type": "Industrial Stack", "attribution_probability": 60, "probability_p10": 35},
            {"source_id": 2, "name": "Site", "type": "Construction", "attribution_probability": 40, "probability_p10": 5},
        ]
        result = rank_interventions(150, sources, exposed_population=200_000, sensitive_sites=4)
        self.assertEqual(result["ranked_actions"][0]["source_id"], 1)
        self.assertTrue(all(action["requires_field_verification"] for action in result["ranked_actions"]))
        self.assertLessEqual(result["recommended_portfolio"]["post_action_pm25"], 150)

    def test_no_excess_means_no_predicted_reduction(self):
        sources = [{"source_id": 1, "name": "Factory", "type": "Industrial Stack", "attribution_probability": 100, "probability_p10": 100}]
        result = rank_interventions(20, sources)
        self.assertEqual(result["ranked_actions"][0]["robust_pm25_reduction"], 0)


if __name__ == "__main__":
    unittest.main()
