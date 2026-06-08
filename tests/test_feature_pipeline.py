import unittest

from src.features import apply_tfidf, combine_features, fit_tfidf, queries_to_handcrafted_matrix
from src.predict import build_feature_matrix, clean_query


class FeaturePipelineTests(unittest.TestCase):
    def test_predict_feature_matrix_matches_shared_feature_builder(self):
        training_queries = [
            "select select user from users",
            "normal normal product lookup",
            "admin admin login form",
            "union union select password",
        ]
        raw_queries = [
            "%27 OR 1=1--",
            " normal\tproduct   lookup ",
            "UNION SELECT password FROM users",
        ]
        vectorizer = fit_tfidf(training_queries)

        cleaned = [clean_query(query) for query in raw_queries]
        expected = combine_features(
            apply_tfidf(vectorizer, cleaned),
            queries_to_handcrafted_matrix(cleaned),
        )
        actual = build_feature_matrix(raw_queries, vectorizer)

        self.assertEqual(actual.shape, expected.shape)
        difference = actual - expected
        self.assertEqual(difference.nnz, 0)


if __name__ == "__main__":
    unittest.main()
