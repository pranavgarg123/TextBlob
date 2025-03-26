import unittest
import pytest

from textblob.sentiments import (
    CONTINUOUS,
    DISCRETE,
    NaiveBayesAnalyzer,
    PatternAnalyzer,
    TransformerAnalyzer,
)


class TestPatternSentiment(unittest.TestCase):
    def setUp(self):
        self.analyzer = PatternAnalyzer()

    def test_kind(self):
        assert self.analyzer.kind == CONTINUOUS

    def test_analyze(self):
        p1 = "I feel great this morning."
        n1 = "This is a terrible car."
        p1_result = self.analyzer.analyze(p1)
        n1_result = self.analyzer.analyze(n1)
        assert p1_result[0] > 0
        assert n1_result[0] < 0
        assert p1_result.polarity == p1_result[0]
        assert p1_result.subjectivity == p1_result[1]

    def test_analyze_assessments(self):
        p1 = "I feel great this morning."
        n1 = "This is a terrible car."
        p1_result = self.analyzer.analyze(p1, keep_assessments=True)
        n1_result = self.analyzer.analyze(n1, keep_assessments=True)
        assert p1_result.polarity > 0
        assert n1_result.polarity < 0
        assert n1_result.subjectivity >= 0


class TestNaiveBayesAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = NaiveBayesAnalyzer()

    def test_kind(self):
        assert self.analyzer.kind == DISCRETE

    @pytest.mark.slow
    def test_analyze(self):
        p1 = "I feel great this morning."
        n1 = "This is a terrible car."
        p1_result = self.analyzer.analyze(p1)
        assert p1_result[0] == "pos"
        assert self.analyzer.analyze(n1)[0] == "neg"
        # The 2nd item should be the probability that it is positive
        assert isinstance(p1_result[1], float)
        # 3rd item is probability that it is negative
        assert isinstance(p1_result[2], float)
        assert_about_equal(p1_result[1] + p1_result[2], 1)
        assert p1_result.classification == p1_result[0]
        assert p1_result.p_pos == p1_result[1]
        assert p1_result.p_neg == p1_result[2]


class TestTransformerAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TransformerAnalyzer()

    def test_empty_input(self):
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze("")
        self.assertEqual(str(context.exception), "Input text cannot be empty or whitespace.")
    
    def test_invalid_model(self):
        with self.assertRaises(RuntimeError) as context:
            self.analyzer = TransformerAnalyzer(model_name="invalid-model")
        self.assertIn("Failed to initialize the pipeline with model 'invalid-model'", str(context.exception))

    def test_valid_output(self):
        """Test that the analyzer returns valid output for a positive sentence."""
        result = self.analyzer.analyze("I love this!")
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE"]

    def test_negative_sentiment(self):
        """Test that the analyzer correctly identifies negative sentiment."""
        result = self.analyzer.analyze("I hate this!")
        self.assertIsNotNone(result, "Analyzer returned None for input text.")
        assert result is not None and "label" in result, "Result does not contain 'label' key."
        assert result["label"] == "NEGATIVE"

    def test_positive_sentiment(self):
        """Test that the analyzer correctly identifies positive sentiment."""
        result = self.analyzer.analyze("This is amazing!")
        self.assertIsNotNone(result, "Analyzer returned None for input text.")
        assert result is not None and "label" in result, "Result does not contain 'label' key."
        assert result["label"] == "POSITIVE"


def assert_about_equal(first, second, places=4):
    assert round(first, places) == second


if __name__ == "__main__":
    unittest.main()
