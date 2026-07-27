from server import calculate_speech_recognition_confidence


def test_asr_confidence_is_plain_average_without_pronunciation_penalty():
    result = calculate_speech_recognition_confidence([
        {"word": "hello", "probability": 0.9},
        {"word": "world", "probability": 0.5},
    ])

    assert result["overall_score"] == 70
    assert result["method"] == "asr_word_probability"
    assert "not a pronunciation-accuracy score" in result["details"]


def test_missing_word_probabilities_is_reported_as_unavailable():
    result = calculate_speech_recognition_confidence([])
    assert result["overall_score"] is None
    assert result["method"] == "unavailable"
