"""Protocol specification tests: constants and validators.

These tests pin the machine-readable spec so that any implementation
(reference service, independent client, or third-party port) can be
validated against the same contract.
"""
import math
import unittest
from jev_laya_free import protocol


class ConstantTests(unittest.TestCase):
    def test_endpoint_and_method(self):
        self.assertEqual(protocol.ENDPOINT, "/v1/systemone")
        self.assertEqual(protocol.HTTP_METHOD, "POST")

    def test_size_limits(self):
        self.assertEqual(protocol.MAX_REQUEST_BYTES, 65536)
        self.assertEqual(protocol.MAX_RESPONSE_BYTES, 1_048_576)
        self.assertEqual(protocol.MAX_NESTING_DEPTH, 16)
        self.assertEqual(protocol.MAX_QUESTION_COUNT, 32)
        self.assertEqual(protocol.MAX_CHOICE_OPTIONS, 255)
        self.assertEqual(protocol.MAX_SCORE_LEVELS, 10)
        self.assertEqual(protocol.MIN_SCORE_LEVELS, 2)
        self.assertEqual(protocol.MAX_DESCRIPTION_BYTES, 4096)
        self.assertEqual(protocol.MAX_STATE_BYTES, 32768)
        self.assertEqual(protocol.MAX_MODEL_NAME_LENGTH, 128)

    def test_question_types(self):
        self.assertEqual(protocol.QUESTION_TYPES, ("choice", "score", "noul"))

    def test_capability_families_and_legacy(self):
        self.assertEqual(len(protocol.CAPABILITY_FAMILIES), 12)
        self.assertEqual(protocol.LEGACY_QUESTIONS, ["next_hand", "needs_review"])
        self.assertEqual(len(protocol.all_question_names()), 14)

    def test_status_codes(self):
        self.assertEqual(protocol.STATUS_OK, 200)
        self.assertEqual(protocol.STATUS_BAD_REQUEST, 400)
        self.assertEqual(protocol.STATUS_UNAUTHORIZED, 401)
        self.assertEqual(protocol.STATUS_NOT_FOUND, 404)
        self.assertEqual(protocol.STATUS_UNSUPPORTED_MEDIA, 415)
        self.assertEqual(protocol.STATUS_UNPROCESSABLE, 422)
        self.assertEqual(protocol.STATUS_REQUEST_TOO_LARGE, 413)
        self.assertEqual(protocol.STATUS_REQUEST_TIMEOUT, 408)
        self.assertEqual(protocol.STATUS_BACKEND_BUSY, 529)
        self.assertEqual(protocol.STATUS_BAD_GATEWAY, 502)
        self.assertEqual(protocol.STATUS_SERVICE_UNAVAILABLE, 503)

    def test_retryable_status_codes(self):
        self.assertEqual(protocol.RETRYABLE_STATUS_CODES, frozenset({429, 502, 503, 504, 529}))

    def test_guard_decisions(self):
        self.assertEqual(protocol.GUARD_DECISIONS, ("allow", "stop", "review", "terminal"))


class JsonValueTests(unittest.TestCase):
    def test_accepts_scalars(self):
        for v in (None, "s", 1, 1.5, True, False):
            protocol.validate_json_value(v)

    def test_rejects_nonfinite_float(self):
        with self.assertRaises(ValueError):
            protocol.validate_json_value(float("inf"))

    def test_rejects_non_json_type(self):
        with self.assertRaises(ValueError):
            protocol.validate_json_value(object())

    def test_rejects_non_string_key(self):
        with self.assertRaises(ValueError):
            protocol.validate_json_value({1: "a"})

    def test_rejects_too_deep_nesting(self):
        value = "x"
        for _ in range(protocol.MAX_NESTING_DEPTH + 2):
            value = [value]
        with self.assertRaises(ValueError):
            protocol.validate_json_value(value)

    def test_accepts_at_max_depth(self):
        value = "x"
        for _ in range(protocol.MAX_NESTING_DEPTH):
            value = [value]
        protocol.validate_json_value(value)


class ProbabilityTests(unittest.TestCase):
    def test_accepts_valid(self):
        for v in (0, 1, 0.5, 1.0):
            protocol.validate_probability(v)

    def test_rejects_out_of_range(self):
        for v in (-0.1, 1.1, float("nan"), "0.5"):
            with self.assertRaises(ValueError):
                protocol.validate_probability(v)


class QuestionTests(unittest.TestCase):
    def test_valid_choice(self):
        protocol.validate_question({"type": "choice", "instructions": "Q?",
                                   "options": ["a", "b"]})

    def test_valid_choice_with_criteria_dict(self):
        protocol.validate_question({"type": "choice", "instructions": "Q?",
                                   "criteria": {"a": "opt a", "b": "opt b"}})

    def test_valid_score(self):
        protocol.validate_question({"type": "score", "instructions": "Q?",
                                   "criteria": ["low", "high"]})

    def test_valid_noul(self):
        protocol.validate_question({"type": "noul", "instructions": "Q?"})

    def test_valid_noul_with_criteria(self):
        protocol.validate_question({"type": "noul", "instructions": "Q?",
                                   "criteria": {"true": "yes", "false": "no"}})

    def test_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "boolean", "instructions": "Q?"})

    def test_rejects_missing_instructions(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "options": ["a"]})

    def test_rejects_choice_with_both_options_and_criteria(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "instructions": "Q?",
                                       "options": ["a"], "criteria": {"a": "x"}})

    def test_rejects_choice_with_neither_options_nor_criteria(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "instructions": "Q?"})

    def test_rejects_score_with_options(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "score", "instructions": "Q?",
                                       "options": ["a"], "criteria": ["low", "high"]})

    def test_rejects_score_too_few_levels(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "score", "instructions": "Q?",
                                       "criteria": ["only"]})

    def test_rejects_score_too_many_levels(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "score", "instructions": "Q?",
                                       "criteria": [str(i) for i in range(11)]})

    def test_rejects_noul_with_options(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "noul", "instructions": "Q?",
                                       "options": ["a"]})

    def test_rejects_unknown_question_field(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "noul", "instructions": "Q?",
                                       "bogus": 1})

    def test_rejects_duplicate_choice_options(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "instructions": "Q?",
                                       "options": ["a", "a"]})

    def test_rejects_empty_choice_options(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "instructions": "Q?",
                                       "options": []})

    def test_rejects_too_many_choice_options(self):
        with self.assertRaises(ValueError):
            protocol.validate_question({"type": "choice", "instructions": "Q?",
                                       "options": [f"o{i}" for i in range(256)]})


class RequestTests(unittest.TestCase):
    def _valid_body(self):
        return {"model": "local-rules-v1", "state": {"modality": "code"},
                "questions": {"q": {"type": "noul", "instructions": "Q?"}}}

    def test_accepts_valid(self):
        protocol.validate_request(self._valid_body())

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            protocol.validate_request([1, 2, 3])

    def test_rejects_unknown_field(self):
        body = self._valid_body()
        body["bogus"] = 1
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_missing_model(self):
        body = self._valid_body()
        del body["model"]
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_missing_state(self):
        body = self._valid_body()
        del body["state"]
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_missing_questions(self):
        body = self._valid_body()
        del body["questions"]
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_invalid_model(self):
        body = self._valid_body()
        body["model"] = ""
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_model_too_long(self):
        body = self._valid_body()
        body["model"] = "m" * 129
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_invalid_state_type(self):
        body = self._valid_body()
        body["state"] = 42
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_no_questions(self):
        body = self._valid_body()
        body["questions"] = {}
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_too_many_questions(self):
        body = self._valid_body()
        body["questions"] = {f"q{i}": {"type": "noul", "instructions": "Q?"}
                            for i in range(33)}
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_too_large_request(self):
        body = self._valid_body()
        body["state"] = "x" * 70000
        with self.assertRaises(ValueError):
            protocol.validate_request(body)

    def test_rejects_invalid_question_name(self):
        body = self._valid_body()
        body["questions"] = {"": {"type": "noul", "instructions": "Q?"}}
        with self.assertRaises(ValueError):
            protocol.validate_request(body)


class AnswerTests(unittest.TestCase):
    def test_valid_noul(self):
        protocol.validate_answer({"type": "noul", "noul": 0.7},
                                {"type": "noul", "instructions": "Q?"})

    def test_rejects_noul_type_mismatch(self):
        with self.assertRaises(ValueError):
            protocol.validate_answer({"type": "choice"},
                                    {"type": "noul", "instructions": "Q?"})

    def test_rejects_noul_invalid_probability(self):
        with self.assertRaises(ValueError):
            protocol.validate_answer({"type": "noul", "noul": 1.5},
                                    {"type": "noul", "instructions": "Q?"})

    def test_valid_choice(self):
        q = {"type": "choice", "instructions": "Q?", "criteria": {"a": "A", "b": "B"}}
        a = {"type": "choice", "confidence": 0.9,
             "probabilities": {"a": 0.8, "b": 0.2}, "choice": "a"}
        protocol.validate_answer(a, q)

    def test_rejects_choice_keys_mismatch(self):
        q = {"type": "choice", "instructions": "Q?", "criteria": {"a": "A", "b": "B"}}
        a = {"type": "choice", "confidence": 0.9,
             "probabilities": {"a": 0.8, "c": 0.2}, "choice": "a"}
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)

    def test_rejects_choice_probabilities_not_summing_to_one(self):
        q = {"type": "choice", "instructions": "Q?", "criteria": {"a": "A", "b": "B"}}
        a = {"type": "choice", "confidence": 0.9,
             "probabilities": {"a": 0.5, "b": 0.2}, "choice": "a"}
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)

    def test_rejects_choice_not_maximizing(self):
        q = {"type": "choice", "instructions": "Q?", "criteria": {"a": "A", "b": "B"}}
        a = {"type": "choice", "confidence": 0.9,
             "probabilities": {"a": 0.3, "b": 0.7}, "choice": "a"}
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)

    def test_valid_score(self):
        q = {"type": "score", "instructions": "Q?", "criteria": ["low", "med", "high"]}
        p = {"0": 0.2, "1": 0.5, "2": 0.3}
        score = 0 * 0.2 + 1 * 0.5 + 2 * 0.3
        a = {"type": "score", "confidence": 0.8, "probabilities": p, "score": score,
             "legend": {"0": "low", "1": "med", "2": "high"}}
        protocol.validate_answer(a, q)

    def test_rejects_score_out_of_range(self):
        q = {"type": "score", "instructions": "Q?", "criteria": ["low", "high"]}
        a = {"type": "score", "confidence": 0.8,
             "probabilities": {"0": 0.5, "1": 0.5}, "score": 5,
             "legend": {"0": "low", "1": "high"}}
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)

    def test_rejects_score_weighted_mean_mismatch(self):
        q = {"type": "score", "instructions": "Q?", "criteria": ["low", "high"]}
        a = {"type": "score", "confidence": 0.8,
             "probabilities": {"0": 0.9, "1": 0.1}, "score": 0.9,
             "legend": {"0": "low", "1": "high"}}
        # weighted mean = 0*0.9 + 1*0.1 = 0.1, but score is 0.9
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)

    def test_rejects_missing_confidence(self):
        q = {"type": "choice", "instructions": "Q?", "criteria": {"a": "A"}}
        a = {"type": "choice",
             "probabilities": {"a": 1.0}, "choice": "a"}
        with self.assertRaises(ValueError):
            protocol.validate_answer(a, q)


class ResponseTests(unittest.TestCase):
    def _valid(self):
        return {"model": "local-rules-v1",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "answers": {"q": {"type": "noul", "noul": 0.5}}}

    def test_accepts_valid(self):
        protocol.validate_response(self._valid(),
                                  {"q": {"type": "noul", "instructions": "Q?"}})

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            protocol.validate_response("not a dict", {})

    def test_rejects_missing_model(self):
        body = self._valid()
        del body["model"]
        with self.assertRaises(ValueError):
            protocol.validate_response(body, {})

    def test_rejects_missing_usage(self):
        body = self._valid()
        del body["usage"]
        with self.assertRaises(ValueError):
            protocol.validate_response(body, {})

    def test_rejects_invalid_usage(self):
        body = self._valid()
        body["usage"] = {"input_tokens": -1, "output_tokens": 0}
        with self.assertRaises(ValueError):
            protocol.validate_response(body, {})

    def test_rejects_answer_names_mismatch(self):
        body = self._valid()
        body["answers"] = {"other": {"type": "noul", "noul": 0.5}}
        with self.assertRaises(ValueError):
            protocol.validate_response(body, {"q": {"type": "noul", "instructions": "Q?"}})

    def test_rejects_invalid_request_id(self):
        body = self._valid()
        body["request_id"] = ""
        with self.assertRaises(ValueError):
            protocol.validate_response(body, {"q": {"type": "noul", "instructions": "Q?"}})


class TransportTests(unittest.TestCase):
    def test_valid_transport(self):
        protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                   "123", False)

    def test_rejects_wrong_method(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("GET", "/v1/systemone", "application/json",
                                       "123", False)

    def test_rejects_wrong_path(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/other", "application/json",
                                       "123", False)

    def test_rejects_wrong_content_type(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "text/plain",
                                       "123", False)

    def test_rejects_chunked(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                       "123", True)

    def test_rejects_missing_content_length(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                       None, False)

    def test_rejects_non_numeric_content_length(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                       "abc", False)

    def test_rejects_too_large_content_length(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                       "70000", False)

    def test_rejects_zero_content_length(self):
        with self.assertRaises(ValueError):
            protocol.validate_transport("POST", "/v1/systemone", "application/json",
                                       "0", False)


class BaseUrlTests(unittest.TestCase):
    def test_accepts_loopback(self):
        protocol.validate_base_url("http://127.0.0.1:8093")
        protocol.validate_base_url("http://localhost:8093")

    def test_rejects_non_loopback(self):
        with self.assertRaises(ValueError):
            protocol.validate_base_url("http://example.com:8093")

    def test_rejects_non_http(self):
        with self.assertRaises(ValueError):
            protocol.validate_base_url("https://127.0.0.1:8093")

    def test_rejects_userinfo(self):
        with self.assertRaises(ValueError):
            protocol.validate_base_url("http://user:pass@127.0.0.1:8093")

    def test_rejects_query(self):
        with self.assertRaises(ValueError):
            protocol.validate_base_url("http://127.0.0.1:8093?x=1")


class StateFieldTests(unittest.TestCase):
    def test_accepts_valid(self):
        protocol.validate_state_fields({"terminal": True, "same_action_streak": 2,
                                       "modality": "code", "action": "x"})

    def test_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            protocol.validate_state_fields({"bogus": 1})

    def test_rejects_non_bool_flag(self):
        with self.assertRaises(ValueError):
            protocol.validate_state_fields({"terminal": 1})

    def test_rejects_out_of_range_counter(self):
        with self.assertRaises(ValueError):
            protocol.validate_state_fields({"same_action_streak": 2_000_000})

    def test_rejects_too_long_text(self):
        with self.assertRaises(ValueError):
            protocol.validate_state_fields({"modality": "x" * 513})


if __name__ == "__main__":
    unittest.main()
