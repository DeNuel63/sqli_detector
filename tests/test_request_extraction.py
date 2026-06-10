import unittest

from api.middleware import extract_params
from api.schema import ScanRequestParams


class RequestExtractionTests(unittest.TestCase):
    def test_scan_schema_accepts_json_array_body(self):
        request = ScanRequestParams(
            json_body=[
                {"search": "normal"},
                {"password": "' OR 1=1--"},
            ]
        )

        self.assertEqual(request.json_body[1]["password"], "' OR 1=1--")

    def test_extract_params_preserves_repeated_form_values(self):
        params = extract_params(form_data={"tag": ["safe", "' OR 1=1--"]})

        self.assertEqual(params["form:tag[0]"], "safe")
        self.assertEqual(params["form:tag[1]"], "' OR 1=1--")

    def test_extract_params_flattens_json_array_root(self):
        params = extract_params(json_body=[{"search": "safe"}, {"id": "' OR 1=1--"}])

        self.assertEqual(params["json:[0].search"], "safe")
        self.assertEqual(params["json:[1].id"], "' OR 1=1--")


if __name__ == "__main__":
    unittest.main()
