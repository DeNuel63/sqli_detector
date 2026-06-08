import asyncio
import json
import unittest

from api.app import global_exception_handler


class ApiErrorHandlerTests(unittest.TestCase):
    def test_global_exception_handler_does_not_leak_exception_detail(self):
        response = asyncio.run(
            global_exception_handler(None, RuntimeError("database password leaked"))
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload, {"error": "Internal server error"})
        self.assertNotIn("database password leaked", response.body.decode())


if __name__ == "__main__":
    unittest.main()
