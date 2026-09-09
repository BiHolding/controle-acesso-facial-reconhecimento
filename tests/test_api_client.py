import json
import unittest

import httpx
import numpy as np

from reconhecimento.api.client import (
    AccessControlClient,
    ApiAuthenticationError,
    ApiResponseError,
    ApiUnavailableError,
)


def unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


def response_body(**overrides: object) -> dict:
    data = {
        "recognized": True,
        "allowed": True,
        "guestId": "1",
        "name": "Guest Teste",
        "similarity": 0.82,
        "reason": "AUTHORIZED",
    }
    data.update(overrides)
    return {"success": True, "data": data}


class AccessControlClientTest(unittest.TestCase):
    def make_client(self, handler) -> AccessControlClient:
        client = AccessControlClient(
            "https://ispevolution.com.br/api/v1",
            "device-secret",
            "vip_room",
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(client.close)
        return client

    def test_recognized_allowed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/v1/access-control/recognize")
            self.assertEqual(request.headers["X-Device-Key"], "device-secret")
            payload = json.loads(request.content)
            self.assertEqual(payload["accessPoint"], "vip_room")
            self.assertEqual(len(payload["embedding"]), 512)
            return httpx.Response(200, json=response_body())

        result = self.make_client(handler).recognize(unit_embedding())

        self.assertTrue(result.recognized)
        self.assertTrue(result.allowed)
        self.assertEqual(result.guest_id, "1")
        self.assertEqual(result.reason, "AUTHORIZED")

    def test_recognized_denied(self) -> None:
        body = response_body(allowed=False, reason="NOT_AUTHORIZED")
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        result = client.recognize(unit_embedding())

        self.assertTrue(result.recognized)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "NOT_AUTHORIZED")

    def test_unknown(self) -> None:
        body = response_body(
            recognized=False,
            allowed=False,
            guestId=None,
            name=None,
            similarity=None,
            reason="UNKNOWN_FACE",
        )
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        result = client.recognize(unit_embedding())

        self.assertFalse(result.recognized)
        self.assertFalse(result.allowed)
        self.assertIsNone(result.guest_id)

    def test_timeout_fails_closed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout", request=request)

        with self.assertRaises(ApiUnavailableError):
            self.make_client(handler).recognize(unit_embedding())

    def test_unauthorized_statuses_fail_closed(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status), self.assertRaises(ApiAuthenticationError):
                client = self.make_client(lambda _request, status=status: httpx.Response(status))
                client.recognize(unit_embedding())

    def test_server_error_fails_closed(self) -> None:
        client = self.make_client(lambda _request: httpx.Response(500))

        with self.assertRaises(ApiUnavailableError):
            client.recognize(unit_embedding())

    def test_invalid_json_fails_closed(self) -> None:
        client = self.make_client(
            lambda _request: httpx.Response(
                200,
                content=b"{invalid",
                headers={"Content-Type": "application/json"},
            )
        )

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())

    def test_inconsistent_types_fail_closed(self) -> None:
        body = response_body(recognized="true")
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())

    def test_inconsistent_authorization_reason_fails_closed(self) -> None:
        body = response_body(reason="NOT_AUTHORIZED")
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())

    def test_unexpected_personal_data_fails_closed(self) -> None:
        body = response_body()
        body["data"]["email"] = "not-needed@example.com"
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())

    def test_unexpected_envelope_data_fails_closed(self) -> None:
        body = response_body()
        body["email"] = "not-needed@example.com"
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())

    def test_redirect_fails_closed(self) -> None:
        client = self.make_client(lambda _request: httpx.Response(302, json=response_body()))

        with self.assertRaises(ApiResponseError):
            client.recognize(unit_embedding())


if __name__ == "__main__":
    unittest.main()
