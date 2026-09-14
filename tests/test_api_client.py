import json
import unittest

import httpx
import numpy as np

from reconhecimento.api.client import (
    AccessControlClient,
    ApiAuthenticationError,
    ApiResponseError,
    ApiUnavailableError,
    EnrollmentClient,
)


def unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


def response_body(**overrides: object) -> dict:
    data = {
        "recognized": True,
        "allowed": True,
        "participantType": "guest",
        "participantId": "1",
        "guestId": "1",
        "userId": None,
        "name": "Guest Teste",
        "similarity": 0.82,
        "reason": "AUTHORIZED",
        "direction": "ENTRY",
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
            participantType=None,
            participantId=None,
            guestId=None,
            userId=None,
            name=None,
            similarity=None,
            reason="UNKNOWN_FACE",
            direction=None,
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

    def test_client_participant(self) -> None:
        body = response_body(
            participantType="client",
            participantId="7",
            guestId=None,
            userId="7",
            name="Cliente Teste",
            direction="EXIT",
        )
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        result = client.recognize(unit_embedding())

        self.assertEqual(result.participant_type, "client")
        self.assertEqual(result.user_id, "7")
        self.assertEqual(result.direction, "EXIT")

    def test_known_denial_accepts_backend_reason_without_name(self) -> None:
        body = response_body(allowed=False, reason="DUPLICATE_ENTRY", name=None)
        client = self.make_client(lambda _request: httpx.Response(200, json=body))

        result = client.recognize(unit_embedding())

        self.assertTrue(result.recognized)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "DUPLICATE_ENTRY")

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


class EnrollmentClientTest(unittest.TestCase):
    def test_publishes_guest_embedding_with_backend_contract(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "PUT")
            self.assertEqual(request.url.path, "/vip/api/v1/access-control/guests/42/embedding")
            self.assertEqual(request.headers["X-Device-Key"], "enrollment-secret")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "buffalo_l")
            self.assertEqual(payload["dimension"], 512)
            self.assertEqual(payload["normalization"], "l2")
            self.assertEqual(len(payload["embedding"]), 512)
            return httpx.Response(200, json={"success": True, "data": {}})

        client = EnrollmentClient(
            "https://ispevolution.com.br/vip/api/v1",
            "enrollment-secret",
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(client.close)

        client.enroll_guest("42", unit_embedding(), "a" * 64)

    def test_rejects_invalid_checksum_before_network(self) -> None:
        client = EnrollmentClient(
            "https://ispevolution.com.br/vip/api/v1",
            "enrollment-secret",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        )
        self.addCleanup(client.close)

        with self.assertRaises(ValueError):
            client.enroll_guest("42", unit_embedding(), "invalid")


if __name__ == "__main__":
    unittest.main()
