from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEMO_EVENTS = [
    {
        "event_id": "demo-lucas-20260714-1400",
        "camera_id": "cam-ti-01",
        "user_id": "EMP001",
        "room_id": "sala_ti_01",
        "timestamp": "2026-07-14T14:00:00-03:00",
        "door_result": "GRANTED",
        "recognition_confidence": 0.98,
    },
    {
        "event_id": "demo-mariana-20260714-0200",
        "camera_id": "cam-ti-01",
        "user_id": "EMP002",
        "room_id": "sala_ti_01",
        "timestamp": "2026-07-14T02:00:00-03:00",
        "door_result": "GRANTED",
        "recognition_confidence": 0.97,
    },
    {
        "event_id": "demo-roberto-20260719-2300",
        "camera_id": "cam-ti-01",
        "user_id": "EMP003",
        "room_id": "sala_ti_01",
        "timestamp": "2026-07-19T23:00:00-03:00",
        "door_result": "GRANTED",
        "recognition_confidence": 0.96,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Envia os três cenários do PDF ao RAG-Audit.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--camera-key", default="dev-camera-key")
    args = parser.parse_args()

    for event in DEMO_EVENTS:
        request = Request(
            f"{args.base_url.rstrip('/')}/v1/webhooks/access-events",
            data=json.dumps(event).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Camera-Key": args.camera_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                body = json.load(response)
                saved = body["event"]
                print(
                    f"{response.status} {event['event_id']}: "
                    f"{saved['decision']} / {saved['risk_level']}"
                )
        except HTTPError as exc:
            print(f"{exc.code} {event['event_id']}: {exc.read().decode('utf-8')}")


if __name__ == "__main__":
    main()
