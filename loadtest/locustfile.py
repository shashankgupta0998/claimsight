"""Load test for the ClaimSight API.

    make loadtest HOST=http://localhost:8000 USERS=20 DURATION=60s

Each simulated user files claims back to back: a golden claim under a fresh id (the replay
cassettes are keyed by content, so every claim still resolves offline). If the claim ends
up waiting for an adjuster, the same user posts the review, so every claim finishes its
whole life.

The API tells us how it runs claims (`GET /healthz`):

* inline: POST /claims returns the decision; that request is the end-to-end time.
* queued: POST /claims returns 202, then the user polls GET /claims/{id} every 100 ms.
  The custom `claim decided` row is submit-to-decision time, the number a client sees.

Run the API in replay mode (the default) so no model calls are made or billed: this
measures the service (HTTP, graph, checkpointer, queue), not Claude.
"""

from __future__ import annotations

import itertools
import json
import random
import time
import uuid
from pathlib import Path

from locust import HttpUser, between, constant, events, task

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden"
CLAIMS = [
    json.loads(line)["claim"]
    for name in ("claims.jsonl", "variants.jsonl", "synthetic.jsonl")
    for line in (GOLDEN / name).read_text().splitlines()
    if line.strip()
]
POLL_S = 0.1
POLL_TIMEOUT_S = 60
DONE = {"decided", "awaiting_review", "failed"}


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--think", type=float, default=0.0, help="seconds between claims")


class Claimant(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        think = self.environment.parsed_options.think
        if think:
            self.wait_time = between(think * 0.5, think * 1.5)
        health = self.client.get("/healthz", name="/healthz").json()
        self.queued = health.get("queue") == "redis"
        self.claims = itertools.cycle(random.sample(CLAIMS, len(CLAIMS)))

    def _fire(self, name: str, started: float, error: str | None = None) -> None:
        self.environment.events.request.fire(
            request_type="CLAIM",
            name=name,
            response_time=(time.perf_counter() - started) * 1000,
            response_length=0,
            exception=RuntimeError(error) if error else None,
            context={},
        )

    @task
    def file_claim(self):
        claim = {**next(self.claims), "claim_id": f"LT-{uuid.uuid4().hex[:12]}"}
        cid = claim["claim_id"]
        started = time.perf_counter()
        with self.client.post("/claims", json=claim, name="/claims", catch_response=True) as r:
            if r.status_code not in (200, 202):
                r.failure(f"submit {r.status_code}")
                return
            body = r.json()
        if self.queued:
            deadline = started + POLL_TIMEOUT_S
            while body["status"] not in DONE:
                if time.perf_counter() > deadline:
                    self._fire("claim decided", started, "timed out waiting for the worker")
                    return
                time.sleep(POLL_S)
                body = self.client.get(f"/claims/{cid}", name="/claims/[id]").json()
            if body["status"] == "failed":
                self._fire("claim decided", started, body.get("error") or "failed")
                return
            self._fire("claim decided", started)
        if body["status"] == "awaiting_review":
            decision = random.choice(["approve", "deny"])
            self.client.post(
                f"/claims/{cid}/review",
                json={"decision": decision, "reviewer": "load-test"},
                name="/claims/[id]/review",
            )
