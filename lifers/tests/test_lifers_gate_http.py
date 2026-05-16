"""HTTP 黑盒：子进程启动 lifers_gate，探测 /health、POST /v1/step、POST /v1/stream。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _gate_url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


@unittest.skipIf(os.environ.get("LIFERS_SKIP_GATE_HTTP") == "1", "LIFERS_SKIP_GATE_HTTP=1")
class LifersGateHttpBlackboxTests(unittest.TestCase):
    """子进程跑 scripts/lifers_gate.py；默认 Markov + taskflow 关以缩短首包。"""

    _proc: subprocess.Popen | None = None
    _port: int = 0
    _root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._root = Path(__file__).resolve().parents[1]
        cls._proc = None
        base = 56100 + (os.getpid() % 300)
        last_err: Exception | None = None
        for off in range(30):
            cls._port = base + off
            env = {**os.environ, "PYTHONPATH": str(cls._root)}
            env.setdefault("MODEL", "markov")
            env.setdefault("LIFERS_TASKFLOW", "0")
            env.setdefault("SANDBOX", "1")
            try:
                cls._proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(cls._root / "scripts" / "lifers_gate.py"),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(cls._port),
                    ],
                    cwd=str(cls._root),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                last_err = e
                continue
            time.sleep(0.6)
            if cls._proc.poll() is not None:
                cls._proc = None
                continue
            ok = False
            for _ in range(40):
                try:
                    with urllib.request.urlopen(_gate_url(cls._port, "/health"), timeout=0.5) as r:
                        if r.status == 200:
                            ok = True
                            break
                except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                    time.sleep(0.1)
            if ok:
                return
            if cls._proc:
                cls._proc.terminate()
                cls._proc.wait(timeout=5)
            cls._proc = None
        raise unittest.SkipTest(
            "lifers_gate 在尝试端口范围内未就绪"
            + (f"（最后一次 OSError: {last_err!r}）" if last_err else "")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        proc = getattr(cls, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_get_health(self) -> None:
        with urllib.request.urlopen(_gate_url(self._port, "/health"), timeout=5.0) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read().decode("utf-8"))
        self.assertTrue(data.get("ok"))
        self.assertIn("lifers_root", data)

    def test_post_v1_step(self) -> None:
        body = json.dumps({"text": "ping", "contextFiles": []}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _gate_url(self._port, "/v1/step"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as r:
            self.assertEqual(r.status, 200)
            out = json.loads(r.read().decode("utf-8"))
        self.assertIn("ok", out)
        self.assertIn("text", out)
        self.assertTrue(out.get("ok"), msg=out)

    def test_post_v1_stream_plain(self) -> None:
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=60)
        payload = json.dumps({"text": "ab", "maxChars": 5}, ensure_ascii=False)
        conn.request(
            "POST",
            "/v1/stream",
            body=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertIn("text/plain", resp.getheader("Content-Type", ""))
        raw = resp.read()
        conn.close()
        self.assertGreater(len(raw), 0, msg="stream body empty")


if __name__ == "__main__":
    unittest.main()
