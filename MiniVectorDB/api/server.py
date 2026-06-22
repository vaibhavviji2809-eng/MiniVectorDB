"""Minimal HTTP server for the MiniVectorDB API."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict

from .routes import MiniVectorDBRoutes


class RequestHandler(BaseHTTPRequestHandler):
    routes = MiniVectorDBRoutes()

    def _write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"ok": True})
        else:
            self._write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        mapping: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "/collection/create": self.routes.create_collection,
            "/insert": self.routes.insert,
            "/search": self.routes.search,
            "/delete": self.routes.delete,
        }
        handler = mapping.get(self.path)
        if handler is None:
            self._write_json({"error": "not found"}, status=404)
            return
        try:
            self._write_json(handler(payload))
        except Exception as exc:  # pragma: no cover - surfaced through API
            self._write_json({"error": str(exc)}, status=400)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), RequestHandler)


def main() -> None:
    server = create_server()
    print("MiniVectorDB listening on http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()

