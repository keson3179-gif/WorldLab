"""Serve the V1 replay on localhost. This process does not write research data."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
SYNTHESIS = ROOT / "docs" / "RESEARCH_SYNTHESIS_V020.md"


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO), **kwargs)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/static/index.html")
            self.end_headers()
            return
        if path == "/source/research-synthesis-v020.md":
            body = SYNTHESIS.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        self.send_error(405, "replay is read-only")

    def do_PUT(self) -> None:
        self.send_error(405, "replay is read-only")

    def do_DELETE(self) -> None:
        self.send_error(405, "replay is read-only")


def make_server(port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), PreviewHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the WorldLab V1 replay preview")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"http://127.0.0.1:{args.port}/static/index.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
