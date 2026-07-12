#!/usr/bin/env python3
from __future__ import annotations

import http.server
import socketserver
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "viewer"
PORT = 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"Serving {ROOT}")
    print(f"http://127.0.0.1:{PORT}")
    httpd.serve_forever()
