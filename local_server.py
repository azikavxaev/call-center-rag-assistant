"""Local server: serves web/ and routes POST /api/ask to the Lambda handler.

    python local_server.py            # http://127.0.0.1:8000
"""
import http.server
import os

import rag

rag.load_env_file()

import lambda_function  # noqa: E402  (reads env vars at import)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        event = {
            "requestContext": {"http": {"method": "POST", "sourceIp": self.client_address[0]}},
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": self.rfile.read(length).decode("utf-8"),
        }
        result = lambda_function.handler(event, None)
        body = result["body"].encode("utf-8")
        self.send_response(result["statusCode"])
        for key, value in result["headers"].items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://{host}:{port}")
    http.server.ThreadingHTTPServer((host, port), Handler).serve_forever()
