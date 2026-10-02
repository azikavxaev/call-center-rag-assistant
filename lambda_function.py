"""AWS Lambda entry point: POST {"question": "..."} -> {"answer", "sources", "rejected", "error"}.

Invoked through a Lambda function URL behind CloudFront (/api/*). local_server.py calls
the same handler with an equivalent event for local runs.
"""
import base64
import json
import os
import time

import rag

MAX_QUESTION_CHARS = 500
PER_IP_DAILY_LIMIT = int(os.environ.get("PER_IP_DAILY_LIMIT", "20"))
GLOBAL_DAILY_LIMIT = int(os.environ.get("GLOBAL_DAILY_LIMIT", "500"))
RATE_LIMIT_TABLE = os.environ.get("RATE_LIMIT_TABLE")  # unset locally -> no limits

_table = None


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def _client_ip(event):
    # CloudFront passes the visitor's address as "ip:port"; direct calls fall back to the source IP.
    viewer = (event.get("headers") or {}).get("cloudfront-viewer-address", "")
    if viewer:
        return viewer.rsplit(":", 1)[0]
    return event.get("requestContext", {}).get("http", {}).get("sourceIp", "unknown")


def _within_limits(ip):
    """Counts this question against per-visitor and global daily limits (DynamoDB, items expire after 2 days)."""
    global _table
    if not RATE_LIMIT_TABLE:
        return True
    if _table is None:
        import boto3
        _table = boto3.resource("dynamodb").Table(RATE_LIMIT_TABLE)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    expires_at = int(time.time()) + 2 * 86400
    for key, limit in ((f"ip#{ip}#{day}", PER_IP_DAILY_LIMIT), (f"global#{day}", GLOBAL_DAILY_LIMIT)):
        try:
            _table.update_item(
                Key={"pk": key},
                UpdateExpression="ADD hits :one SET expires_at = :exp",
                ConditionExpression="attribute_not_exists(hits) OR hits < :limit",
                ExpressionAttributeValues={":one": 1, ":limit": limit, ":exp": expires_at},
            )
        except _table.meta.client.exceptions.ConditionalCheckFailedException:
            print("RATE LIMITED:", key)
            return False
    return True


def handler(event, context):
    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return _response(405, {"error": "Use POST."})

    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    try:
        question = str(json.loads(body).get("question", "")).strip()
    except (ValueError, AttributeError):
        return _response(400, {"error": "Send JSON like {\"question\": \"...\"}."})
    if not question:
        return _response(400, {"error": "Please type a question."})
    if len(question) > MAX_QUESTION_CHARS:
        return _response(400, {"error": f"Please keep questions under {MAX_QUESTION_CHARS} characters."})

    if not _within_limits(_client_ip(event)):
        return _response(429, {"error": "Daily question limit reached for this demo. Please try again tomorrow."})

    return _response(200, rag.answer(question))
