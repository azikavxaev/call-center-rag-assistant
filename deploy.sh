#!/usr/bin/env bash
# Publishes code and page changes to the existing AWS resources (eu-central-1).
# The page is uploaded with Cache-Control: no-cache, so CloudFront serves the new version right away.
set -euo pipefail
cd "$(dirname "$0")"

FUNCTION=call-center-assistant
BUCKET=call-center-assistant-web-302277511972
ZIP="$(mktemp -d)/lambda.zip"

zip -qj "$ZIP" rag.py lambda_function.py faq_meta.json
aws lambda update-function-code --function-name "$FUNCTION" --zip-file "fileb://$ZIP" --query LastModified --output text
aws lambda wait function-updated-v2 --function-name "$FUNCTION"

aws s3 cp web/index.html "s3://$BUCKET/index.html" --content-type "text/html; charset=utf-8" --cache-control "no-cache" --only-show-errors

echo "Deployed: https://dqjt38zzjlm5q.cloudfront.net"
