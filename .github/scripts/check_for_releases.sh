#!/bin/bash

set -eo pipefail

UPSTREAM_REPO="${UPSTREAM_REPO:?UPSTREAM_REPO must be set, e.g. state-spaces/mamba}"
LIMIT="${LIMIT:-3}"
OUTPUT_FILE="${OUTPUT_FILE:-releases.json}"

echo "Fetching last ${LIMIT} stable releases of ${UPSTREAM_REPO}..."
echo "---------------------------------------------------------------------"

gh release list \
  --repo "$UPSTREAM_REPO" \
  --limit "$LIMIT" \
  --exclude-drafts \
  --exclude-pre-releases \
  --json tagName \
  --jq '[.[].tagName]' \
  > "$OUTPUT_FILE"

echo "Resolved release tags:"
cat "$OUTPUT_FILE"
echo
echo "---"
echo "Check complete."
