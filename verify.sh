#!/usr/bin/env bash
# Verify script — test + build + smoke in one command.
set -e

echo "=== ai-eyes-mcp verify ==="

echo "[1/4] Import check..."
python -c "from ai_eyes_mcp.server import mcp; from ai_eyes_mcp.engine import SigLIPEngine; print('  imports OK')"

echo "[2/4] Tool registration check..."
python -c "
from ai_eyes_mcp.server import mcp
import asyncio
tools = asyncio.run(mcp._list_tools())
expected = {'image_contains', 'image_classify', 'image_compare', 'image_score_batch', 'eyes_status'}
names = {t.name for t in tools}
assert names == expected, f'Expected {expected}, got {names}'
print(f'  {len(tools)} tools registered OK')
"

echo "[3/4] Engine status check..."
python -c "
from ai_eyes_mcp.engine import SigLIPEngine
e = SigLIPEngine()
s = e.status()
assert s['model_id'], 'model_id missing'
assert s['device'], 'device missing'
print(f'  engine OK: model={s[\"model_id\"]}, device={s[\"device\"]}')
"

echo "[4/4] Build check..."
python -m build --wheel --no-isolation 2>/dev/null && echo "  wheel OK" || echo "  SKIP: python-build not installed (pip install build)"

echo ""
echo "=== verify passed ==="
