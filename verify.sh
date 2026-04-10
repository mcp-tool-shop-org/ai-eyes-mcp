#!/usr/bin/env bash
# Verify script — test + build + smoke in one command.
set -e

echo "=== ai-eyes-mcp verify ==="

echo "[1/5] Import check..."
python -c "from ai_eyes_mcp.server import mcp; from ai_eyes_mcp.engine import SigLIPEngine; print('  imports OK')"

echo "[2/5] Tool registration check..."
python -c "
from ai_eyes_mcp.server import mcp
import asyncio
tools = asyncio.run(mcp._list_tools())
expected = {'image_contains', 'image_classify', 'image_compare', 'image_score_batch', 'eyes_status'}
names = {t.name for t in tools}
assert names == expected, f'Expected {expected}, got {names}'
print(f'  {len(tools)} tools registered OK')
"

echo "[3/5] Engine status check..."
python -c "
from ai_eyes_mcp.engine import SigLIPEngine
e = SigLIPEngine()
s = e.status()
assert s['model_id'], 'model_id missing'
assert s['device'], 'device missing'
print(f'  engine OK: model={s[\"model_id\"]}, device={s[\"device\"]}')
"

echo "[4/5] Build check..."
if python -m build --help > /dev/null 2>&1; then
  python -m build --wheel --no-isolation
  echo "  wheel OK"
else
  echo "  SKIP: python-build not installed (pip install build)"
fi

echo "[5/5] Edge-case tests..."
pytest tests/test_edge_cases.py -v

echo ""
echo "=== verify passed ==="
