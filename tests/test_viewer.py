"""Unit tests for viewer/index.html's chunked file reader (jsonlLines, logprobStats).

Purpose : the viewer reads large run files (a full control run is ~1 GB) in chunks instead of
          as one string. These tests check that chunking never changes what is read: lines split
          across chunks, a UTF-8 character split across chunks, CRLF endings, a missing final
          newline and empty lines all come out exactly as a whole-file split would give them.
How     : the block between "// BEGIN jsonlLines" and "// END jsonlLines" in viewer/index.html
          is copied into a small Node script and run on synthetic text (no GPQA content).
Needs   : Node 18+ (ReadableStream and TextDecoderStream are built in). Skipped without Node.
Run     : python3 tests/run_all.py
"""
import json
import shutil
import subprocess
from pathlib import Path

VIEWER = Path(__file__).resolve().parent.parent / "viewer" / "index.html"


def _viewer_block() -> str:
    html = VIEWER.read_text(encoding="utf-8")
    start, end = html.index("// BEGIN jsonlLines"), html.index("// END jsonlLines")
    return html[start:end]


def _run_node(script: str) -> dict:
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# Node driver: feeds `text` to jsonlLines as UTF-8 bytes cut into chunks of every size in SIZES,
# and returns what was read for each size, plus logprobStats results.
DRIVER = r"""
%s
const text = %s;
const bytes = new TextEncoder().encode(text);
function streamOf(size) {
  let pos = 0;
  return new ReadableStream({ pull(c) {
    if (pos >= bytes.length) { c.close(); return; }
    c.enqueue(bytes.slice(pos, pos + size)); pos += size; } });
}
(async () => {
  const out = {};
  for (const size of [1, 2, 3, 5, 7, 64, 100000]) {
    const got = [];
    for await (const pair of jsonlLines(streamOf(size))) got.push(pair);
    out[size] = got;
  }
  const big = Array.from({ length: 90000 }, (_, i) => [-(i %% 7) / 10, i, "t"]);
  out.stats = [logprobStats(big), logprobStats([]), logprobStats(undefined), logprobStats([-0.5, -1.5])];
  console.log(JSON.stringify(out));
})();
"""


def test_chunked_reader_matches_whole_file_split():
    if not shutil.which("node"):
        return "SKIP: node not installed"
    # é (2 bytes) and 😀 (4 bytes) get split by the small chunk sizes; CRLF line; empty line;
    # no newline at the end.
    text = '{"a": "é"}\n{"b": "😀 x"}\r\n\n{"c": 3}\n{"d": "end"}'
    want = [[i + 1, line.rstrip("\r")] for i, line in enumerate(text.split("\n"))]
    got = _run_node(DRIVER % (_viewer_block(), json.dumps(text)))
    for size in ("1", "2", "3", "5", "7", "64", "100000"):
        assert got[size] == want, (size, got[size])


def test_logprob_stats_handles_long_outputs_and_empty():
    if not shutil.which("node"):
        return "SKIP: node not installed"
    got = _run_node(DRIVER % (_viewer_block(), json.dumps("x")))["stats"]
    # 90,000 entries: more than the 81,920-token cap, so a spread Math.min(...) would overflow.
    assert got[0]["n"] == 90000 and abs(got[0]["min"] - (-0.6)) < 1e-9
    assert got[1] is None and got[2] is None
    assert got[3] == {"n": 2, "mean": -1.0, "min": -1.5}
