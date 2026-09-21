#!/usr/bin/env bash
# Cold fresh-clone verification of ANY ref (default: the submission tag).
#
#   bash scripts/verify_fresh_clone.sh [ref] [workdir]
#
# Clones the repo into an empty directory, checks out `ref` (resolved to its commit SHA), installs
# dependencies with `uv sync --frozen` against an EMPTY uv cache and a new venv, runs the demo from the
# COMMITTED artifacts, runs the ten `make reproduce` stages in README order, and checks that the
# regenerated results/metrics.json is byte-identical to the committed one. Prints one summary block
# (the commit SHA it actually verified is the first line). Re-run it whenever the tag moves.
set -u
REF=${1:-v1.1-konnect-submission}
D=${2:-$HOME/fresh_clone_verify}
URL=${REPO_URL:-https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking.git}
LOG="$D.log"
: > "$LOG"
stamp() { echo "$(date +%s) $*" >> "$LOG"; }
if [ -e "$D" ]; then echo "refusing to reuse existing $D (must be a cold, empty directory)" >&2; exit 2; fi
T0=$(date +%s)
mkdir -p "$D" && cd "$D" || exit 1
git clone -q "$URL" repo >> "$LOG" 2>&1 || { echo "clone failed"; exit 1; }
cd repo || exit 1
# --verify -q: plain `git rev-parse <unknown>` echoes its argument to stdout, which turned a raw-SHA REF into a
# two-line value and left the clone on the default branch (verified the wrong commit, 2026-09-21).
SHA=$(git rev-parse --verify -q "refs/tags/$REF^{commit}" || git rev-parse --verify "$REF^{commit}")
[ -n "$SHA" ] || { echo "cannot resolve $REF" >&2; exit 2; }
git checkout -q "$SHA" >> "$LOG" 2>&1
export UV_CACHE_DIR="$D/uv-cache"
unset VIRTUAL_ENV
t=$(date +%s); uv sync --frozen >> "$LOG" 2>&1; SYNC_RC=$?; SYNC=$(( $(date +%s) - t ))
export PYTHONHASHSEED=0
t=$(date +%s); uv run python -m poi_rank.cli demo --traveler U0005 > "$D/demo_traveler.txt" 2>&1; DEMO_T_RC=$?; DEMO_T=$(( $(date +%s) - t ))
t=$(date +%s); uv run python -m poi_rank.cli demo --interests local_food,neighborhoods --budget medium \
  --mobility public_transport --touristiness -0.8 --party solo --dest seoul > "$D/demo_profile.txt" 2>&1
DEMO_P_RC=$?; DEMO_P=$(( $(date +%s) - t ))
T1=$(date +%s); FAILED=""
for stage in generate prepare features candidates gate-dgp train evaluate representation \
             gate-representation compose; do
  t=$(date +%s)
  uv run python -m poi_rank.cli "$stage" >> "$D/stage_$stage.log" 2>&1; rc=$?
  stamp "STAGE $stage rc=$rc secs=$(( $(date +%s) - t ))"
  if [ $rc -ne 0 ]; then FAILED="$stage"; break; fi
done
PIPE=$(( $(date +%s) - T1 ))
IDENT=false; git diff --quiet results/metrics.json && IDENT=true
HASH=$(sha256sum results/metrics.json | cut -d' ' -f1)
MODEL_LF=$(git ls-files --eol artifacts/model.txt | awk '{print $1}')
# regenerated model text may contain a few CRLF from LightGBM on Windows: compare ignoring line endings
MODEL_SAME=$(python - <<'PY'
import subprocess
d = open("artifacts/model.txt", "rb").read().replace(b"\r\n", b"\n")
o = subprocess.run(["git", "show", "HEAD:artifacts/model.txt"], capture_output=True).stdout
print("true" if d == o else "false")
PY
)
{
  echo "verified_commit=$SHA"
  echo "ref=$REF"
  echo "uv_sync_seconds=$SYNC rc=$SYNC_RC"
  echo "demo_traveler_seconds=$DEMO_T rc=$DEMO_T_RC"
  echo "demo_profile_seconds=$DEMO_P rc=$DEMO_P_RC"
  echo "pipeline_seconds=$PIPE failed_stage=${FAILED:-none}"
  echo "total_seconds=$(( $(date +%s) - T0 ))"
  echo "metrics_json_identical_to_committed=$IDENT"
  echo "metrics_json_sha256=$HASH"
  echo "committed_model_text_eol=$MODEL_LF"
  echo "regenerated_model_identical_ignoring_line_endings=$MODEL_SAME"
} | tee "$D/summary.txt"
