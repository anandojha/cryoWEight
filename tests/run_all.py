"""Run every test suite and print one verdict.

    python tests/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = (
    "test_cryoweight.py",
    "test_end_to_end.py",
    "test_three_systems.py",
    "test_phase2_equivalence.py",
)

failed = []
for suite in SUITES:
    print(f"==== {suite}")
    result = subprocess.run([sys.executable, os.path.join(HERE, suite)])
    if result.returncode != 0:
        failed.append(suite)

print()
if failed:
    print(f"FAILED: {', '.join(failed)}")
    raise SystemExit(1)
print(f"all {len(SUITES)} suites passed")
