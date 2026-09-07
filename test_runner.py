"""Continuous Automated Test Runner & Verification Loop for GoBot."""

import sys
import os
import time
import subprocess
from typing import List, Dict, Any

# Unset invalid SSLKEYLOGFILE if present in environment
if "SSLKEYLOGFILE" in os.environ and not os.access(os.path.dirname(os.environ["SSLKEYLOGFILE"]) or ".", os.W_OK):
    os.environ.pop("SSLKEYLOGFILE", None)
os.environ.pop("SSLKEYLOGFILE", None)


TEST_SUITES = [
    ("Smoke & Core Engine Tests", "tests/test_smoke.py"),
    ("Capability & Tactical Tests", "tests/test_capabilities.py"),
    ("Training & Curriculum Tests", "tests/test_training.py"),
    ("End-to-End System & API Tests", "tests/test_e2e.py"),
]


def run_suite(name: str, file_path: str) -> Dict[str, Any]:
    t0 = time.time()
    cmd = [sys.executable, "-m", "pytest", file_path, "-v", "--tb=short"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0
    passed = res.returncode == 0

    return {
        "name": name,
        "file": file_path,
        "passed": passed,
        "duration": duration,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


def run_all_tests(max_loops: int = 3) -> bool:
    print("=" * 70)
    print("       GoBot Continuous Verification & Test Runner Loop")
    print("=" * 70)

    loop_count = 0
    all_passed = False

    while loop_count < max_loops and not all_passed:
        loop_count += 1
        print(f"\n[>>>] Starting Test Execution Pass #{loop_count}...\n")
        suite_results = []

        for name, path in TEST_SUITES:
            print(f"  • Running: {name} ({path})... ", end="", flush=True)
            result = run_suite(name, path)
            suite_results.append(result)
            if result["passed"]:
                print(f"[\033[92mPASSED\033[0m] in {result['duration']:.2f}s")
            else:
                print(f"[\033[91mFAILED\033[0m] in {result['duration']:.2f}s")
                print("\n--- Failure Details ---")
                print(result["stdout"])
                print(result["stderr"])
                print("-----------------------\n")

        all_passed = all(r["passed"] for r in suite_results)
        total_time = sum(r["duration"] for r in suite_results)

        print("\n" + "-" * 70)
        print(f"Pass #{loop_count} Summary: {'ALL PASSED' if all_passed else 'SOME FAILED'} | Total Duration: {total_time:.2f}s")
        print("-" * 70)

        if not all_passed and loop_count < max_loops:
            print("[!] Tests failed. Retrying verification loop...")
            time.sleep(1)

    if all_passed:
        print("\n\033[92m[SUCCESS] All smoke, capability, training, and e2e tests passed successfully!\033[0m\n")
        return True
    else:
        print("\n\033[91m[FAILURE] Test loop finished with failing tests.\033[0m\n")
        return False


if __name__ == "__main__":
    success = run_all_tests(max_loops=1)
    sys.exit(0 if success else 1)
