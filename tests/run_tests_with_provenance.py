import subprocess
import json
import hashlib
import re
import datetime
import sys
from pathlib import Path

def run_tests():
    # Run pytest
    cmd = [sys.executable, "-m", "pytest"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Write log
    log_path = Path("tests/pytest_provenance.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(process.stdout)
        if process.stderr:
            f.write("\nSTDERR:\n" + process.stderr)
            
    # Calculate log sha256
    sha256_hash = hashlib.sha256()
    with open(log_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    log_sha256 = sha256_hash.hexdigest()
    
    # Get git head sha
    git_cmd = ["git", "rev-parse", "HEAD"]
    git_process = subprocess.run(git_cmd, capture_output=True, text=True)
    head_sha = git_process.stdout.strip()
    
    # Parse passed/failed from pytest output
    output = process.stdout
    passed = 0
    failed = 0
    
    lines = output.split('\n')
    summary_line = ""
    for line in reversed(lines):
        if " passed" in line or " failed" in line or " error" in line:
            summary_line = line
            break
            
    if summary_line:
        passed_match = re.search(r'(\d+)\s+passed', summary_line)
        if passed_match:
            passed = int(passed_match.group(1))
            
        failed_match = re.search(r'(\d+)\s+failed', summary_line)
        if failed_match:
            failed = int(failed_match.group(1))
            
    metadata = {
        "head_sha": head_sha,
        "command": f"{sys.executable} -m pytest",
        "passed": passed,
        "failed": failed,
        "log_sha256": log_sha256,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "return_code": process.returncode,
        "status": "PASS" if process.returncode == 0 else "FAIL"
    }
    
    meta_path = Path("tests/test_provenance.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Test run complete. Passed: {passed}, Failed: {failed}, Return Code: {process.returncode}")
    print(f"Log written to {log_path}")
    print(f"Metadata written to {meta_path}")
    
    sys.exit(process.returncode)

if __name__ == "__main__":
    run_tests()
