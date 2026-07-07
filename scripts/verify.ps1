# scripts/verify.ps1 — deterministic verification gate for agent loops
# Runs ruff + the offline pytest suite; the FINAL line is a machine-readable verdict.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 [-TestPath tests\test_x.py]
#   -TestPath scopes pytest to one file for fast inner-loop iteration (ruff still runs on everything).
# Output contract: last line is "VERIFY: PASS" (exit 0) or "VERIFY: FAIL (<stages>)" (exit 1).
# This line is the stop condition for /goal loops — see how-to-loop.md.

param(
    [string]$TestPath = "tests/"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$failed = @()

Write-Host "== ruff =="
& $python -m ruff check .
if ($LASTEXITCODE -ne 0) { $failed += "ruff" }

Write-Host "== pytest (offline: $TestPath) =="
& $python -m pytest $TestPath -q -m "not live" --maxfail=10
if ($LASTEXITCODE -ne 0) { $failed += "pytest" }

if ($failed.Count -eq 0) {
    Write-Host "VERIFY: PASS"
    exit 0
} else {
    Write-Host ("VERIFY: FAIL ({0})" -f ($failed -join ", "))
    exit 1
}
