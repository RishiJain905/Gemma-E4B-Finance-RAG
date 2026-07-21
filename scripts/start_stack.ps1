# start_stack.ps1 - Bring up the Gemma-E4B-Finance-RAG middleware on Windows.
# NOTE: keep this file pure ASCII - Windows PowerShell 5.1 reads BOM-less
# files as ANSI, and multi-byte characters (em-dashes) misdecode into quote
# characters that break parsing.
# Assumes llama-server is already running on :8087 (started separately with
# --embeddings enabled). Starts the FastAPI middleware on :8000.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

New-Item -ItemType Directory -Force -Path logs | Out-Null

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# Check model server
try {
    $h = Invoke-WebRequest -Uri "http://127.0.0.1:8087/health" -TimeoutSec 5 -UseBasicParsing
    Write-Host "[INFO] llama-server reachable: $($h.Content)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] llama-server not reachable on :8087 - start it manually with --embeddings" -ForegroundColor Yellow
}

# Check configs
foreach ($cfg in @("configs/storage.yaml", "configs/middleware.yaml", "configs/watchlist.yaml")) {
    if (-not (Test-Path $cfg)) { Write-Host "[WARN] Missing config: $cfg" -ForegroundColor Yellow }
}

Write-Host "[INFO] Starting FastAPI middleware on :8000..." -ForegroundColor Green
$mw = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "src.middleware.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info" `
    -RedirectStandardOutput "logs/middleware.log" -RedirectStandardError "logs/middleware.err.log" `
    -PassThru -NoNewWindow
Start-Sleep -Seconds 3

try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5 -UseBasicParsing
    Write-Host "[INFO] Middleware health check passed (HTTP $($r.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Middleware health check failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Stack is running!"
Write-Host "  Middleware:  http://127.0.0.1:8000  (PID $($mw.Id))"
Write-Host "  Docs:        http://127.0.0.1:8000/docs"
Write-Host "  Model:       http://127.0.0.1:8087"
Write-Host ""
Write-Host "To stop: .\scripts\stop_stack.ps1"
