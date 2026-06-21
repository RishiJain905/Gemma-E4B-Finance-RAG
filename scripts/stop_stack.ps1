# stop_stack.ps1 — Gracefully shut down the RAG middleware on Windows.
# Stops the uvicorn middleware process; leaves llama-server running (it is
# managed separately).

Write-Host "Shutting down RAG middleware..."
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "uvicorn" -and $_.CommandLine -match "src.middleware.app" }

if ($procs) {
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped middleware (PID $($p.ProcessId))"
    }
} else {
    Write-Host "Middleware not running"
}
Write-Host "Done. (llama-server left running — stop it manually if needed.)"
