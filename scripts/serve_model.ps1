# scripts/serve_model.ps1 — TraceAlchemy llama-server Wrapper
# For Windows + AMD RDNA3
# Usage: .\scripts\serve_model.ps1 [start|stop|status]

param(
    [string]$Action = "start",
    [int]$Port = 8087
)

$BuildDir   = "F:\Personal\TQ-Optimizer-Test\atomic-llama-cpp-turboquant\build-rdna3-gfx1101"
$ServerExe  = "$BuildDir\bin\llama-server.exe"
$MainModel  = "D:\LOCAL-MODELS\trjxter\TraceAlchemy-Gemma-4-E4B-Finance-IT-gguf\gemma-4-E4B-it.Q8_0.gguf"
$DraftModel = "D:\LOCAL-MODELS\AtomicChat\gemma-4-E4B-it-assistant-GGUF\gemma-4-E4B-it-assistant.F16.gguf"
$LogFile    = "$env:USERPROFILE\models\tracealchemy\logs\server.log"
$PidFile    = "$env:USERPROFILE\models\tracealchemy\logs\server.pid"

function Start-Server {
    New-Item -ItemType Directory -Force -Path (Split-Path $LogFile -Parent) | Out-Null

    if (Test-Path $PidFile) {
        $serverPid = [int](Get-Content $PidFile)
        if (Get-Process -Id $serverPid -ErrorAction SilentlyContinue) {
            Write-Host "ERROR: Server already running (PID $serverPid)"
            exit 1
        }
    }

    Write-Host "Starting TraceAlchemy on port $Port ..."
    Write-Host "   GPU: AMD RDNA3 (gfx1101) | Quant: Q8_0 | Context: 131072"

    $args = @(
        "-m", "`"$MainModel`"",
        "--mtp-head", "`"$DraftModel`"",
        "--spec-type", "mtp",
        "--draft-block-size", "3",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "-c", "131072",
        "-ngl", "99",
        "-np", "1",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "turbo4",
        "-t", "8",
        "--embeddings",
        "--pooling", "mean"
    )

    $proc = Start-Process -FilePath $ServerExe -ArgumentList $args -NoNewWindow -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" -PassThru
    $proc.Id | Out-File -FilePath $PidFile -Encoding ascii

    Write-Host "Server started (PID $($proc.Id))"
    Write-Host "   API: http://127.0.0.1:$Port"
    Write-Host "   Log: $LogFile"

    Start-Sleep -Seconds 2
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        Write-Host "   Status: running"
    } else {
        Write-Host "   WARNING: Exited early. Check log: $LogFile"
        Get-Content $LogFile -Tail 5
    }
}

function Stop-Server {
    if (Test-Path $PidFile) {
        $serverPid = [int](Get-Content $PidFile)
        if (Get-Process -Id $serverPid -ErrorAction SilentlyContinue) {
            Write-Host "Stopping TraceAlchemy (PID $serverPid)..."
            Stop-Process -Id $serverPid -Force
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped."
    } else {
        Write-Host "INFO: No PID file found."
        $proc = Get-Process | Where-Object { $_.ProcessName -like "*llama-server*" } | Select-Object -First 1
        if ($proc) {
            Write-Host "   Found llama-server (PID $($proc.Id)). Stopping..."
            Stop-Process -Id $proc.Id -Force
            Write-Host "Stopped."
        } else {
            Write-Host "   No server running."
        }
    }
}

function Get-Status {
    $running = $false

    if (Test-Path $PidFile) {
        $serverPid = [int](Get-Content $PidFile)
        if (Get-Process -Id $serverPid -ErrorAction SilentlyContinue) {
            Write-Host "TraceAlchemy is running (PID $serverPid)"
            Write-Host "   API: http://127.0.0.1:$Port"
            $running = $true
        }
    }

    if (-not $running) {
        $proc = Get-Process | Where-Object { $_.ProcessName -like "*llama-server*" } | Select-Object -First 1
        if ($proc) {
            Write-Host "llama-server running (PID $($proc.Id)) - no PID file"
            $running = $true
        }
    }

    if (-not $running) {
        Write-Host "TraceAlchemy is not running"
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

switch ($Action) {
    "start"   { Start-Server }
    "stop"    { Stop-Server }
    "restart" { Stop-Server; Start-Sleep 1; Start-Server }
    "status"  { Get-Status }
    default   {
        Write-Host "Usage: .\scripts\serve_model.ps1 {start|stop|restart|status} [-Port 8087]"
    }
}
