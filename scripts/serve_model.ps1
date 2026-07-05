# scripts/serve_model.ps1 — TraceAlchemy llama-server Wrapper
# Paths and build settings come from configs/model.yaml + configs/model.local.yaml
# (copy configs/model.example.yaml to model.local.yaml on first setup).
# Usage: .\scripts\serve_model.ps1 [start|stop|status] [-Port 8087]

param(
    [string]$Action = "start",
    [int]$Port = 0
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

function Get-ServeConfig {
    $json = & $python (Join-Path $ScriptDir "resolve_model_paths.py") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host $json
        Write-Host ""
        Write-Host "Setup: copy configs\model.example.yaml to configs\model.local.yaml and set your paths."
        exit 1
    }
    return ($json | ConvertFrom-Json)
}

$cfg = Get-ServeConfig
if ($Port -le 0) { $Port = [int]$cfg.port }

$ServerExe  = $cfg.server_exe
$MainModel  = $cfg.main_model
$DraftModel = $cfg.draft_model
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

    if (-not (Test-Path $ServerExe)) {
        Write-Host "ERROR: llama-server not found: $ServerExe"
        Write-Host "       Set paths.build_dir and paths.binary in configs\model.local.yaml"
        exit 1
    }
    if (-not (Test-Path $MainModel)) {
        Write-Host "ERROR: Main model not found: $MainModel"
        Write-Host "       Set paths.main_model in configs\model.local.yaml"
        exit 1
    }

    Write-Host "Starting TraceAlchemy on port $Port ..."
    Write-Host "   GPU: $($cfg.gpu_arch) | Quant: $($cfg.quantization) | Context: $($cfg.max_context)"

    $args = @(
        "-m", "`"$MainModel`"",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "-c", "$($cfg.max_context)",
        "-ngl", "$($cfg.gpu_layers)",
        "-np", "1",
        "-fa", "on",
        "-ctk", "$($cfg.cache_type_key)",
        "-ctv", "$($cfg.cache_type_value)",
        "-t", "$($cfg.cpu_threads)",
        "--embeddings",
        "--pooling", "mean"
    )

    if ($cfg.speculative_enabled) {
        if (-not (Test-Path $DraftModel)) {
            Write-Host "ERROR: Draft model not found: $DraftModel"
            Write-Host "       Set speculative_decoding.draft_model_path in configs\model.local.yaml"
            exit 1
        }
        $args += @(
            "--mtp-head", "`"$DraftModel`"",
            "--spec-type", "mtp",
            "--draft-block-size", "$($cfg.draft_block_size)"
        )
    }

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
