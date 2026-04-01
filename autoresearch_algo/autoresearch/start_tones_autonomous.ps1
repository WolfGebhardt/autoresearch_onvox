# Defaults tuned for ~12 logical CPUs / 32GB RAM / local Ollama (leave headroom for LLM + GUI).
param(
  [string]$Model = "",
  [int]$MaxCycles = 0,
  [double]$MinImprovement = 0.03,
  [ValidateSet("classic","v2")][string]$OptimizerMode = "v2",
  [int]$ParallelWorkers = 5,
  [int]$BatchSize = 10,
  [int]$Stage1Participants = 4,
  [int]$Stage1TopK = 4,
  [double]$EarlyStopMargin = 0.4,
  [switch]$DisableEarlyStop,
  [switch]$Background,
  [switch]$Stop,
  [switch]$Health,
  [int]$HealthWindow = 100,
  [double]$HealthMaxErrorRate = 0.30,
  [int]$HealthMinEvaluated = 10,
  [int]$HealthMaxStaleMinutes = 8,
  [switch]$Watchdog,
  [int]$WatchdogIntervalSec = 30,
  [int]$WatchdogConsecutiveFails = 3,
  [switch]$Status,
  [switch]$Watch,
  [switch]$PopupWatch,
  [switch]$GuiMonitor,
  [int]$GuiIntervalMs = 4000,
  [int]$GuiSnapshotMinutes = 10,
  [int]$GuiSnapshotDpi = 160,
  [double]$GuiTargetSelectionScore = 11.5,
  [double]$GuiTargetMae = 11.0,
  [double]$GuiTargetPopR = 0.10,
  [double]$GuiTargetTempR = 0.05,
  [int]$OllamaTimeoutSec = 300,
  [int]$OllamaRetries = 3,
  [double]$OllamaRetrySleepSec = 2.0,
  [switch]$RestartOllama
)

$ErrorActionPreference = "Stop"

$tonesRoot = Split-Path -Parent $PSScriptRoot
Set-Location $tonesRoot

Write-Host "[init] TONES root: $tonesRoot"

function Resolve-DefaultModel {
  param(
    [string]$LlmfitExe
  )

  if (-not (Test-Path $LlmfitExe)) {
    Write-Host "[init] llmfit not found, using fallback model."
    return "qwen2.5-coder:7b"
  }

  try {
    $raw = & $LlmfitExe recommend --json --use-case coding --limit 1 --no-dashboard | Out-String
    $obj = $raw | ConvertFrom-Json
    $topName = $obj.models[0].name
    Write-Host "[init] llmfit top coding model: $topName"

    if ($topName -match "Qwen2.5-Coder-7B") {
      return "qwen2.5-coder:7b"
    }
    if ($topName -match "Llama-3.1-8B") {
      return "llama3.1:8b"
    }
    if ($topName -match "Mistral-7B") {
      return "mistral:7b"
    }
  }
  catch {
    Write-Host "[warn] llmfit recommendation parse failed, using fallback."
  }

  return "qwen2.5-coder:7b"
}

if ($RestartOllama) {
  Write-Host "[init] Restarting Ollama (stop existing, then serve)..."
  Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Seconds 2
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
  Start-Sleep -Seconds 4
}

# Ensure Ollama is running
try {
  $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
  Write-Host "[init] Ollama already running."
}
catch {
  Write-Host "[init] Starting Ollama server..."
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
  Start-Sleep -Seconds 3
}

$scriptPath = Join-Path $PSScriptRoot "tones_autonomous_llm_loop.py"
$monitorPath = Join-Path $PSScriptRoot "monitor_autonomous_progress.py"
$monitorGuiPath = Join-Path $PSScriptRoot "monitor_autonomous_gui.py"
$llmfitExe = Join-Path $PSScriptRoot "tools\llmfit\llmfit-v0.8.1-x86_64-pc-windows-msvc\llmfit.exe"
$statusFile = Join-Path $tonesRoot "output\autoresearch\status.json"
$runsFile = Join-Path $tonesRoot "output\autoresearch\autonomous_runs_v2.tsv"
$pidFile = Join-Path $tonesRoot "output\autoresearch\loop.pid"
$logFile = Join-Path $tonesRoot "output\autoresearch\loop.log"
$errFile = Join-Path $tonesRoot "output\autoresearch\loop.err.log"

if ($Stop) {
  if (Test-Path $pidFile) {
    $loopPid = Get-Content $pidFile | Select-Object -First 1
    if ($loopPid) {
      try {
        Stop-Process -Id ([int]$loopPid) -ErrorAction Stop
        Write-Host "[stop] stopped loop pid $loopPid"
      }
      catch {
        Write-Host "[stop] unable to stop pid $loopPid (already exited?)"
      }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
  else {
    Write-Host "[stop] no pid file found at $pidFile"
  }
  exit 0
}

if ($PopupWatch) {
  $watchScript = Join-Path $PSScriptRoot "start_tones_autonomous.ps1"
  $popupCmd = @(
    "`$Host.UI.RawUI.WindowTitle = 'TONES Monitor'"
    "try { `$size = `$Host.UI.RawUI.WindowSize; `$size.Width = 96; `$size.Height = 28; `$Host.UI.RawUI.WindowSize = `$size } catch {}"
    "Set-Location `"$tonesRoot`""
    "powershell -ExecutionPolicy Bypass -File `"$watchScript`" -Watch"
  ) -join "; "
  Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-Command", $popupCmd) -WorkingDirectory $tonesRoot
  Write-Host "[watch] opened popup monitor window."
  exit 0
}

if ($GuiMonitor) {
  $guiCmd = @(
    "`$Host.UI.RawUI.WindowTitle = 'TONES GUI Monitor Launcher'"
    "Set-Location `"$tonesRoot`""
    "python `"$monitorGuiPath`" --status-file `"$statusFile`" --tsv-file `"$runsFile`" --interval-ms $GuiIntervalMs --snapshot-minutes $GuiSnapshotMinutes --snapshot-dpi $GuiSnapshotDpi --target-selection-score $GuiTargetSelectionScore --target-mae $GuiTargetMae --target-pop-r $GuiTargetPopR --target-temp-r $GuiTargetTempR"
  ) -join "; "
  Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-Command", $guiCmd) -WorkingDirectory $tonesRoot
  Write-Host "[watch] opened GUI monitor window."
  exit 0
}

if ($Status -or $Watch -or $Health) {
  $monArgs = @($monitorPath, "--status-file", $statusFile, "--tsv-file", $runsFile)
  if ($Watch) {
    $monArgs += @("--watch", "--tui")
  }
  if ($Health) {
    $monArgs += @(
      "--health",
      "--health-window", "$HealthWindow",
      "--max-error-rate", "$HealthMaxErrorRate",
      "--min-evaluated", "0",
      "--max-stale-minutes", "$HealthMaxStaleMinutes",
      "--require-running"
    )
  }
  python @monArgs
  exit $LASTEXITCODE
}

if ($Model -eq "") {
  $Model = Resolve-DefaultModel -LlmfitExe $llmfitExe
}

Write-Host "[init] selected local model: $Model"

# Ensure selected model is available locally
& ollama show $Model *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "[init] pulling model: $Model"
  & ollama pull $Model
}

$argsList = @($scriptPath, "--max-cycles", "$MaxCycles", "--min-improvement", "$MinImprovement")
$argsList += @("--model", $Model)
$argsList += @("--status-file", $statusFile, "--pid-file", $pidFile, "--log-file", $runsFile)
$argsList += @(
  "--optimizer-mode", $OptimizerMode,
  "--parallel-workers", "$ParallelWorkers",
  "--batch-size", "$BatchSize",
  "--stage1-participants", "$Stage1Participants",
  "--stage1-top-k", "$Stage1TopK",
  "--early-stop-margin", "$EarlyStopMargin"
)
if ($DisableEarlyStop) {
  $argsList += "--disable-early-stop"
}
$argsList += @(
  "--ollama-timeout-sec", "$OllamaTimeoutSec",
  "--ollama-retries", "$OllamaRetries",
  "--ollama-retry-sleep-sec", "$OllamaRetrySleepSec"
)

if ($Watchdog) {
  $interval = [Math]::Max($WatchdogIntervalSec, 5)
  $failLimit = [Math]::Max($WatchdogConsecutiveFails, 1)
  Write-Host "[watchdog] enabled interval=${interval}s fail_limit=$failLimit stale_limit=${HealthMaxStaleMinutes}m"
  $failCount = 0

  while ($true) {
    $healthArgs = @(
      $monitorPath,
      "--status-file", $statusFile,
      "--tsv-file", $runsFile,
      "--health",
      "--health-window", "$HealthWindow",
      "--max-error-rate", "$HealthMaxErrorRate",
      "--min-evaluated", "0",
      "--max-stale-minutes", "$HealthMaxStaleMinutes",
      "--require-running"
    )

    python @healthArgs
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok) {
      if ($failCount -gt 0) {
        Write-Host "[watchdog] health recovered."
      }
      $failCount = 0
      Start-Sleep -Seconds $interval
      continue
    }

    $failCount += 1
    Write-Host "[watchdog] health check failed ($failCount/$failLimit)."
    if ($failCount -lt $failLimit) {
      Start-Sleep -Seconds $interval
      continue
    }

    Write-Host "[watchdog] threshold reached; restarting autonomous loop..."
    if (Test-Path $pidFile) {
      $loopPid = Get-Content $pidFile | Select-Object -First 1
      if ($loopPid) {
        try {
          Stop-Process -Id ([int]$loopPid) -ErrorAction Stop
          Write-Host "[watchdog] stopped loop pid $loopPid"
        }
        catch {
          Write-Host "[watchdog] unable to stop pid $loopPid (already exited?)"
        }
      }
      Remove-Item $pidFile -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Path (Split-Path $logFile) -Force | Out-Null
    $bgArgs = @("-u") + $argsList
    Start-Process -FilePath "python" -ArgumentList $bgArgs -WorkingDirectory $tonesRoot -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Write-Host "[watchdog] restart requested; re-checking in ${interval}s."
    $failCount = 0
    Start-Sleep -Seconds $interval
  }
}

if ($Background) {
  New-Item -ItemType Directory -Path (Split-Path $logFile) -Force | Out-Null
  Write-Host "[run] starting in background"
  $bgArgs = @("-u") + $argsList
  Start-Process -FilePath "python" -ArgumentList $bgArgs -WorkingDirectory $tonesRoot -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden
  Start-Sleep -Seconds 1
  Write-Host "[run] background start requested. Monitor with:"
  Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSScriptRoot\start_tones_autonomous.ps1`" -Status"
}
else {
  Write-Host "[run] python -u $($argsList -join ' ')"
  python -u @argsList
}
