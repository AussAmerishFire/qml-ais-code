# scripts/run_all.ps1 -- the whole data-collection chain, unattended.
#
#   .\scripts/run_all.ps1
#
# Waits for any running scripts/PIPELINE.py, then runs, in order:
#   scripts/07_validate_dataset.py    42 property assertions
#   scripts/09_independent_check.py   recomputation on a separate code path
#   scripts/10_classical_baselines.py five classical baselines, test read once
#   scripts/16_sweep.py               the label-scarcity experiment
#
# Every stage logs to its own file and appends a one-line verdict to
# logs/run_all_status.txt, so the morning check is one file.
#
# Safe to re-run. scripts/16_sweep.py resumes from results/sweep_results.csv, so a shutdown
# costs at most one model fit.

$ErrorActionPreference = "Continue"
Set-Location -Path (Split-Path $PSScriptRoot -Parent)   # repo root
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$status = "logs/run_all_status.txt"

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $status -Value $line
}

Say "=== run_all started ==="

# --- wait for the pipeline rebuild if it is already going -------------------
$pipe = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*scripts/PIPELINE.py*" }
if ($pipe) {
    Say "scripts/PIPELINE.py already running (PID $($pipe.ProcessId)) -- waiting"
    while (Get-Process -Id $pipe.ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
    Say "scripts/PIPELINE.py finished"
} elseif (-not (Test-Path "data/dataset_train.csv")) {
    Say "running scripts/PIPELINE.py"
    & $py -u scripts/PIPELINE.py *> "logs/pipeline_v2.log"
    Say "scripts/PIPELINE.py exit $LASTEXITCODE"
}

# The datasets have changed shape, so anything tuned on the old 8 features is
# stale. Removing these forces a clean re-selection rather than silently
# reusing hyperparameters chosen for different inputs.
foreach ($f in @("results/baseline_selection.json", "results/sweep_results.csv",
                 "results/qnn_selection.json", "results/sweep_best_cfg.json")) {
    if (Test-Path $f) { Remove-Item $f -Force; Say "cleared stale $f" }
}

# --- the chain ---------------------------------------------------------------
# -u is deliberate: without it Python buffers stdout when redirected to a file,
# and a 14-hour run shows an empty log until it finishes. That happened.
$steps = @(
    @{ Name = "scripts/07_validate_dataset.py";    Log = "logs/validate_v2.log";  Halt = $true  },
    @{ Name = "scripts/09_independent_check.py";   Log = "logs/independent_v2.log"; Halt = $true },
    @{ Name = "scripts/10_classical_baselines.py"; Log = "logs/baselines_v2.log"; Halt = $false },
    @{ Name = "scripts/16_sweep.py";               Log = "logs/sweep.log";        Halt = $false }
)

foreach ($s in $steps) {
    Say "starting $($s.Name)"
    $t0 = Get-Date
    & $py -u $s.Name *> $s.Log
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Say "$($s.Name) exit $code  ($mins min)  -> $($s.Log)"

    if ($code -ne 0 -and $s.Halt) {
        # Validation failing means the dataset itself is wrong. Everything
        # downstream would be measuring broken data, so stop rather than spend
        # the night on it.
        Say "!!! HALTED -- $($s.Name) failed. Dataset is not trustworthy."
        Say "!!! Check $($s.Log) before running anything else."
        exit 1
    }
}

Say "=== run_all complete ==="
