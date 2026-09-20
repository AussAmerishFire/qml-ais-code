# run_all.ps1 -- the whole data-collection chain, unattended.
#
#   .\run_all.ps1
#
# Waits for any running PIPELINE.py, then runs, in order:
#   07_validate_dataset.py    42 property assertions
#   09_independent_check.py   recomputation on a separate code path
#   10_classical_baselines.py five classical baselines, test read once
#   16_sweep.py               the label-scarcity experiment
#
# Every stage logs to its own file and appends a one-line verdict to
# run_all_status.txt, so the morning check is one file.
#
# Safe to re-run. 16_sweep.py resumes from sweep_results.csv, so a shutdown
# costs at most one model fit.

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$status = "run_all_status.txt"

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $status -Value $line
}

Say "=== run_all started ==="

# --- wait for the pipeline rebuild if it is already going -------------------
$pipe = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*PIPELINE.py*" }
if ($pipe) {
    Say "PIPELINE.py already running (PID $($pipe.ProcessId)) -- waiting"
    while (Get-Process -Id $pipe.ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
    Say "PIPELINE.py finished"
} elseif (-not (Test-Path "dataset_train.csv")) {
    Say "running PIPELINE.py"
    & $py -u PIPELINE.py *> "pipeline_v2.log"
    Say "PIPELINE.py exit $LASTEXITCODE"
}

# The datasets have changed shape, so anything tuned on the old 8 features is
# stale. Removing these forces a clean re-selection rather than silently
# reusing hyperparameters chosen for different inputs.
foreach ($f in @("baseline_selection.json", "sweep_results.csv",
                 "qnn_selection.json", "sweep_best_cfg.json")) {
    if (Test-Path $f) { Remove-Item $f -Force; Say "cleared stale $f" }
}

# --- the chain ---------------------------------------------------------------
# -u is deliberate: without it Python buffers stdout when redirected to a file,
# and a 14-hour run shows an empty log until it finishes. That happened.
$steps = @(
    @{ Name = "07_validate_dataset.py";    Log = "validate_v2.log";  Halt = $true  },
    @{ Name = "09_independent_check.py";   Log = "independent_v2.log"; Halt = $true },
    @{ Name = "10_classical_baselines.py"; Log = "baselines_v2.log"; Halt = $false },
    @{ Name = "16_sweep.py";               Log = "sweep.log";        Halt = $false }
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
