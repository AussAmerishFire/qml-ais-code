# scripts/run_final.ps1 -- the remaining data collection, unattended.
#
#   .\scripts/run_final.ps1
#
# Order is deliberate: the cheap things that make everything else reportable
# run first, so an interrupted night still leaves usable output.
#
#   scripts/11_baseline_viz.py   regenerate the 8 baseline figures on 12 features (~20 min)
#   scripts/15_qnn_confusion.py  regenerate quantum confusion matrices  (~40 min)
#   scripts/18_sweep_viz.py      headline scarcity figure                (~1 min)
#   scripts/19_final.py          10-seed endpoints + TEST evaluation     (~7 h)
#
# scripts/19_final.py checkpoints every run to results/final_results.csv and skips finished
# work on restart, so a shutdown costs at most one model fit.

$ErrorActionPreference = "Continue"
Set-Location -Path (Split-Path $PSScriptRoot -Parent)   # repo root
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$status = "logs/run_final_status.txt"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Path $status -Value $line
}

# Refuse to fight another run for the same CPU. Two heavy jobs on these cores
# roughly doubled each one's wall time earlier in this project.
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "16_sweep|19_final|PIPELINE" }
if ($busy) {
    Say "ABORT -- already running: PID $($busy.ProcessId)"
    exit 1
}

Say "=== run_final started ==="

$steps = @(
    @{ N = "scripts/11_baseline_viz.py";   L = "logs/viz_baseline.log" },
    @{ N = "scripts/15_qnn_confusion.py";  L = "logs/viz_qnn.log" },
    @{ N = "scripts/18_sweep_viz.py";      L = "logs/viz_sweep.log" },
    @{ N = "scripts/19_final.py";          L = "logs/final.log" }
)

foreach ($s in $steps) {
    Say "starting $($s.N)"
    $t0 = Get-Date
    & $py -u $s.N *> $s.L
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Say "$($s.N) exit $code  ($mins min)  -> $($s.L)"
    # Figures failing must not cost the night's compute; 19_final is what
    # matters and it runs last regardless.
    if ($code -ne 0) { Say "  (continuing despite non-zero exit)" }
}

Say "=== run_final complete ==="
