# scripts/run_complete.ps1 -- the complete, all-measured data collection.
#   scripts/22_full_sweep.py  every model, every N, validation + test, 5 seeds  (~6 h)
#   scripts/20_qsvm.py        quantum kernel, N = 571 / 1,429 / 2,857, 5 seeds  (~36 h)
# Sequential on purpose: two heavy jobs on these cores roughly doubled each
# one's wall time earlier in this project. Both checkpoint and resume.
Set-Location -Path (Split-Path $PSScriptRoot -Parent)   # repo root
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$st = "logs/run_complete_status.txt"
function Say($m){ $l="{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"),$m; Add-Content $st $l }
Say "=== started ==="
foreach ($s in @(@{N="scripts/22_full_sweep.py";L="logs/full_sweep.log"}, @{N="scripts/20_qsvm.py";L="logs/qsvm3.log"})) {
    Say "starting $($s.N)"; $t=Get-Date
    & $py -u $s.N *> $s.L
    Say "$($s.N) exit $LASTEXITCODE ($([math]::Round(((Get-Date)-$t).TotalHours,2)) h)"
}
Say "=== complete ==="
