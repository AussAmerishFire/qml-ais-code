# scripts/run_confusion.ps1 -- regenerate figures/earlier/fig_q1_confusion.png on the 12-feature data.
# Waits for scripts/19_final.py to finish first; two heavy jobs on these cores roughly
# double each one's wall time.
Set-Location -Path (Split-Path $PSScriptRoot -Parent)   # repo root
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match "19_final" }) {
    Start-Sleep -Seconds 60
}
Write-Host "19_final done -- building confusion figure"
& $py -u scripts/15_qnn_confusion.py *> logs/viz_qnn.log
Write-Host "exit $LASTEXITCODE -> logs/viz_qnn.log"
