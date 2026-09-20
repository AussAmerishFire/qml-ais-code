# run_confusion.ps1 -- regenerate fig_q1_confusion.png on the 12-feature data.
# Waits for 19_final.py to finish first; two heavy jobs on these cores roughly
# double each one's wall time.
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match "19_final" }) {
    Start-Sleep -Seconds 60
}
Write-Host "19_final done -- building confusion figure"
& $py -u 15_qnn_confusion.py *> viz_qnn.log
Write-Host "exit $LASTEXITCODE -> viz_qnn.log"
