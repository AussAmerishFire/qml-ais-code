# resume.ps1 -- restart the sweep after a reboot, crash, or shutdown.
#
#   Right-click -> Run with PowerShell,  or:  .\resume.ps1
#
# Safe to run at any time. Every completed run is already in sweep_results.csv,
# and 16_sweep.py skips work it finds there, so this resumes rather than
# restarting. Running it while the sweep is already going is caught below.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No virtualenv found. Create it first:" -ForegroundColor Yellow
    Write-Host "  py -3.13 -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

# Don't start a second copy -- two sweeps would fight for the same CPU and
# both would slow down, which is what happened when the probe and the sweep
# ran together.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like "*16_sweep.py*" }
if ($running) {
    Write-Host "Sweep is ALREADY running (PID $($running.ProcessId)). Nothing to do." -ForegroundColor Green
    exit 0
}

if (Test-Path "sweep_results.csv") {
    $n = (Import-Csv sweep_results.csv).Count
    Write-Host "Found $n completed runs on disk -- these will be skipped." -ForegroundColor Cyan
} else {
    Write-Host "No checkpoint found; starting from the beginning." -ForegroundColor Cyan
}

if (Test-Path "sweep.log") {
    Copy-Item "sweep.log" ("sweep_prev_{0}.log" -f (Get-Date -Format "MMdd_HHmm")) -Force
}

Start-Process -FilePath $py -ArgumentList "16_sweep.py" `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput "sweep.log" -RedirectStandardError "sweep.err" `
    -WindowStyle Hidden

Start-Sleep -Seconds 20
Write-Host "`nStarted. First lines of sweep.log:" -ForegroundColor Green
Get-Content sweep.log -Tail 6

Write-Host "`nProgress:  Get-Content sweep.log -Tail 20 -Wait" -ForegroundColor DarkGray
Write-Host "Stop:      Get-CimInstance Win32_Process -Filter `"Name='python.exe'`" | Where-Object { `$_.CommandLine -like '*16_sweep.py*' } | ForEach-Object { Stop-Process -Id `$_.ProcessId }" -ForegroundColor DarkGray
