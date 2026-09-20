# qsvm_handover.ps1
#
# The running 20_qsvm.py loaded SWEEP_N = [571, 1429, 2857] before that list was
# cut to [571, 1429], so it would go on to spend ~18 h on N=2857 after finishing.
# Killing it now would throw away the N=571 seed-2 run already 60% through its
# validation kernel.
#
# So: wait for N=571 to reach 3 seeds on disk, then restart. The restart reads
# qsvm_results.csv, skips everything already done, runs N=1429 only, and exits.
# Nothing is recomputed and nothing is wasted.

Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$log = "qsvm_handover.log"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Say "waiting for N=571 to finish all 3 seeds"

while ($true) {
    Start-Sleep -Seconds 120
    if (-not (Test-Path "qsvm_results.csv")) { continue }
    $done = @(Import-Csv qsvm_results.csv | Where-Object { $_.N -eq "571" }).Count
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
             Where-Object { $_.CommandLine -match "20_qsvm" }

    if ($done -ge 3) {
        Say "N=571 complete ($done seeds). Stopping the old process."
        if ($alive) { $alive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force } }
        Start-Sleep -Seconds 5
        Say "restarting with SWEEP_N = [571, 1429] -- resumes, runs 1429 only"
        Start-Process -FilePath $py -ArgumentList "-u","20_qsvm.py" `
            -WorkingDirectory $PSScriptRoot `
            -RedirectStandardOutput "qsvm2.log" -RedirectStandardError "qsvm2.err" `
            -WindowStyle Hidden
        Say "restarted. N=1429 is ~3 h/seed, ~9 h for three."
        break
    }

    if (-not $alive) {
        Say "process died with only $done of 3 seeds at N=571 -- restarting to resume"
        Start-Process -FilePath $py -ArgumentList "-u","20_qsvm.py" `
            -WorkingDirectory $PSScriptRoot `
            -RedirectStandardOutput "qsvm2.log" -RedirectStandardError "qsvm2.err" `
            -WindowStyle Hidden
        Say "restarted to finish N=571"
    }
}
