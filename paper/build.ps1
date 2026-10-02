# Builds paper/main.pdf in one step. Run it from anywhere:
#
#     powershell -ExecutionPolicy Bypass -File paper\build.ps1
#
# 1. Regenerates every table from the result CSVs (scripts/27_paper_tables.py),
#    so the PDF can never show a number the data does not.
# 2. Runs pdflatex -> bibtex -> pdflatex -> pdflatex, the sequence LaTeX needs
#    to resolve citations and cross-references.
# 3. Reports the page count and any errors, undefined references or overflows.

$ErrorActionPreference = "Stop"
$paper = $PSScriptRoot
$root  = Split-Path $paper -Parent

$miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktex) { $env:PATH = "$miktex;$env:PATH" }
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host "pdflatex not found. Install MiKTeX: winget install MiKTeX.MiKTeX" -ForegroundColor Red
    exit 1
}

Push-Location $root
& (Join-Path $root ".venv\Scripts\python.exe") scripts\27_paper_tables.py | Out-Null
Pop-Location

Push-Location $paper
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null

$log = Get-Content main.log -Raw
$written = (Select-String -Path main.log -Pattern "Output written on main.pdf \((\d+) pages").Matches
$errors   = ([regex]::Matches($log, "(?m)^!")).Count
$undef    = ([regex]::Matches($log, "undefined")).Count
$overfull = ([regex]::Matches($log, "Overfull")).Count
Pop-Location

if ($written.Count -gt 0) {
    Write-Host "Built paper\main.pdf: $($written[0].Groups[1].Value) pages" -ForegroundColor Green
} else {
    Write-Host "No PDF produced. See paper\main.log" -ForegroundColor Red
}
Write-Host "errors: $errors   undefined references: $undef   overflows: $overfull"
