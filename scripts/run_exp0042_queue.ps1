param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0042: LFM3-VL-2B alone / with a host tool / with the compiled machine on 700 seeded arithmetic
# items (GPU, inference only). Resumable (answers appended per arm); serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = Join-Path $ProjectRoot 'data/envs/tf5'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0042-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'

function Write-Status([string]$message) {
    "$(Get-Date -Format o) $message" | Add-Content -LiteralPath $status -Encoding utf8
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
Write-Status "EXP-0042 queue started (pid $PID)"
if (-not (Test-Path -LiteralPath 'runs/EXP-0042/summary.json') -or
    ((Get-Content 'runs/EXP-0042/summary.json' -Raw | ConvertFrom-Json).answered.C -lt 700)) {
    $log = Join-Path $logDir 'exp0042-run.log'
    Write-Status 'start exp0042-run'
    $ErrorActionPreference = 'Continue'
    & $python 'scripts/run_exp0042.py' '--device' 'cuda' *>> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { Write-Status "FAILED exp0042-run (exit $code); see $log"; exit $code }
    Write-Status 'done exp0042-run'
}
Write-Status 'EXP-0042 complete'
