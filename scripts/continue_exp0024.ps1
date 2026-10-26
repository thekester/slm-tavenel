param(
    [Parameter(Mandatory = $true)][int]$WaitForProcessId,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$logDir = Join-Path $ProjectRoot 'runs/EXP-0024-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$controlDir = Join-Path $ProjectRoot 'runs/EXP-0024-50m-control-s20261025'
$controlMetrics = Join-Path $controlDir 'metrics.jsonl'
if (-not (Test-Path -LiteralPath (Join-Path $controlDir 'checkpoints/step_00001500.pt'))) {
    "Control process exited without a step-1500 checkpoint; stopping follow-up." |
        Set-Content -LiteralPath (Join-Path $logDir 'stopped.txt') -Encoding utf8
    exit 2
}

& $python -m slm train configs/train/exp0024-100m-fsc-grown.yaml *> (Join-Path $logDir '100m-train.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts/evaluate_exp0023_50m.py `
    --checkpoint runs/EXP-0024-50m-control-s20261025/checkpoints/step_00001500.pt `
    --out runs/EXP-0024-50m-control-s20261025/evaluation.json `
    --resume-progress *> (Join-Path $logDir '50m-evaluation.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts/evaluate_exp0023_50m.py `
    --checkpoint runs/EXP-0024-100m-fsc-grown-s20261025/checkpoints/step_00001500.pt `
    --out runs/EXP-0024-100m-fsc-grown-s20261025/evaluation.json `
    --resume-progress *> (Join-Path $logDir '100m-evaluation.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

"Training and evaluations completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $logDir 'complete.txt') -Encoding utf8
