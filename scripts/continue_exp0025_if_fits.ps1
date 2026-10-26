param(
    [Parameter(Mandatory = $true)][int]$WaitForProcessId,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$runDir = Join-Path $ProjectRoot 'runs/EXP-0025-300m-growth-check'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$baseCheckpoint = 'runs/EXP-0024-100m-fsc-grown-s20261025/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $baseCheckpoint)) {
    'EXP-0024 100M final checkpoint is missing; EXP-0025 was not started.' |
        Set-Content -LiteralPath (Join-Path $runDir 'stopped.txt') -Encoding utf8
    exit 2
}

$grown = 'runs/EXP-0025-300m-grown-s20261025/initialization/300m-grown-from-exp0024-100m.pt'
& $python -m slm grow --checkpoint $baseCheckpoint --model configs/model/slm-300m-fsc-growth.yaml `
    --out $grown --seed 20261025 *> (Join-Path $runDir 'growth.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts/verify_exp0023_growth.py --source $baseCheckpoint --grown $grown `
    --out runs/EXP-0025-300m-grown-s20261025/initialization/growth-function-check.json `
    *> (Join-Path $runDir 'growth-verification.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Measure the activation/optimizer memory with one 512-token micro-batch before committing
# to the long run. The production configs use the same shape and accumulate 64 such batches.
& $python -m slm bench configs/train/exp0025-300m-fsc-grown.yaml --steps 1 --warmup 1 `
    --set seq_len=512 micro_batch_size=1 grad_accum_steps=1 `
    *> (Join-Path $runDir 'memory-check.log')
if ($LASTEXITCODE -ne 0) {
    "300M memory pilot failed with exit code $LASTEXITCODE; no EXP-0025 training started." |
        Set-Content -LiteralPath (Join-Path $runDir 'stopped.txt') -Encoding utf8
    exit $LASTEXITCODE
}

& $python -m slm train configs/train/exp0025-100m-control.yaml *> (Join-Path $runDir '100m-control.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m slm train configs/train/exp0025-300m-fsc-grown.yaml *> (Join-Path $runDir '300m-train.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts/evaluate_exp0023_50m.py `
    --checkpoint runs/EXP-0025-100m-control-s20261025/checkpoints/step_00001500.pt `
    --out runs/EXP-0025-100m-control-s20261025/evaluation.json `
    --resume-progress *> (Join-Path $runDir '100m-control-evaluation.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts/evaluate_exp0023_50m.py `
    --checkpoint runs/EXP-0025-300m-grown-s20261025/checkpoints/step_00001500.pt `
    --out runs/EXP-0025-300m-grown-s20261025/evaluation.json `
    --resume-progress *> (Join-Path $runDir '300m-evaluation.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

"EXP-0025 training and evaluation finished at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $runDir 'complete.txt') -Encoding utf8
