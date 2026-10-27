param(
    [Parameter(Mandatory = $true)][int]$WaitForProcessId,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$runDir = Join-Path $ProjectRoot 'runs/EXP-0025-hplt-growth-check'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$baseCheckpoint = 'runs/EXP-0029-100m-hplt-treatment-s20261026/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $baseCheckpoint)) {
    'EXP-0028 100M checkpoint missing; EXP-0025 not started.' |
        Set-Content -LiteralPath (Join-Path $runDir 'stopped.txt') -Encoding utf8
    exit 2
}

$grown = 'runs/EXP-0025-300m-hplt-growth-s20261026/initialization/300m-grown-from-exp0028-100m.pt'
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -m slm grow --checkpoint $baseCheckpoint --model configs/model/slm-300m-fsc-growth.yaml `
    --out $grown --seed 20261026 *> (Join-Path $runDir 'growth.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) { exit $exitCode }

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python scripts/verify_exp0023_growth.py --source $baseCheckpoint --grown $grown `
    --out runs/EXP-0025-300m-hplt-growth-s20261026/initialization/growth-function-check.json `
    *> (Join-Path $runDir 'growth-verification.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) { exit $exitCode }

# Check one conservative 512-token microbatch before starting either long run.
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -m slm bench configs/train/exp0025-300m-hplt-growth.yaml --steps 1 --warmup 1 `
    --set seq_len=512 micro_batch_size=1 grad_accum_steps=1 *> (Join-Path $runDir 'memory-check.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) {
    "300M memory pilot failed with exit code $exitCode; no EXP-0025 training started." |
        Set-Content -LiteralPath (Join-Path $runDir 'stopped.txt') -Encoding utf8
    exit $exitCode
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -m slm train configs/train/exp0025-100m-hplt-control.yaml *> (Join-Path $runDir '100m-control-train.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) { exit $exitCode }
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -m slm train configs/train/exp0025-300m-hplt-growth.yaml *> (Join-Path $runDir '300m-train.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) { exit $exitCode }

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)
foreach ($arm in @(
    @{ slug='100m-control'; checkpoint='runs/EXP-0025-100m-hplt-control-s20261026/checkpoints/step_00001500.pt' },
    @{ slug='300m-treatment'; checkpoint='runs/EXP-0025-300m-hplt-growth-s20261026/checkpoints/step_00001500.pt' }
)) {
    $outDir = Join-Path $ProjectRoot (Join-Path 'runs/EXP-0025-hplt-evaluations' $arm.slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -m slm eval mc --checkpoint $arm.checkpoint `
        --tokenizer data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json `
        --batch-items 2 --device cuda --benchmarks $benchmarks --out (Join-Path $outDir 'benchmarks.json') `
        *> (Join-Path $outDir 'benchmarks.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python scripts/evaluate_exp0028_hplt_test.py --checkpoint $arm.checkpoint `
        --tokenizer data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json --experiment-id EXP-0025 `
        --out (Join-Path $outDir 'corpus-test-bpb.json') *> (Join-Path $outDir 'corpus-test-bpb.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }
}

"EXP-0025 HPLT scale comparison finished at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $runDir 'complete.txt') -Encoding utf8
