param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# Peer-scale queue (EXP-0029 completion -> EXP-0025 100M/301M -> EXP-0026 301M/498M).
# Every stage is skipped when its output exists and training resumes from the latest checkpoint,
# so the script can simply be relaunched after a shutdown. GPU work runs strictly one job at a time.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/PEER-SCALE-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'
$tokenizer = 'data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json'
$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)

function Write-Status([string]$message) {
    "$(Get-Date -Format o) $message" | Add-Content -LiteralPath $status -Encoding utf8
}

function Invoke-Python([string]$name, [string[]]$arguments) {
    $log = Join-Path $logDir ($name + '.log')
    Write-Status "start $name"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @arguments *>> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) {
        Write-Status "FAILED $name (exit $code); see $log"
        exit $code
    }
    Write-Status "done $name"
}

function Invoke-Train([string]$name, [string]$config, [string]$finalCheckpoint) {
    if (Test-Path -LiteralPath $finalCheckpoint) { Write-Status "skip $name (final checkpoint exists)"; return }
    Invoke-Python $name @('-m', 'slm', 'train', $config, '--set', 'resume=true')
}

function Invoke-Grow([string]$name, [string]$source, [string]$model, [string]$grown, [int]$seed) {
    $check = Join-Path (Split-Path -Parent $grown) 'growth-function-check.json'
    if ((Test-Path -LiteralPath $grown) -and (Test-Path -LiteralPath $check)) { Write-Status "skip $name"; return }
    Invoke-Python "$name-grow" @('-m', 'slm', 'grow', '--checkpoint', $source, '--model', $model,
        '--out', $grown, '--seed', "$seed")
    Invoke-Python "$name-verify" @('scripts/verify_exp0023_growth.py', '--source', $source, '--grown', $grown,
        '--out', $check)
}

function Invoke-MemoryPilot([string]$name, [string]$config) {
    $out = Join-Path $logDir ($name + '.json')
    if (Test-Path -LiteralPath $out) { Write-Status "skip $name"; return }
    # One full-size micro-batch with the run's own memory options, before committing GPU-days.
    Invoke-Python $name @('-m', 'slm', 'bench', $config, '--steps', '2', '--warmup', '1',
        '--set', 'grad_accum_steps=1')
    Get-Content -LiteralPath (Join-Path $logDir ($name + '.log')) | Set-Content -LiteralPath $out -Encoding utf8
}

function Invoke-Evaluation([string]$experiment, [string]$slug, [string]$checkpoint) {
    $outDir = Join-Path $ProjectRoot (Join-Path "runs/$experiment-evaluations" $slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $mc = Join-Path $outDir 'benchmarks.json'
    if (-not (Test-Path -LiteralPath $mc)) {
        Invoke-Python "$experiment-$slug-benchmarks" (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $checkpoint,
            '--tokenizer', $tokenizer, '--batch-items', '2', '--device', 'cuda', '--benchmarks') + $benchmarks +
            @('--out', $mc))
    }
    $bpb = Join-Path $outDir 'hplt-test-bpb.json'
    if (-not (Test-Path -LiteralPath $bpb)) {
        Invoke-Python "$experiment-$slug-hplt-test-bpb" @('scripts/evaluate_exp0028_hplt_test.py',
            '--checkpoint', $checkpoint, '--tokenizer', $tokenizer, '--experiment-id', $experiment, '--out', $bpb)
    }
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "queue started (pid $PID)"

# EXP-0029: finish the HPLT treatment arm, then score both arms.
$exp29Control = 'runs/EXP-0029-100m-old-corpus-control-s20261026/checkpoints/step_00001500.pt'
$exp29Hplt = 'runs/EXP-0029-100m-hplt-treatment-s20261026/checkpoints/step_00001500.pt'
Invoke-Train 'exp0029-hplt-treatment-train' 'configs/train/exp0029-100m-hplt-treatment.yaml' $exp29Hplt
Invoke-Evaluation 'EXP-0029' 'old-corpus-control' $exp29Control
Invoke-Evaluation 'EXP-0029' 'hplt-rich-treatment' $exp29Hplt
if (-not (Test-Path -LiteralPath 'runs/EXP-0029-evaluations/paired-comparison.json')) {
    Invoke-Python 'exp0029-paired-analysis' @('scripts/analyze_exp0029_data_comparison.py',
        '--control-mc', 'runs/EXP-0029-evaluations/old-corpus-control/benchmarks.json',
        '--hplt-mc', 'runs/EXP-0029-evaluations/hplt-rich-treatment/benchmarks.json',
        '--control-test', 'runs/EXP-0029-evaluations/old-corpus-control/hplt-test-bpb.json',
        '--hplt-test', 'runs/EXP-0029-evaluations/hplt-rich-treatment/hplt-test-bpb.json',
        '--out', 'runs/EXP-0029-evaluations/paired-comparison.json')
}

# EXP-0025: 100M continuation control versus 301M function-preserving growth (MobileLLM/SmolLM2 scale).
$grown300 = 'runs/EXP-0025-300m-hplt-growth-s20261026/initialization/300m-grown-from-exp0029-hplt-100m.pt'
Invoke-Grow 'exp0025-300m' $exp29Hplt 'configs/model/slm-300m-fsc-growth.yaml' $grown300 20261026
Invoke-MemoryPilot 'exp0025-300m-memory-pilot' 'configs/train/exp0025-300m-hplt-growth.yaml'
$exp25Control = 'runs/EXP-0025-100m-hplt-control-s20261026/checkpoints/step_00001500.pt'
$exp25Grown = 'runs/EXP-0025-300m-hplt-growth-s20261026/checkpoints/step_00001500.pt'
Invoke-Train 'exp0025-300m-train' 'configs/train/exp0025-300m-hplt-growth.yaml' $exp25Grown
Invoke-Evaluation 'EXP-0025' '300m-treatment' $exp25Grown
Invoke-Train 'exp0025-100m-control-train' 'configs/train/exp0025-100m-hplt-control.yaml' $exp25Control
Invoke-Evaluation 'EXP-0025' '100m-control' $exp25Control

# EXP-0026: 301M continuation control versus 498M growth (Qwen3-0.6B scale), with optimizer offload.
$grown500 = 'runs/EXP-0026-500m-hplt-growth-s20261028/initialization/500m-grown-from-exp0025-300m.pt'
Invoke-Grow 'exp0026-500m' $exp25Grown 'configs/model/slm-500m-fsc-growth.yaml' $grown500 20261028
Invoke-MemoryPilot 'exp0026-500m-memory-pilot' 'configs/train/exp0026-500m-hplt-growth.yaml'
$exp26Grown = 'runs/EXP-0026-500m-hplt-growth-s20261028/checkpoints/step_00001500.pt'
$exp26Control = 'runs/EXP-0026-300m-hplt-control-s20261028/checkpoints/step_00001500.pt'
Invoke-Train 'exp0026-500m-train' 'configs/train/exp0026-500m-hplt-growth.yaml' $exp26Grown
Invoke-Evaluation 'EXP-0026' '500m-treatment' $exp26Grown
Invoke-Train 'exp0026-300m-control-train' 'configs/train/exp0026-300m-hplt-control.yaml' $exp26Control
Invoke-Evaluation 'EXP-0026' '300m-control' $exp26Control

Write-Status 'queue complete'
