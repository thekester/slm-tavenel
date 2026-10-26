param(
    [Parameter(Mandatory = $true)][int]$WaitForProcessId,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0024-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

function Invoke-PythonLogged([string]$LogName, [string[]]$Arguments) {
    & $python @Arguments *> (Join-Path $logDir $LogName)
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

$control50 = 'runs/EXP-0024-50m-control-s20261025/checkpoints/step_00001500.pt'
$model100 = 'runs/EXP-0024-100m-fsc-grown-s20261025/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $model100)) {
    'EXP-0024 100M did not reach step 1500; no downstream jobs were started.' |
        Set-Content -LiteralPath (Join-Path $logDir 'stopped.txt') -Encoding utf8
    exit 2
}

Invoke-PythonLogged '50m-control-evaluation.log' @(
    'scripts/evaluate_exp0023_50m.py', '--checkpoint', $control50,
    '--out', 'runs/EXP-0024-50m-control-s20261025/evaluation.json', '--resume-progress'
)
Invoke-PythonLogged '100m-evaluation.log' @(
    'scripts/evaluate_exp0023_50m.py', '--checkpoint', $model100,
    '--out', 'runs/EXP-0024-100m-fsc-grown-s20261025/evaluation.json', '--resume-progress'
)

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)
$peerStatus = @()
foreach ($ref in @(
    @{ slug='smollm2-360m'; repo='HuggingFaceTB/SmolLM2-360M'; revision='f8027fd0eaeea54caa13c31d31b9fdc459c38b49' },
    @{ slug='qwen3-06b'; repo='Qwen/Qwen3-0.6B'; revision='c1899de289a04d12100db370d81485cdf75e47ca' }
)) {
    $outDir = Join-Path $ProjectRoot (Join-Path 'runs/reference/peer-models-2026-10' $ref.slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outFile = Join-Path $outDir 'mc.json'
    $args = @('-m','slm','eval','mc','--hf-model',$ref.repo,'--hf-revision',$ref.revision,
              '--max-seq-len','1024','--device','cuda','--benchmarks') + $benchmarks + @('--out',$outFile)
    & $python @args *> (Join-Path $logDir ($ref.slug + '-evaluation.log'))
    $peerStatus += [pscustomobject]@{ repo=$ref.repo; revision=$ref.revision; exit_code=$LASTEXITCODE; result=$outFile }
}
$peerStatus | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ProjectRoot 'runs/reference/peer-models-2026-10/public-peer-evaluation-status.json') -Encoding utf8

$checkDir = Join-Path $ProjectRoot 'runs/EXP-0025-300m-growth-check'
New-Item -ItemType Directory -Force -Path $checkDir | Out-Null
$grown = 'runs/EXP-0025-300m-grown-s20261025/initialization/300m-grown-from-exp0024-100m.pt'
Invoke-PythonLogged '300m-growth.log' @(
    '-m','slm','grow','--checkpoint',$model100,'--model','configs/model/slm-300m-fsc-growth.yaml',
    '--out',$grown,'--seed','20261025'
)
Invoke-PythonLogged '300m-growth-verification.log' @(
    'scripts/verify_exp0023_growth.py','--source',$model100,'--grown',$grown,
    '--out','runs/EXP-0025-300m-grown-s20261025/initialization/growth-function-check.json'
)
& $python -m slm bench configs/train/exp0025-300m-fsc-grown.yaml --steps 1 --warmup 1 `
    --set seq_len=512 micro_batch_size=1 grad_accum_steps=1 *> (Join-Path $checkDir 'memory-check.log')
if ($LASTEXITCODE -ne 0) {
    "300M memory check failed ($LASTEXITCODE); no EXP-0025 training started." |
        Set-Content -LiteralPath (Join-Path $checkDir 'stopped.txt') -Encoding utf8
    exit $LASTEXITCODE
}

Invoke-PythonLogged '100m-control-train.log' @('-m','slm','train','configs/train/exp0025-100m-control.yaml')
Invoke-PythonLogged '300m-train.log' @('-m','slm','train','configs/train/exp0025-300m-fsc-grown.yaml')
Invoke-PythonLogged '100m-control-evaluation.log' @(
    'scripts/evaluate_exp0023_50m.py','--checkpoint','runs/EXP-0025-100m-control-s20261025/checkpoints/step_00001500.pt',
    '--out','runs/EXP-0025-100m-control-s20261025/evaluation.json','--resume-progress'
)
Invoke-PythonLogged '300m-evaluation.log' @(
    'scripts/evaluate_exp0023_50m.py','--checkpoint','runs/EXP-0025-300m-grown-s20261025/checkpoints/step_00001500.pt',
    '--out','runs/EXP-0025-300m-grown-s20261025/evaluation.json','--resume-progress'
)
"EXP-0024 and conditional EXP-0025 sequence finished at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $checkDir 'complete.txt') -Encoding utf8
