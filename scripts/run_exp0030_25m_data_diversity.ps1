param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$automationDir = Join-Path $ProjectRoot 'runs/EXP-0030-automation'
New-Item -ItemType Directory -Force -Path $automationDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

function Invoke-LoggedPython {
    param([string[]]$Arguments, [string]$LogPath)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @Arguments *> $LogPath
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "Python command failed ($code). See $LogPath" }
}

$arms = @(
    @{ slug='fsc-control'; config='configs/train/exp0030-25m-fsc-control.yaml';
       checkpoint='runs/EXP-0030-25m-fsc-control-s20261027/checkpoints/step_00001500.pt' },
    @{ slug='hplt-treatment'; config='configs/train/exp0030-25m-hplt-treatment.yaml';
       checkpoint='runs/EXP-0030-25m-hplt-treatment-s20261027/checkpoints/step_00001500.pt' }
)

foreach ($arm in $arms) {
    if (-not (Test-Path -LiteralPath $arm.checkpoint)) {
        Invoke-LoggedPython @('-m', 'slm', 'train', $arm.config) `
            (Join-Path $automationDir "$($arm.slug)-train-$stamp.log")
    }
}

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)
foreach ($arm in $arms) {
    $outDir = Join-Path $ProjectRoot (Join-Path 'runs/EXP-0030-evaluations' $arm.slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $checkpoint = $arm.checkpoint
    Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py', '--checkpoint', $checkpoint,
        '--out', (Join-Path $outDir 'development-bpb.json')) `
        (Join-Path $automationDir "$($arm.slug)-development-eval-$stamp.log")
    $mcArgs = @('-m', 'slm', 'eval', 'mc', '--checkpoint', $checkpoint,
        '--tokenizer', 'data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json',
        '--batch-items', '2', '--device', 'cuda', '--benchmarks') + $benchmarks + @(
        '--out', (Join-Path $outDir 'benchmarks.json'))
    Invoke-LoggedPython $mcArgs `
        (Join-Path $automationDir "$($arm.slug)-benchmark-eval-$stamp.log")
}

Invoke-LoggedPython @('scripts/analyze_exp0030_25m_data_diversity.py',
    '--control', 'runs/EXP-0030-evaluations/fsc-control/development-bpb.json',
    '--treatment', 'runs/EXP-0030-evaluations/hplt-treatment/development-bpb.json',
    '--control-mc', 'runs/EXP-0030-evaluations/fsc-control/benchmarks.json',
    '--treatment-mc', 'runs/EXP-0030-evaluations/hplt-treatment/benchmarks.json',
    '--out', 'runs/EXP-0030-evaluations/paired-development-bpb.json') `
    (Join-Path $automationDir "paired-analysis-$stamp.log")

"EXP-0030 pilot completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $automationDir 'complete.txt') -Encoding utf8
