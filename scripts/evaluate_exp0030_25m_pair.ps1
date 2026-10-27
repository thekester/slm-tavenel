param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$automationDir = Join-Path $ProjectRoot 'runs/EXP-0030-automation'
$evalDir = Join-Path $ProjectRoot 'runs/EXP-0030-evaluations'
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
    @{ slug='fsc-control'; checkpoint='runs/EXP-0030-25m-fsc-control-s20261027/checkpoints/step_00001500.pt' },
    @{ slug='hplt-treatment'; checkpoint='runs/EXP-0030-25m-hplt-treatment-s20261027/checkpoints/step_00001500.pt' }
)
$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)

foreach ($arm in $arms) {
    $outDir = Join-Path $evalDir $arm.slug
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $devOut = Join-Path $outDir 'development-bpb.json'
    if (-not (Test-Path -LiteralPath $devOut)) {
        Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py', '--checkpoint', $arm.checkpoint,
            '--out', $devOut) (Join-Path $automationDir "$($arm.slug)-development-eval-recovery-$stamp.log")
    }
    $mcOut = Join-Path $outDir 'benchmarks.json'
    if (-not (Test-Path -LiteralPath $mcOut)) {
        $mcArgs = @('-m', 'slm', 'eval', 'mc', '--checkpoint', $arm.checkpoint,
            '--tokenizer', 'data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json',
            '--batch-items', '2', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', $mcOut)
        Invoke-LoggedPython $mcArgs (Join-Path $automationDir "$($arm.slug)-benchmark-eval-recovery-$stamp.log")
    }
}

Invoke-LoggedPython @('scripts/analyze_exp0030_25m_data_diversity.py',
    '--control', 'runs/EXP-0030-evaluations/fsc-control/development-bpb.json',
    '--treatment', 'runs/EXP-0030-evaluations/hplt-treatment/development-bpb.json',
    '--control-mc', 'runs/EXP-0030-evaluations/fsc-control/benchmarks.json',
    '--treatment-mc', 'runs/EXP-0030-evaluations/hplt-treatment/benchmarks.json',
    '--out', 'runs/EXP-0030-evaluations/paired-development-bpb.json') `
    (Join-Path $automationDir "paired-analysis-recovery-$stamp.log")

"EXP-0030 paired evaluation completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $automationDir 'complete.txt') -Encoding utf8
