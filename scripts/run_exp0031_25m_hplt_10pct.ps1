param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$automationDir = Join-Path $ProjectRoot 'runs/EXP-0031-automation'
$evalDir = Join-Path $ProjectRoot 'runs/EXP-0031-evaluations'
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

$dataDir = 'data/tokens/exp0031-25m-hplt-10pct-fsc-tokenizer'
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "$dataDir/manifest.json"))) {
    Invoke-LoggedPython @('scripts/prepare_exp0031_25m_hplt_10pct.py') `
        (Join-Path $automationDir "prepare-$stamp.log")
}

$controlDir = Join-Path $evalDir 'fsc-control'
$treatmentDir = Join-Path $evalDir 'hplt-10pct-treatment'
New-Item -ItemType Directory -Force -Path $controlDir,$treatmentDir | Out-Null
$controlSource = Join-Path $ProjectRoot 'runs/EXP-0030-evaluations/fsc-control'
foreach ($file in @('development-bpb.json','benchmarks.json')) {
    $src = Join-Path $controlSource $file
    $dst = Join-Path $controlDir $file
    if (-not (Test-Path -LiteralPath $src)) { throw "Missing completed EXP-0030 control result: $src" }
    if (-not (Test-Path -LiteralPath $dst)) { Copy-Item -LiteralPath $src -Destination $dst }
}

$checkpoint = 'runs/EXP-0031-25m-hplt-10pct-s20261027/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $checkpoint)) {
    Invoke-LoggedPython @('-m','slm','train','configs/train/exp0031-25m-hplt-10pct.yaml') `
        (Join-Path $automationDir "train-$stamp.log")
}
if (-not (Test-Path -LiteralPath (Join-Path $treatmentDir 'development-bpb.json'))) {
    Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py','--checkpoint',$checkpoint,
        '--out',(Join-Path $treatmentDir 'development-bpb.json')) `
        (Join-Path $automationDir "development-eval-$stamp.log")
}
if (-not (Test-Path -LiteralPath (Join-Path $treatmentDir 'benchmarks.json'))) {
    $benchmarks = @(
        'data/benchmarks/arc_easy.jsonl','data/benchmarks/arc_challenge.jsonl',
        'data/benchmarks/hellaswag.jsonl','data/benchmarks/belebele_en.jsonl',
        'data/benchmarks/belebele_fr.jsonl','data/benchmarks/winogrande.jsonl'
    )
    $mcArgs = @('-m','slm','eval','mc','--checkpoint',$checkpoint,
        '--tokenizer','data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json',
        '--batch-items','2','--device','cuda','--benchmarks') + $benchmarks + @(
        '--out',(Join-Path $treatmentDir 'benchmarks.json'))
    Invoke-LoggedPython $mcArgs (Join-Path $automationDir "benchmark-eval-$stamp.log")
}

Invoke-LoggedPython @('scripts/analyze_exp0030_25m_data_diversity.py',
    '--control',(Join-Path $controlDir 'development-bpb.json'),
    '--treatment',(Join-Path $treatmentDir 'development-bpb.json'),
    '--control-mc',(Join-Path $controlDir 'benchmarks.json'),
    '--treatment-mc',(Join-Path $treatmentDir 'benchmarks.json'),
    '--out',(Join-Path $evalDir 'paired-development-bpb.json')) `
    (Join-Path $automationDir "paired-analysis-$stamp.log")

"EXP-0031 10% HPLT mixture pilot completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $automationDir 'complete.txt') -Encoding utf8
