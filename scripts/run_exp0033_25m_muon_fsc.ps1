param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$automationDir = Join-Path $ProjectRoot 'runs/EXP-0033-automation'
$evalDir = Join-Path $ProjectRoot 'runs/EXP-0033-evaluations'
$referenceDir = Join-Path $ProjectRoot 'runs/EXP-0032-evaluations/fsc-control'
New-Item -ItemType Directory -Force -Path $automationDir,$evalDir | Out-Null
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

if (-not (Test-Path -LiteralPath (Join-Path $referenceDir 'development-bpb.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $referenceDir 'benchmarks.json'))) {
    throw 'EXP-0032 matched AdamW control evaluations must finish before EXP-0033 starts.'
}

$checkpoint = 'runs/EXP-0033-25m-muon-fsc-s20261028/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $checkpoint)) {
    Invoke-LoggedPython @('-m','slm','train','configs/train/exp0033-25m-muon-fsc-s20261028.yaml') `
        (Join-Path $automationDir "train-$stamp.log")
}

$evalOut = Join-Path $evalDir 'development-bpb.json'
if (-not (Test-Path -LiteralPath $evalOut)) {
    Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py','--checkpoint',$checkpoint,
        '--out',$evalOut) (Join-Path $automationDir "development-eval-$stamp.log")
}

$mcOut = Join-Path $evalDir 'benchmarks.json'
if (-not (Test-Path -LiteralPath $mcOut)) {
    $benchmarks = @(
        'data/benchmarks/arc_easy.jsonl','data/benchmarks/arc_challenge.jsonl',
        'data/benchmarks/hellaswag.jsonl','data/benchmarks/belebele_en.jsonl',
        'data/benchmarks/belebele_fr.jsonl','data/benchmarks/winogrande.jsonl'
    )
    $mcArgs = @('-m','slm','eval','mc','--checkpoint',$checkpoint,
        '--tokenizer','data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json',
        '--batch-items','2','--device','cuda','--benchmarks') + $benchmarks + @('--out',$mcOut)
    Invoke-LoggedPython $mcArgs (Join-Path $automationDir "benchmarks-$stamp.log")
}

Invoke-LoggedPython @('scripts/analyze_exp0030_25m_data_diversity.py',
    '--control',(Join-Path $referenceDir 'development-bpb.json'),
    '--treatment',$evalOut,
    '--control-mc',(Join-Path $referenceDir 'benchmarks.json'),
    '--treatment-mc',$mcOut,
    '--kind','slm.exp0033_25m_muon_development_analysis',
    '--out',(Join-Path $evalDir 'paired-development.json')) `
    (Join-Path $automationDir "paired-analysis-$stamp.log")

"EXP-0033 Muon development comparison completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $automationDir 'complete.txt') -Encoding utf8
