param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$automationDir = Join-Path $ProjectRoot 'runs/EXP-0032-automation'
$evalDir = Join-Path $ProjectRoot 'runs/EXP-0032-evaluations'
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
    @{ slug='fsc-control'; config='configs/train/exp0032-25m-fsc-control-s20261028.yaml';
       checkpoint='runs/EXP-0032-25m-fsc-control-s20261028/checkpoints/step_00001500.pt' },
    @{ slug='hplt-10pct-treatment'; config='configs/train/exp0032-25m-hplt-10pct-s20261028.yaml';
       checkpoint='runs/EXP-0032-25m-hplt-10pct-s20261028/checkpoints/step_00001500.pt' }
)

foreach ($arm in $arms) {
    if (-not (Test-Path -LiteralPath $arm.checkpoint)) {
        Invoke-LoggedPython @('-m','slm','train',$arm.config) `
            (Join-Path $automationDir "$($arm.slug)-train-$stamp.log")
    }
}

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl','data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl','data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl','data/benchmarks/winogrande.jsonl'
)
foreach ($arm in $arms) {
    $outDir = Join-Path $evalDir $arm.slug
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $devOut = Join-Path $outDir 'development-bpb.json'
    if (-not (Test-Path -LiteralPath $devOut)) {
        Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py','--checkpoint',$arm.checkpoint,
            '--out',$devOut) (Join-Path $automationDir "$($arm.slug)-dev-$stamp.log")
    }
    $mcOut = Join-Path $outDir 'benchmarks.json'
    if (-not (Test-Path -LiteralPath $mcOut)) {
        $mcArgs = @('-m','slm','eval','mc','--checkpoint',$arm.checkpoint,
            '--tokenizer','data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json',
            '--batch-items','2','--device','cuda','--benchmarks') + $benchmarks + @('--out',$mcOut)
        Invoke-LoggedPython $mcArgs (Join-Path $automationDir "$($arm.slug)-benchmarks-$stamp.log")
    }
}

Invoke-LoggedPython @('scripts/analyze_exp0030_25m_data_diversity.py',
    '--control',(Join-Path $evalDir 'fsc-control/development-bpb.json'),
    '--treatment',(Join-Path $evalDir 'hplt-10pct-treatment/development-bpb.json'),
    '--control-mc',(Join-Path $evalDir 'fsc-control/benchmarks.json'),
    '--treatment-mc',(Join-Path $evalDir 'hplt-10pct-treatment/benchmarks.json'),
    '--kind','slm.exp0032_fresh_seed_development_analysis',
    '--out',(Join-Path $evalDir 'paired-development-bpb.json')) `
    (Join-Path $automationDir "paired-development-$stamp.log")

# The recipe is frozen. Score the complete, previously sealed HPLT test split once for each arm.
foreach ($arm in $arms) {
    $outPath = Join-Path $evalDir (Join-Path $arm.slug 'hplt-test.json')
    if (-not (Test-Path -LiteralPath $outPath)) {
        Invoke-LoggedPython @('scripts/evaluate_exp0030_25m_dev.py','--checkpoint',$arm.checkpoint,
            '--hplt-path','data/processed/slm-tavenel-growth-candidate-v2/test.jsonl.gz',
            '--hplt-split','test','--hplt-only','--out',$outPath) `
            (Join-Path $automationDir "$($arm.slug)-sealed-hplt-test-$stamp.log")
    }
}

Invoke-LoggedPython @('scripts/analyze_exp0032_hplt_test.py',
    '--control',(Join-Path $evalDir 'fsc-control/hplt-test.json'),
    '--treatment',(Join-Path $evalDir 'hplt-10pct-treatment/hplt-test.json'),
    '--out',(Join-Path $evalDir 'paired-hplt-test.json')) `
    (Join-Path $automationDir "paired-hplt-test-$stamp.log")

"EXP-0032 fresh-seed replication completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $automationDir 'complete.txt') -Encoding utf8
