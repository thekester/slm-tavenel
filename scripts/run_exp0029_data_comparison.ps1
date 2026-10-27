param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$runDir = Join-Path $ProjectRoot 'runs/EXP-0029-automation'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$tokenizer = 'data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json'
$tokens = 'data/tokens/slm-tavenel-growth-candidate-v2-fsc-tokenizer'
$expectedTokenizerHash = '558bab3421bc9ee280f2aa9d0c6a82a9bc482e2ecb72a75837e9c3ee7d2dbd39'
$tokenManifest = Join-Path $ProjectRoot (Join-Path $tokens 'manifest.json')

$needEncoding = $true
if (Test-Path -LiteralPath $tokenManifest) {
    $manifest = Get-Content -LiteralPath $tokenManifest -Raw | ConvertFrom-Json
    if ($manifest.tokenizer_sha256 -eq $expectedTokenizerHash) { $needEncoding = $false }
}
if ($needEncoding) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python scripts/encode_growth_candidate_v2_fsc_tokenizer.py `
        *> (Join-Path $runDir 'hplt-reencoding.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }
}

$arms = @(
    @{ slug='old-corpus-control'; config='configs/train/exp0029-100m-old-corpus-control.yaml';
       checkpoint='runs/EXP-0029-100m-old-corpus-control-s20261026/checkpoints/step_00001500.pt' },
    @{ slug='hplt-rich-treatment'; config='configs/train/exp0029-100m-hplt-treatment.yaml';
       checkpoint='runs/EXP-0029-100m-hplt-treatment-s20261026/checkpoints/step_00001500.pt' }
)
foreach ($arm in $arms) {
    if (Test-Path -LiteralPath $arm.checkpoint) { continue }
    $trainLog = Join-Path $runDir ($arm.slug + '-train.log')
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -m slm train $arm.config *> $trainLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }
}

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)
foreach ($arm in $arms) {
    $outDir = Join-Path $ProjectRoot (Join-Path 'runs/EXP-0029-evaluations' $arm.slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -m slm eval mc --checkpoint $arm.checkpoint --tokenizer $tokenizer `
        --batch-items 2 --device cuda --benchmarks $benchmarks --out (Join-Path $outDir 'benchmarks.json') `
        *> (Join-Path $outDir 'benchmarks.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python scripts/evaluate_exp0028_hplt_test.py --checkpoint $arm.checkpoint --tokenizer $tokenizer `
        --experiment-id EXP-0029 `
        --out (Join-Path $outDir 'hplt-test-bpb.json') *> (Join-Path $outDir 'hplt-test-bpb.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { exit $exitCode }
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python scripts/analyze_exp0029_data_comparison.py `
    --control-mc runs/EXP-0029-evaluations/old-corpus-control/benchmarks.json `
    --hplt-mc runs/EXP-0029-evaluations/hplt-rich-treatment/benchmarks.json `
    --control-test runs/EXP-0029-evaluations/old-corpus-control/hplt-test-bpb.json `
    --hplt-test runs/EXP-0029-evaluations/hplt-rich-treatment/hplt-test-bpb.json `
    --out runs/EXP-0029-evaluations/paired-comparison.json `
    *> (Join-Path $runDir 'paired-analysis.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($exitCode -ne 0) { exit $exitCode }

"EXP-0029 completed at $(Get-Date -Format o)" |
    Set-Content -LiteralPath (Join-Path $runDir 'complete.txt') -Encoding utf8
