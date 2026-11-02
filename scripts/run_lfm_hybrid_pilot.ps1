param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0041 LFM-style hybrid pilot: training, benchmarks, held-out BPB, paired comparison with the
# EXP-0036a all-attention Muon arm. Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0041-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'
$tokenizer = 'data/tokenizers/smollm2-f8027fd0/tokenizer.json'
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
    if ($code -ne 0) { Write-Status "FAILED $name (exit $code); see $log"; exit $code }
    Write-Status "done $name"
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "queue started (pid $PID)"
$ckpt = 'runs/EXP-0041-75m-lfm2-hybrid-muon/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $ckpt)) {
    Invoke-Python 'hybrid-train' @('-m', 'slm', 'train', 'configs/train/exp0041-75m-lfm2-hybrid-muon.yaml', '--set', 'resume=true')
}
$outDir = 'runs/EXP-0041-evaluations/75m-lfm2-hybrid-muon'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
    Invoke-Python 'hybrid-benchmarks' (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
        '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
}
if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
    Invoke-Python 'hybrid-edu-test-bpb' @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
        '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v1', '--experiment-id', 'EXP-0041',
        '--forward-batch', '4', '--out', "$outDir/edu-test-bpb.json")
}
if (-not (Test-Path -LiteralPath 'runs/EXP-0041-evaluations/paired-comparison.json')) {
    Invoke-Python 'hybrid-analysis' @('scripts/analyze_scale_comparison.py', '--experiment-id', 'EXP-0041',
        '--labels', 'attention,hybrid', '--test-name', 'edu_test',
        '--smaller-mc', 'runs/EXP-0036-evaluations/75m-muon/benchmarks.json', '--larger-mc', "$outDir/benchmarks.json",
        '--smaller-test', 'runs/EXP-0036-evaluations/75m-muon/edu-test-bpb.json', '--larger-test', "$outDir/edu-test-bpb.json",
        '--out', 'runs/EXP-0041-evaluations/paired-comparison.json',
        '--limitations', 'single seed, 75M scale; the hybrid has 5.1% more non-embedding parameters',
        'LFM2-350M layout copied, not searched; GPU throughput only, no CPU/edge latency')
}
Write-Status 'EXP-0041 LFM hybrid pilot complete'
