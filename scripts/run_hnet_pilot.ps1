param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0040 H-Net pilot: byte corpus check, memory/throughput pilot (falls back to micro-batch 1 with the
# same bytes per update if micro-batch 2 fails), training, benchmarks, held-out BPB, paired comparison
# with the EXP-0036a BPE Muon arm. Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0040-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'
$config = 'configs/train/exp0040-hnet-75m-muon.yaml'
$tokenizer = 'data/tokenizers/bytes-v1/tokenizer.json'
$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)

function Write-Status([string]$message) {
    "$(Get-Date -Format o) $message" | Add-Content -LiteralPath $status -Encoding utf8
}

function Invoke-Python([string]$name, [string[]]$arguments, [switch]$AllowFailure) {
    $log = Join-Path $logDir ($name + '.log')
    Write-Status "start $name"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @arguments *>> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) {
        Write-Status "FAILED $name (exit $code); see $log"
        if (-not $AllowFailure) { exit $code }
        return $false
    }
    Write-Status "done $name"
    return $true
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "queue started (pid $PID)"
if (-not (Test-Path -LiteralPath 'data/tokens/slm-tavenel-edu-corpus-v1-bytes/manifest.json')) {
    Invoke-Python 'encode-bytes-v1' @('scripts/encode_bytes_corpus.py') | Out-Null
}

$overrides = @('resume=true')
$fallback = Join-Path $logDir 'use-microbatch-1.txt'
if (-not (Test-Path -LiteralPath (Join-Path $logDir 'pilot-ok.txt')) -and -not (Test-Path -LiteralPath $fallback)) {
    $ok = Invoke-Python 'hnet-pilot-mb2' @('-m', 'slm', 'bench', $config, '--steps', '2', '--warmup', '1',
        '--set', 'grad_accum_steps=2') -AllowFailure
    if ($ok) { 'micro-batch 2 fits' | Set-Content -LiteralPath (Join-Path $logDir 'pilot-ok.txt') -Encoding utf8 }
    else { 'micro-batch 2 failed; using micro-batch 1 x 36 (same bytes per update)' | Set-Content -LiteralPath $fallback -Encoding utf8 }
}
if (Test-Path -LiteralPath $fallback) { $overrides += @('micro_batch_size=1', 'grad_accum_steps=36') }

$ckpt = 'runs/EXP-0040-hnet-75m-muon/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $ckpt)) {
    Invoke-Python 'hnet-train' (@('-m', 'slm', 'train', $config, '--set') + $overrides) | Out-Null
}
$outDir = 'runs/EXP-0040-evaluations/hnet-75m-muon'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
    Invoke-Python 'hnet-benchmarks' (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
        '--batch-items', '2', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json")) | Out-Null
}
if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
    Invoke-Python 'hnet-edu-test-bpb' @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
        '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v1', '--experiment-id', 'EXP-0040',
        '--forward-batch', '2', '--out', "$outDir/edu-test-bpb.json") | Out-Null
}
if (-not (Test-Path -LiteralPath 'runs/EXP-0040-evaluations/paired-comparison.json')) {
    Invoke-Python 'hnet-analysis' @('scripts/analyze_scale_comparison.py', '--experiment-id', 'EXP-0040',
        '--labels', 'bpe,hnet', '--test-name', 'edu_test',
        '--smaller-mc', 'runs/EXP-0036-evaluations/75m-muon/benchmarks.json', '--larger-mc', "$outDir/benchmarks.json",
        '--smaller-test', 'runs/EXP-0036-evaluations/75m-muon/edu-test-bpb.json', '--larger-test', "$outDir/edu-test-bpb.json",
        '--out', 'runs/EXP-0040-evaluations/paired-comparison.json',
        '--limitations', 'single seed, 75M scale; H-Net gets about 4-7% more nominal FLOPs per byte',
        'sliding-window attention replaces Mamba-2 in the byte encoder/decoder; no per-stage learning-rate modulation') | Out-Null
}
Write-Status 'EXP-0040 H-Net pilot complete'
