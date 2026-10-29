param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0034a queue: edu corpus encoding -> 75M CE and distillation arms -> evaluations -> paired analysis.
# Every stage is skipped when its output exists and training resumes from the latest checkpoint, so the
# script can be relaunched as is after a shutdown. GPU work runs strictly one job at a time.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0034-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'
$tokenizer = 'data/tokenizers/smollm2-f8027fd0/tokenizer.json'
$corpus = 'data/processed/slm-tavenel-edu-corpus-v1'
$tokens = 'data/tokens/slm-tavenel-edu-corpus-v1-smollm2'
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

function Invoke-Pilot([string]$name, [string]$config) {
    $out = Join-Path $logDir ($name + '.json')
    if (Test-Path -LiteralPath $out) { Write-Status "skip $name"; return }
    Invoke-Python $name @('-m', 'slm', 'bench', $config, '--steps', '3', '--warmup', '1')
    Get-Content -LiteralPath (Join-Path $logDir ($name + '.log')) | Set-Content -LiteralPath $out -Encoding utf8
}

function Invoke-Evaluation([string]$slug, [string]$checkpoint) {
    $outDir = Join-Path $ProjectRoot (Join-Path 'runs/EXP-0034-evaluations' $slug)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $mc = Join-Path $outDir 'benchmarks.json'
    if (-not (Test-Path -LiteralPath $mc)) {
        Invoke-Python "$slug-benchmarks" (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $checkpoint,
            '--tokenizer', $tokenizer, '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks +
            @('--out', $mc))
    }
    $bpb = Join-Path $outDir 'edu-test-bpb.json'
    if (-not (Test-Path -LiteralPath $bpb)) {
        Invoke-Python "$slug-edu-test-bpb" @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $checkpoint,
            '--tokenizer', $tokenizer, '--corpus', $corpus, '--experiment-id', 'EXP-0034', '--forward-batch', '4',
            '--out', $bpb)
    }
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId (corpus build)"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "queue started (pid $PID)"
if (-not (Test-Path -LiteralPath "$corpus/manifest.json")) {
    Write-Status 'FAILED corpus manifest missing; rerun scripts/build_edu_corpus_v1.py'
    exit 2
}
if (-not (Test-Path -LiteralPath "$tokens/manifest.json")) {
    Invoke-Python 'encode-edu-corpus' @('scripts/encode_edu_corpus_v1.py')
}

Invoke-Pilot 'exp0034a-75m-ce-pilot' 'configs/train/exp0034a-75m-ce.yaml'
Invoke-Pilot 'exp0034a-75m-distill-pilot' 'configs/train/exp0034a-75m-distill.yaml'

$ce = 'runs/EXP-0034a-75m-ce/checkpoints/step_00001500.pt'
$kd = 'runs/EXP-0034a-75m-distill/checkpoints/step_00001500.pt'
Invoke-Train 'exp0034a-75m-ce-train' 'configs/train/exp0034a-75m-ce.yaml' $ce
Invoke-Evaluation '75m-ce' $ce
Invoke-Train 'exp0034a-75m-distill-train' 'configs/train/exp0034a-75m-distill.yaml' $kd
Invoke-Evaluation '75m-distill' $kd

$analysis = 'runs/EXP-0034-evaluations/exp0034a-paired-comparison.json'
if (-not (Test-Path -LiteralPath $analysis)) {
    Invoke-Python 'exp0034a-analysis' @('scripts/analyze_scale_comparison.py', '--experiment-id', 'EXP-0034a',
        '--labels', 'ce,distill', '--test-name', 'edu_test',
        '--smaller-mc', 'runs/EXP-0034-evaluations/75m-ce/benchmarks.json',
        '--larger-mc', 'runs/EXP-0034-evaluations/75m-distill/benchmarks.json',
        '--smaller-test', 'runs/EXP-0034-evaluations/75m-ce/edu-test-bpb.json',
        '--larger-test', 'runs/EXP-0034-evaluations/75m-distill/edu-test-bpb.json',
        '--out', $analysis,
        '--limitations', 'single-seed token-matched ablation; the distillation arm also spends teacher compute',
        'teacher SmolLM2-360M may carry benchmark exposure from its own pretraining data',
        'web sources: page-level rights not verified')
}
Write-Status 'EXP-0034a queue complete'
