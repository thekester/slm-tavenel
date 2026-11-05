param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0036b: general-web continuation of the GPT-2-matched 124M. Builds and encodes the general-web
# corpus (CPU, may run while another queue trains), then waits for the previous GPU queue before
# training and evaluating (six pinned tasks, paired against GPT-2 and EXP-0036; full edu v2 test BPB).
# Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0036b-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'
$tokenizer = 'data/tokenizers/smollm2-f8027fd0/tokenizer.json'
$corpus = 'data/processed/slm-tavenel-general-web-v1'
$tokens = 'data/tokens/slm-tavenel-general-web-v1-smollm2'
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

Write-Status "EXP-0036b queue started (pid $PID)"
if (-not (Test-Path -LiteralPath "$corpus/manifest.json")) {
    Invoke-Python 'general-web-build' @('scripts/build_edu_corpus_v1.py', '--en-source', 'fineweb', '--en-files', '2',
        '--fr-files', '1', '--workers', '6', '--schema-version', 'general-web-v1', '--out', $corpus,
        '--purpose', 'EXP-0036b general-web continuation of the GPT-2-matched 124M (EN FineWeb + FR replay)')
}
if (-not (Test-Path -LiteralPath "$tokens/manifest.json")) {
    Invoke-Python 'general-web-encode' @('scripts/encode_edu_corpus_v1.py', '--corpus', $corpus, '--out', $tokens)
}

if ($WaitForProcessId -gt 0) {
    Write-Status "waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}

$ckpt = 'runs/EXP-0036b-124m-general-web/checkpoints/step_00015000.pt'
if (-not (Test-Path -LiteralPath $ckpt)) {
    Invoke-Python 'exp0036b-train' @('-m', 'slm', 'train', 'configs/train/exp0036b-124m-general-web-continuation.yaml',
        '--set', 'resume=true')
}
$outDir = 'runs/EXP-0036-evaluations/124m-general-web-continuation'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
    Invoke-Python 'exp0036b-benchmarks' (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
        '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
}
if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
    Invoke-Python 'exp0036b-edu-test-bpb' @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
        '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v2', '--experiment-id', 'EXP-0036b',
        '--forward-batch', '4', '--out', "$outDir/edu-test-bpb.json")
}
if (-not (Test-Path -LiteralPath 'runs/EXP-0036-evaluations/exp0036b-vs-gpt2-paired.json')) {
    Invoke-Python 'exp0036b-vs-gpt2' @('scripts/compare_mc_paired.py',
        '--control', 'runs/reference/peer-models-2026-10/gpt-2/mc.json', '--treatment', "$outDir/benchmarks.json",
        '--labels', 'gpt2,slm124m_general', '--out', 'runs/EXP-0036-evaluations/exp0036b-vs-gpt2-paired.json')
    Invoke-Python 'exp0036b-vs-exp0036' @('scripts/compare_mc_paired.py',
        '--control', 'runs/EXP-0036-evaluations/124m-gpt2match-muon/benchmarks.json', '--treatment', "$outDir/benchmarks.json",
        '--labels', 'slm124m_edu,slm124m_general', '--out', 'runs/EXP-0036-evaluations/exp0036b-vs-exp0036-paired.json')
}
Write-Status 'EXP-0036b complete'
