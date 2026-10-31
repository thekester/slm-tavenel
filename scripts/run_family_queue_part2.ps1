param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# Peer-family queue, part 2: EXP-0036 GPT-2-matched 124M (Muon) training and evaluation on the
# six pinned tasks and the edu corpus v2 held-out test split. Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/FAMILY-automation'
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
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
Write-Status "part 2 started (pid $PID)"

$ckpt = 'runs/EXP-0036-124m-gpt2match-muon/checkpoints/step_00015000.pt'
if (-not (Test-Path -LiteralPath $ckpt)) {
    Invoke-Python 'exp0036-124m-train' @('-m', 'slm', 'train', 'configs/train/exp0036-124m-gpt2match-muon.yaml',
        '--set', 'resume=true')
}
$outDir = 'runs/EXP-0036-evaluations/124m-gpt2match-muon'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
    Invoke-Python 'exp0036-124m-benchmarks' (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
        '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
}
if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
    Invoke-Python 'exp0036-124m-edu-test-bpb' @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
        '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v2', '--experiment-id', 'EXP-0036',
        '--forward-batch', '4', '--out', "$outDir/edu-test-bpb.json")
}
Write-Status 'family queue part 2 complete'
