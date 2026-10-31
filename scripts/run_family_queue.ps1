param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# Peer-family queue, part 1 (PLAN-peer-family-2026-10-30): public peer evaluations, then the EXP-0036a
# Muon arm on the 75M proxy, its evaluation and the paired comparison with the EXP-0034a CE arm.
# Resumable: finished stages are skipped; training resumes from the latest checkpoint. Serial GPU use.

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
Write-Status "queue started (pid $PID)"

# 1. Public peers at pinned revisions (the script skips models already scored).
$previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
Write-Status 'start peer-manifest-evaluation'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_peer_manifest.ps1 -ProjectRoot $ProjectRoot
$ErrorActionPreference = $previous
Write-Status 'done peer-manifest-evaluation'

# 2. EXP-0036a: Muon arm on the 75M proxy (control: EXP-0034a CE arm).
$muon = 'runs/EXP-0036a-75m-muon/checkpoints/step_00001500.pt'
if (-not (Test-Path -LiteralPath $muon)) {
    Invoke-Python 'exp0036a-75m-muon-train' @('-m', 'slm', 'train', 'configs/train/exp0036a-75m-muon.yaml', '--set', 'resume=true')
}
$outDir = 'runs/EXP-0036-evaluations/75m-muon'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
    Invoke-Python 'exp0036a-75m-muon-benchmarks' (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $muon, '--tokenizer', $tokenizer,
        '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
}
if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
    Invoke-Python 'exp0036a-75m-muon-edu-test-bpb' @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $muon,
        '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v1', '--experiment-id', 'EXP-0036a',
        '--forward-batch', '4', '--out', "$outDir/edu-test-bpb.json")
}
if (-not (Test-Path -LiteralPath 'runs/EXP-0036-evaluations/exp0036a-paired-comparison.json')) {
    Invoke-Python 'exp0036a-analysis' @('scripts/analyze_scale_comparison.py', '--experiment-id', 'EXP-0036a',
        '--labels', 'adamw,muon', '--test-name', 'edu_test',
        '--smaller-mc', 'runs/EXP-0034-evaluations/75m-ce/benchmarks.json', '--larger-mc', "$outDir/benchmarks.json",
        '--smaller-test', 'runs/EXP-0034-evaluations/75m-ce/edu-test-bpb.json', '--larger-test', "$outDir/edu-test-bpb.json",
        '--out', 'runs/EXP-0036-evaluations/exp0036a-paired-comparison.json',
        '--limitations', 'single-seed token-matched optimizer ablation; the AdamW control was resumed once from step 250',
        'web sources: page-level rights not verified')
}
Write-Status 'family queue part 1 complete'
