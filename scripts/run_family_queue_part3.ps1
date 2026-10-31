param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# Peer-family queue, part 3: 362M (EXP-0034b), 550M (EXP-0037) and 1B (EXP-0035), all Muon with
# master-weight offload on edu corpus v2. Each model: one-update memory/throughput pilot, training,
# six-task benchmarks and held-out BPB on a systematic 1-in-5 subsample of the v2 test split.
# Resumable (stages skip finished outputs, training resumes from checkpoints); serial GPU use.

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
    Write-Status "part 3 waiting for process $WaitForProcessId"
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "part 3 started (pid $PID)"
$models = @(
    @{ slug = '360m'; config = 'configs/train/exp0034b-360m-muon.yaml'; run = 'runs/EXP-0034b-360m-muon'; steps = 8000 },
    @{ slug = '550m'; config = 'configs/train/exp0037-550m-muon.yaml'; run = 'runs/EXP-0037-550m-muon'; steps = 6100 },
    @{ slug = '1b'; config = 'configs/train/exp0035-1b-muon.yaml'; run = 'runs/EXP-0035-1b-muon'; steps = 4600 }
)
foreach ($m in $models) {
    $pilot = Join-Path $logDir "$($m.slug)-muon-pilot.json"
    if (-not (Test-Path -LiteralPath $pilot)) {
        Invoke-Python "$($m.slug)-muon-pilot" @('-m', 'slm', 'bench', $m.config, '--steps', '2', '--warmup', '1')
        Get-Content -LiteralPath (Join-Path $logDir "$($m.slug)-muon-pilot.log") | Set-Content -LiteralPath $pilot -Encoding utf8
    }
    $ckpt = "$($m.run)/checkpoints/step_{0:D8}.pt" -f $m.steps
    if (-not (Test-Path -LiteralPath $ckpt)) {
        Invoke-Python "$($m.slug)-train" @('-m', 'slm', 'train', $m.config, '--set', 'resume=true')
    }
    $outDir = "runs/FAMILY-evaluations/$($m.slug)-muon"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
        Invoke-Python "$($m.slug)-benchmarks" (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
            '--batch-items', '2', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
    }
    if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
        Invoke-Python "$($m.slug)-edu-test-bpb" @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
            '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v2', '--experiment-id', 'FAMILY',
            '--every', '5', '--forward-batch', '2', '--out', "$outDir/edu-test-bpb.json")
    }
}
Write-Status 'family queue part 3 complete'
