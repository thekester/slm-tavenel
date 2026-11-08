param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0043: good-view JEPA on English/French translation pairs vs the same pairs through next-token loss
# (75M proxy, 1,500 updates each), then benchmarks, held-out corpus-v1 BPB, paired comparisons against the
# control and the EXP-0036a reference, and EN/FR alignment on Belebele (CPU). Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0043-automation'
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
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
Write-Status "EXP-0043 queue started (pid $PID)"

# Amendment 2026-11-08: the v1 treatment (loss variant "cov") collapsed onto z = 0 and is archived as
# runs/EXP-0043-75m-goodview-v1-collapsed; the treatment is rerun as v2 (VICReg-style anti-collapse).
$arms = [ordered]@{ 'parallel-ntp' = 'configs/train/exp0043-75m-parallel-ntp.yaml'; 'goodview-v2' = 'configs/train/exp0043-75m-goodview-v2.yaml' }
foreach ($arm in $arms.Keys) {
    $ckpt = "runs/EXP-0043-75m-$arm/checkpoints/step_00001500.pt"
    if (-not (Test-Path -LiteralPath $ckpt)) {
        Invoke-Python "exp0043-$arm-train" @('-m', 'slm', 'train', $arms[$arm], '--set', 'resume=true')
    }
    $outDir = "runs/EXP-0043-evaluations/$arm"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    if (-not (Test-Path -LiteralPath "$outDir/benchmarks.json")) {
        Invoke-Python "exp0043-$arm-benchmarks" (@('-m', 'slm', 'eval', 'mc', '--checkpoint', $ckpt, '--tokenizer', $tokenizer,
            '--batch-items', '4', '--device', 'cuda', '--benchmarks') + $benchmarks + @('--out', "$outDir/benchmarks.json"))
    }
    if (-not (Test-Path -LiteralPath "$outDir/edu-test-bpb.json")) {
        Invoke-Python "exp0043-$arm-edu-test-bpb" @('scripts/evaluate_exp0028_hplt_test.py', '--checkpoint', $ckpt,
            '--tokenizer', $tokenizer, '--corpus', 'data/processed/slm-tavenel-edu-corpus-v1', '--experiment-id', 'EXP-0043',
            '--forward-batch', '4', '--out', "$outDir/edu-test-bpb.json")
    }
}
$reference = 'runs/EXP-0036-evaluations/75m-muon'
foreach ($pair in @(@('control-vs-goodview-v2', 'runs/EXP-0043-evaluations/parallel-ntp', 'runs/EXP-0043-evaluations/goodview-v2', 'parallel_ntp,goodview'),
                    @('reference-vs-goodview-v2', $reference, 'runs/EXP-0043-evaluations/goodview-v2', 'ntp_only,goodview'),
                    @('reference-vs-parallel-ntp', $reference, 'runs/EXP-0043-evaluations/parallel-ntp', 'ntp_only,parallel_ntp'))) {
    $out = "runs/EXP-0043-evaluations/$($pair[0]).json"
    if (-not (Test-Path -LiteralPath $out)) {
        Invoke-Python "exp0043-$($pair[0])" @('scripts/analyze_scale_comparison.py', '--experiment-id', 'EXP-0043',
            '--labels', $pair[3], '--test-name', 'edu_test',
            '--smaller-mc', "$($pair[1])/benchmarks.json", '--larger-mc', "$($pair[2])/benchmarks.json",
            '--smaller-test', "$($pair[1])/edu-test-bpb.json", '--larger-test', "$($pair[2])/edu-test-bpb.json",
            '--out', $out, '--limitations', 'single seed per arm; untuned auxiliary weight 0.1')
    }
}
if (-not (Test-Path -LiteralPath 'runs/EXP-0043-evaluations/belebele-alignment.json')) {
    $env:CUDA_VISIBLE_DEVICES = '-1'
    Invoke-Python 'exp0043-alignment' @('-u', 'scripts/platonic_alignment.py', '--no-random-control',
        '--checkpoint', 'ntp_only=runs/EXP-0036a-75m-muon/checkpoints/step_00001500.pt',
        '--checkpoint', 'parallel_ntp=runs/EXP-0043-75m-parallel-ntp/checkpoints/step_00001500.pt',
        '--checkpoint', 'goodview=runs/EXP-0043-75m-goodview-v2/checkpoints/step_00001500.pt',
        '--checkpoint', 'goodview_v1_collapsed=runs/EXP-0043-75m-goodview-v1-collapsed/checkpoints/step_00001500.pt',
        '--compare', 'parallel_ntp=goodview', '--compare', 'ntp_only=goodview', '--compare', 'ntp_only=parallel_ntp',
        '--out', 'runs/EXP-0043-evaluations/belebele-alignment.json')
    Remove-Item Env:CUDA_VISIBLE_DEVICES
}
Write-Status 'EXP-0043 complete'
