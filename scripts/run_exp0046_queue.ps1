param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0046: 124M text JEPA on COCO caption pairs vs next-token control (GPU), then text-image alignment
# against DINOv2 on COCO val2017 (CPU, with peers). Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0046-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir 'queue-status.log'

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
Write-Status "EXP-0046 queue started (pid $PID)"

foreach ($mode in 'jepa', 'ntp') {
    $out = "runs/EXP-0046-124m-$mode"
    if (-not (Test-Path -LiteralPath "$out/checkpoints/step_00009000.pt")) {
        Invoke-Python "exp0046-$mode-train" @('scripts/train_text_jepa.py', '--mode', $mode, '--out', $out)
    }
}

if (-not (Test-Path -LiteralPath 'runs/EXP-0046-evaluations/text-image-alignment.json')) {
    $peers = (Get-Content 'configs/eval/peer-models-2026-10.json' -Raw -Encoding UTF8 | ConvertFrom-Json)
    if ($peers.models) { $peers = $peers.models }
    $hf = @()
    foreach ($name in 'GPT-2', 'Pythia-160M', 'SmolLM2-360M', 'Qwen2.5-0.5B', 'Qwen3-0.6B', 'OLMo-2-1B', 'SmolLM2-1.7B') {
        $peer = $peers | Where-Object { $_.name -eq $name }
        if (-not $peer) { Write-Status "FAILED peer $name not in config"; exit 1 }
        $hf += @('--hf', "$name=$($peer.repo)@$($peer.revision)")
    }
    $env:CUDA_VISIBLE_DEVICES = '-1'
    Invoke-Python 'exp0046-alignment' (@('-u', 'scripts/evaluate_text_image_alignment.py',
        '--native', 'J=runs/EXP-0046-124m-jepa/checkpoints/step_00009000.pt',
        '--native', 'N=runs/EXP-0046-124m-ntp/checkpoints/step_00009000.pt',
        '--native', 'web-124M=runs/EXP-0036-124m-gpt2match-muon/checkpoints/step_00015000.pt',
        '--random-control', 'random-124M=runs/EXP-0046-124m-jepa/checkpoints/step_00009000.pt') + $hf +
        @('--compare', 'N=J', '--compare', 'web-124M=J', '--compare', 'GPT-2=J', '--compare', 'Pythia-160M=J',
          '--compare', 'SmolLM2-360M=J', '--compare', 'Qwen2.5-0.5B=J', '--compare', 'Qwen3-0.6B=J',
          '--compare', 'OLMo-2-1B=J', '--compare', 'SmolLM2-1.7B=J',
          '--out', 'runs/EXP-0046-evaluations/text-image-alignment.json'))
    Remove-Item Env:CUDA_VISIBLE_DEVICES
}
Write-Status 'EXP-0046 complete'
