$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = 'C:\Program Files\Python311\python.exe'
$controlConfig = Join-Path $workspacePath 'configs/train/exp0012-10m-fsc-control-py311.yaml'
$mixConfig = Join-Path $workspacePath 'configs/train/exp0012-10m-fsc-mot-mixture-py311.yaml'
$controlRun = Join-Path $workspacePath 'runs/EXP-0012-10m-fsc-control-py311-torch210-2026-10-22'
$mixRun = Join-Path $workspacePath 'runs/EXP-0012-10m-fsc-mot-mixture-py311-torch210-2026-10-22'
$queueLog = Join-Path $workspacePath 'runs/EXP-0012-py311-queue.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $queueLog) -Force | Out-Null
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The recorded Python runtime is unavailable: $pythonPath"
}

function Write-QueueLog([string]$message) {
    $stamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz')
    Add-Content -LiteralPath $queueLog -Value "$stamp $message" -Encoding utf8
}

Set-Location -LiteralPath $workspacePath
$env:PYTHONPATH = Join-Path $workspacePath 'src'
$env:PYTHONUTF8 = '1'
Write-QueueLog 'Starting matched FSC-only control with Python 3.11 / PyTorch 2.10.'
New-Item -ItemType Directory -Path $controlRun -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train $controlConfig *> (Join-Path $controlRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Control failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'FSC control finished; starting matched 20% MOT mixture arm.'
New-Item -ItemType Directory -Path $mixRun -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train $mixConfig *> (Join-Path $mixRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Mixture arm failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Both training arms finished; evaluating control on frozen held-out sets.'
$ErrorActionPreference = 'Continue'
& $pythonPath (Join-Path $workspacePath 'scripts/evaluate_exp0012_checkpoint.py') --checkpoint (Join-Path $controlRun 'checkpoints/step_00015000.pt') --out (Join-Path $controlRun 'analysis/heldout-evaluation.json') *> (Join-Path $controlRun 'evaluation-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Control evaluation failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Control evaluation finished; evaluating MOT mixture arm.'
$ErrorActionPreference = 'Continue'
& $pythonPath (Join-Path $workspacePath 'scripts/evaluate_exp0012_checkpoint.py') --checkpoint (Join-Path $mixRun 'checkpoints/step_00015000.pt') --out (Join-Path $mixRun 'analysis/heldout-evaluation.json') *> (Join-Path $mixRun 'evaluation-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Mixture evaluation failed with exit code $exitCode."; exit $exitCode }
Write-QueueLog 'Both EXP-0012 held-out evaluations completed.'
