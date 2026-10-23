$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = 'C:\Program Files\Python311\python.exe'
$controlRun = Join-Path $workspacePath 'runs/EXP-0020-25m-fsc-control-s2-2026-10-23'
$mirRun = Join-Path $workspacePath 'runs/EXP-0021-25m-fsc-mir-s2-2026-10-23'
$queueLog = Join-Path $workspacePath 'runs/EXP-0020-0021-mir-queue.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $queueLog) -Force | Out-Null
Set-Location -LiteralPath $workspacePath
$env:PYTHONPATH = Join-Path $workspacePath 'src'
$env:PYTHONUTF8 = '1'

function Write-QueueLog([string]$message) {
    $stamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz')
    Add-Content -LiteralPath $queueLog -Value "$stamp $message" -Encoding utf8
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Python unavailable: $pythonPath" }
if (Test-Path -LiteralPath $controlRun) { throw "Refusing to overwrite existing EXP-0020 output: $controlRun" }
if (Test-Path -LiteralPath $mirRun) { throw "Refusing to overwrite existing EXP-0021 output: $mirRun" }

Write-QueueLog 'Starting matched FSC-only seed-2 control.'
New-Item -ItemType Directory -Path $controlRun | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train 'configs/train/exp0020-25m-fsc-control-seed2.yaml' *> (Join-Path $controlRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0020 failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'EXP-0020 completed; starting matched FSC-only MIR run.'
New-Item -ItemType Directory -Path $mirRun | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train 'configs/train/exp0021-25m-fsc-mir-seed2.yaml' *> (Join-Path $mirRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0021 failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Both training runs completed; evaluating the control held-out text.'
$ErrorActionPreference = 'Continue'
& $pythonPath 'scripts/evaluate_exp0016_25m_wiki.py' --checkpoint (Join-Path $controlRun 'checkpoints/step_00001500.pt') --out (Join-Path $controlRun 'evaluation-primary-text.json') --primary-text-only --text-only --experiment-id EXP-0020 *> (Join-Path $controlRun 'evaluation-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0020 evaluation failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Control evaluation completed; evaluating the MIR held-out text.'
$ErrorActionPreference = 'Continue'
& $pythonPath 'scripts/evaluate_exp0016_25m_wiki.py' --checkpoint (Join-Path $mirRun 'checkpoints/step_00001500.pt') --out (Join-Path $mirRun 'evaluation-primary-text.json') --primary-text-only --text-only --experiment-id EXP-0021 *> (Join-Path $mirRun 'evaluation-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0021 evaluation failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Both BPB evaluations completed; running paired document-bootstrap analysis.'
New-Item -ItemType Directory -Path (Join-Path $workspacePath 'runs/EXP-0020-0021-25m-mir-2026-10-23') -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath 'scripts/analyze_exp0020_21_mir.py' --control (Join-Path $controlRun 'evaluation-primary-text.json') --mir (Join-Path $mirRun 'evaluation-primary-text.json') --out (Join-Path $workspacePath 'runs/EXP-0020-0021-25m-mir-2026-10-23/analysis.json') *> (Join-Path $workspacePath 'runs/EXP-0020-0021-25m-mir-2026-10-23/analysis-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "MIR paired analysis failed with exit code $exitCode."; exit $exitCode }
Write-QueueLog 'EXP-0020/0021 training, BPB evaluation, and analysis completed.'
