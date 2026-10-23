$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = 'C:\Program Files\Python311\python.exe'
$wikiRun = Join-Path $workspacePath 'runs/EXP-0022-25m-wiki-10pct-s2-2026-10-23'
$queueLog = Join-Path $workspacePath 'runs/EXP-0022-wiki-replication-queue.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $queueLog) -Force | Out-Null
Set-Location -LiteralPath $workspacePath
$env:PYTHONPATH = Join-Path $workspacePath 'src'
$env:PYTHONUTF8 = '1'

function Write-QueueLog([string]$message) {
    $stamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz')
    Add-Content -LiteralPath $queueLog -Value "$stamp $message" -Encoding utf8
}

if (Test-Path -LiteralPath $wikiRun) { throw "Refusing to overwrite existing EXP-0022 output: $wikiRun" }
Write-QueueLog 'Starting the matched seed-2 10% wiki replication arm.'
New-Item -ItemType Directory -Path $wikiRun | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train 'configs/train/exp0022-25m-wiki-10pct-seed2.yaml' *> (Join-Path $wikiRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0022 training failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'EXP-0022 training completed; evaluating matched held-out BPB.'
$ErrorActionPreference = 'Continue'
& $pythonPath 'scripts/evaluate_exp0016_25m_wiki.py' --checkpoint (Join-Path $wikiRun 'checkpoints/step_00001500.pt') --out (Join-Path $wikiRun 'evaluation-primary-text.json') --primary-text-only --text-only --experiment-id EXP-0022 *> (Join-Path $wikiRun 'evaluation-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "EXP-0022 evaluation failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'EXP-0022 evaluation completed; comparing both matched seeds.'
New-Item -ItemType Directory -Path (Join-Path $workspacePath 'runs/EXP-0022-25m-wiki-replication-2026-10-23') -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath 'scripts/analyze_exp0022_seed_replication.py' `
    --seed1-control 'runs/EXP-0016-25m-wiki-mixture-2026-10-23/control-primary-text.json' `
    --seed1-wiki 'runs/EXP-0017-25m-wiki-10pct-2026-10-23/evaluation.json' `
    --seed2-control 'runs/EXP-0020-25m-fsc-control-s2-2026-10-23/evaluation-primary-text.json' `
    --seed2-wiki (Join-Path $wikiRun 'evaluation-primary-text.json') `
    --out 'runs/EXP-0022-25m-wiki-replication-2026-10-23/analysis.json' `
    *> (Join-Path $workspacePath 'runs/EXP-0022-25m-wiki-replication-2026-10-23/analysis-console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Two-seed analysis failed with exit code $exitCode."; exit $exitCode }
Write-QueueLog 'EXP-0022 training, evaluation, and two-seed analysis completed.'
