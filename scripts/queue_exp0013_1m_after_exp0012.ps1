$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = 'C:\Program Files\Python311\python.exe'
$completionLog = Join-Path $workspacePath 'runs/EXP-0012-py311-queue.log'
$queueLog = Join-Path $workspacePath 'runs/EXP-0013-1m-queue.log'
$motTrain = Join-Path $workspacePath 'data/processed/exp0012-mot-v1.11/temporal/train.jsonl'
$motTokens = Join-Path $workspacePath 'data/tokens/exp0013-1m-mot-tokenized'
$mixtureTokens = Join-Path $workspacePath 'data/tokens/exp0013-1m-fsc-mot-mixture'
$controlRun = Join-Path $workspacePath 'runs/EXP-0013-1m-fsc-control-2026-10-22'
$mixRun = Join-Path $workspacePath 'runs/EXP-0013-1m-fsc-mot-mixture-2026-10-22'
New-Item -ItemType Directory -Path (Split-Path -Parent $queueLog) -Force | Out-Null
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The recorded Python runtime is unavailable: $pythonPath"
}

function Get-PredecessorState([string]$contents) {
    $state = 'waiting'
    foreach ($line in ($contents -split "`r?`n")) {
        if ($line -match 'Starting matched FSC-only control') { $state = 'waiting' }
        elseif ($line -match 'Control failed|Mixture arm failed|evaluation failed') { $state = 'failed' }
        elseif ($line -match 'Both EXP-0012 held-out evaluations completed') { $state = 'complete' }
    }
    return $state
}

function Write-QueueLog([string]$message) {
    $stamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz')
    Add-Content -LiteralPath $queueLog -Value "$stamp $message" -Encoding utf8
}

Set-Location -LiteralPath $workspacePath
$env:PYTHONPATH = Join-Path $workspacePath 'src'
$env:PYTHONUTF8 = '1'
Write-QueueLog 'Queued behind completion of both EXP-0012 10M training runs and evaluations.'
while ($true) {
    if (Test-Path -LiteralPath $completionLog) {
        $contents = Get-Content -LiteralPath $completionLog -Raw
        $predecessorState = Get-PredecessorState $contents
        if ($predecessorState -eq 'complete') { break }
        if ($predecessorState -eq 'failed') {
            Write-QueueLog 'EXP-0012 did not complete cleanly; EXP-0013 was not started.'
            exit 2
        }
    }
    Start-Sleep -Seconds 30
}

Write-QueueLog '10M pipeline complete; encoding temporal MOT training documents with the frozen 1M tokenizer.'
if (-not (Test-Path -LiteralPath (Join-Path $motTokens 'manifest.json'))) {
    if (Test-Path -LiteralPath $motTokens) { throw "Refusing to overwrite incomplete token output: $motTokens" }
    $ErrorActionPreference = 'Continue'
    & $pythonPath -m slm data encode --tokenizer 'data/tokenizers/fsc-cc0-open-weights-v4' --inputs $motTrain --out $motTokens --val-fraction 0 --shard-tokens 50000000 *> (Join-Path $workspacePath 'runs/EXP-0013-1m-mot-encode.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exitCode -ne 0) { Write-QueueLog "MOT tokenization failed with exit code $exitCode."; exit $exitCode }
}

Write-QueueLog 'Building the matched 80/20 FSC/MOT 1M data stream.'
$ErrorActionPreference = 'Continue'
& $pythonPath (Join-Path $workspacePath 'scripts/build_exp0013_1m_mixture.py') --mot $motTokens --out $mixtureTokens *> (Join-Path $workspacePath 'runs/EXP-0013-1m-mixture-build.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "Mixture build failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Starting the matched 1M FSC-only control.'
New-Item -ItemType Directory -Path $controlRun -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train 'configs/train/exp0013-1m-fsc-control.yaml' *> (Join-Path $controlRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "1M FSC control failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'FSC control finished; starting the matched 1M 20% MOT arm.'
New-Item -ItemType Directory -Path $mixRun -Force | Out-Null
$ErrorActionPreference = 'Continue'
& $pythonPath -m slm train 'configs/train/exp0013-1m-fsc-mot-mixture.yaml' *> (Join-Path $mixRun 'console.log')
$exitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exitCode -ne 0) { Write-QueueLog "1M MOT mixture arm failed with exit code $exitCode."; exit $exitCode }

Write-QueueLog 'Both 1M arms finished; running paired, one-shot held-out evaluations.'
foreach ($arm in @(
    @{ name = 'FSC control'; run = $controlRun },
    @{ name = 'FSC/MOT mixture'; run = $mixRun }
)) {
    $checkpoint = Join-Path $arm.run 'checkpoints/step_00000488.pt'
    $outPath = Join-Path $arm.run 'analysis/heldout-evaluation.json'
    $ErrorActionPreference = 'Continue'
    & $pythonPath (Join-Path $workspacePath 'scripts/evaluate_exp0013_1m.py') --checkpoint $checkpoint --out $outPath *> (Join-Path $arm.run 'evaluation-console.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exitCode -ne 0) { Write-QueueLog "Evaluation failed for $($arm.name), exit code $exitCode."; exit $exitCode }
}
Write-QueueLog 'Both EXP-0013 held-out evaluations completed. Results are a one-seed pilot only.'
