param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0038 Othello world-model pilot: arm A (next-move prediction) and arm C (+ action-conditioned
# JEPA), then legal-move rate and board-state probes for both. Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0038-automation'
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
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "queue started (pid $PID)"
if (-not (Test-Path -LiteralPath 'data/worlds/othello-v1/manifest.json')) {
    Invoke-Python 'build-othello' @('scripts/build_othello_corpus.py', '--workers', '10')
}
foreach ($arm in @(@{ slug = 'a-ntp' }, @{ slug = 'c-jepa' })) {
    $run = "runs/EXP-0038-othello-$($arm.slug)"
    $ckpt = "$run/checkpoints/step_00015000.pt"
    if (-not (Test-Path -LiteralPath $ckpt)) {
        Invoke-Python "exp0038-$($arm.slug)-train" @('-m', 'slm', 'train', "configs/train/exp0038-othello-$($arm.slug).yaml",
            '--set', 'resume=true')
    }
    $out = "runs/EXP-0038-evaluations/othello-$($arm.slug).json"
    if (-not (Test-Path -LiteralPath $out)) {
        Invoke-Python "exp0038-$($arm.slug)-eval" @('scripts/evaluate_othello_world_model.py', '--checkpoint', $ckpt,
            '--out', $out)
    }
}
Write-Status 'EXP-0038 Othello pilot complete'
