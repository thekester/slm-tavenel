param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0038 amendment (2026-11-02): low-data symmetry arms S0 (50k games) and S1 (same games under the
# 4 opening-preserving symmetries), training and the same evaluation as arms A and C. Resumable.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0038-automation'
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
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Write-Status "symmetry arms started (pid $PID)"
if (-not (Test-Path -LiteralPath 'data/worlds/othello-v1-50k-sym4/manifest.json')) {
    Invoke-Python 'build-symmetry-subsets' @('scripts/build_othello_symmetry_subsets.py')
}
foreach ($arm in @('s0-50k', 's1-50k-sym4')) {
    $ckpt = "runs/EXP-0038-othello-$arm/checkpoints/step_00005000.pt"
    if (-not (Test-Path -LiteralPath $ckpt)) {
        Invoke-Python "exp0038-$arm-train" @('-m', 'slm', 'train', "configs/train/exp0038-othello-$arm.yaml", '--set', 'resume=true')
    }
    $out = "runs/EXP-0038-evaluations/othello-$arm.json"
    if (-not (Test-Path -LiteralPath $out)) {
        Invoke-Python "exp0038-$arm-eval" @('scripts/evaluate_othello_world_model.py', '--checkpoint', $ckpt, '--out', $out)
    }
}
Write-Status 'EXP-0038 symmetry arms complete'
