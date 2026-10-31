param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# EXP-0039 gridworld world-model pilot: arms A (transitions), C (transitions + state-level JEPA) and
# B (plans, verifier tool at test time); next-state accuracy (val, interventions, held-out layouts)
# for A and C, planning success for all three. Resumable; serial GPU use.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$logDir = Join-Path $ProjectRoot 'runs/EXP-0039-automation'
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
$arms = @(
    @{ slug = 'a-transitions'; kind = 'transition' },
    @{ slug = 'c-jepa'; kind = 'transition' },
    @{ slug = 'b-plans'; kind = 'plan' }
)
foreach ($arm in $arms) {
    $ckpt = "runs/EXP-0039-grid-$($arm.slug)/checkpoints/step_00010000.pt"
    if (-not (Test-Path -LiteralPath $ckpt)) {
        Invoke-Python "exp0039-$($arm.slug)-train" @('-m', 'slm', 'train', "configs/train/exp0039-grid-$($arm.slug).yaml",
            '--set', 'resume=true')
    }
    $evalDir = 'runs/EXP-0039-evaluations'
    if ($arm.kind -eq 'transition' -and -not (Test-Path -LiteralPath "$evalDir/$($arm.slug)-transitions.json")) {
        Invoke-Python "exp0039-$($arm.slug)-transitions-eval" @('scripts/evaluate_gridworld_transitions.py',
            '--checkpoint', $ckpt, '--out', "$evalDir/$($arm.slug)-transitions.json")
    }
    if (-not (Test-Path -LiteralPath "$evalDir/$($arm.slug)-planning.json")) {
        Invoke-Python "exp0039-$($arm.slug)-planning-eval" @('scripts/evaluate_gridworld_planning.py',
            '--checkpoint', $ckpt, '--kind', $arm.kind, '--out', "$evalDir/$($arm.slug)-planning.json")
    }
}
Write-Status 'EXP-0039 gridworld pilot complete'
