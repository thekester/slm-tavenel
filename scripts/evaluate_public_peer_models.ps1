param(
    [Parameter(Mandatory = $true)][int]$WaitForProcessId,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$logDir = Join-Path $ProjectRoot 'runs/EXP-0024-automation'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl',
    'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl',
    'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl',
    'data/benchmarks/winogrande.jsonl'
)
$references = @(
    @{ slug = 'smollm2-360m'; repo = 'HuggingFaceTB/SmolLM2-360M'; revision = 'f8027fd0eaeea54caa13c31d31b9fdc459c38b49' },
    @{ slug = 'qwen3-06b'; repo = 'Qwen/Qwen3-0.6B'; revision = 'c1899de289a04d12100db370d81485cdf75e47ca' }
)
$base = Join-Path $ProjectRoot 'runs/reference/peer-models-2026-10'
New-Item -ItemType Directory -Force -Path $base | Out-Null
$status = @()
foreach ($ref in $references) {
    $outDir = Join-Path $base $ref.slug
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outFile = Join-Path $outDir 'mc.json'
    $logFile = Join-Path $logDir ($ref.slug + '-evaluation.log')
    # PowerShell can turn Python warnings written to stderr into terminating
    # NativeCommandError records when ErrorActionPreference is Stop.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -m slm eval mc --hf-model $ref.repo --hf-revision $ref.revision `
        --max-seq-len 1024 --batch-items 2 --device cuda --benchmarks $benchmarks --out $outFile *> $logFile
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    $status += [pscustomobject]@{
        repo = $ref.repo
        revision = $ref.revision
        exit_code = $exitCode
        result = $outFile
        log = $logFile
    }
}
$status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $base 'public-peer-evaluation-status.json') -Encoding utf8
