param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [int]$WaitForProcessId = 0
)

# Score every public, non-gated model of configs/eval/peer-models-2026-10.json that has no result yet,
# at its pinned revision, with the same six files, evaluator and 1,024-token context cap as the
# earlier SmolLM2-360M / Qwen3-0.6B runs. Resumable: finished models are skipped. GPU work is serial.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectRoot
$python = 'C:\Program Files\Python311\python.exe'
$env:PYTHONPATH = 'src'
$base = Join-Path $ProjectRoot 'runs/reference/peer-models-2026-10'
New-Item -ItemType Directory -Force -Path $base | Out-Null
$status = Join-Path $base 'queue-status.log'
$benchmarks = @(
    'data/benchmarks/arc_easy.jsonl', 'data/benchmarks/arc_challenge.jsonl',
    'data/benchmarks/hellaswag.jsonl', 'data/benchmarks/belebele_en.jsonl',
    'data/benchmarks/belebele_fr.jsonl', 'data/benchmarks/winogrande.jsonl'
)

if ($WaitForProcessId -gt 0) {
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
$manifest = Get-Content -LiteralPath 'configs/eval/peer-models-2026-10.json' -Raw | ConvertFrom-Json
foreach ($model in $manifest.models) {
    if ($model.gated) { continue }
    $slug = ($model.name.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    $outDir = Join-Path $base $slug
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outFile = Join-Path $outDir 'mc.json'
    if ((Test-Path -LiteralPath $outFile) -or (Test-Path -LiteralPath (Join-Path $base "$($slug -replace '-0-6b','-06b')/mc.json"))) { continue }
    "$(Get-Date -Format o) start $($model.repo)@$($model.revision)" | Add-Content -LiteralPath $status -Encoding utf8
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -m slm eval mc --hf-model $model.repo --hf-revision $model.revision --max-seq-len 1024 `
        --batch-items 2 --device cuda --benchmarks $benchmarks --out $outFile *> (Join-Path $outDir 'evaluation.log')
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    "$(Get-Date -Format o) exit $code $($model.repo)" | Add-Content -LiteralPath $status -Encoding utf8
}
"$(Get-Date -Format o) peer manifest evaluation complete" | Add-Content -LiteralPath $status -Encoding utf8
