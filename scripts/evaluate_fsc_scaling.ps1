# Evaluate the controlled FSC scaling runs and paired benchmark differences.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
$env:PATH = "$torchLib;$env:PATH"

$runs = @(
  @{ name = "1m-gated-s1234"; dir = "runs/fsc-scaling/1m-gated-s1234" },
  @{ name = "10m-gated-s1234"; dir = "runs/fsc-scaling/10m-gated-s1234" },
  @{ name = "25m-gated-grown-s1234"; dir = "runs/fsc-scaling/25m-gated-grown-s1234" }
)
$runs = @($runs | Where-Object { Test-Path -LiteralPath "$($_.dir)/checkpoints/step_00015000.pt" })
if ($runs.Count -lt 1) { throw "No completed local checkpoints are available for evaluation." }
$benchmarks = @("data/benchmarks/arc_easy.jsonl", "data/benchmarks/hellaswag.jsonl",
                "data/benchmarks/belebele_en.jsonl", "data/benchmarks/belebele_fr.jsonl")
foreach ($run in $runs) {
  $checkpoint = "$($run.dir)/checkpoints/step_00015000.pt"
  if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint $checkpoint" }
  $analysis = "$($run.dir)/analysis"
  New-Item -ItemType Directory -Force $analysis | Out-Null
  & python scripts/evaluate_fsc_checkpoint.py --checkpoint $checkpoint --out-dir $analysis --device cuda
  if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($run.name)" }
  Write-Host "EVALUATED $($run.name)"
}

New-Item -ItemType Directory -Force runs/fsc-scaling/comparisons | Out-Null
$pairs = @(@{ base = "1m-gated-s1234"; treatment = "10m-gated-s1234"; label = "1m-vs-10m" })
if (Test-Path -LiteralPath "runs/fsc-scaling/25m-gated-grown-s1234/checkpoints/step_00015000.pt") {
  $pairs += @{ base = "10m-gated-s1234"; treatment = "25m-gated-grown-s1234"; label = "10m-vs-25m-grown" }
}
foreach ($pair in $pairs) {
  if ($pair.base -notin $runs.name -or $pair.treatment -notin $runs.name) { continue }
  & python scripts/compare_mc.py `
    "runs/fsc-scaling/$($pair.base)/analysis/fsc-pilot-v1-mc.json" `
    "runs/fsc-scaling/$($pair.treatment)/analysis/fsc-pilot-v1-mc.json" `
    --seed 20261020 --draws 20000 --out "runs/fsc-scaling/comparisons/$($pair.label).json"
  if ($LASTEXITCODE -ne 0) { throw "Paired MC comparison failed: $($pair.label)" }
}

& python scripts/evaluate_fsc_references.py --device cuda
if ($LASTEXITCODE -ne 0) { throw "External reference evaluation failed" }
$referencePairs = @(
  @{ name = "pythia-14m"; base = "10m-gated-s1234" },
  @{ name = "smollm2-135m"; base = "10m-gated-s1234" },
  @{ name = "tiny1m"; base = "1m-gated-s1234" }
)
foreach ($referencePair in $referencePairs) {
  if ($referencePair.base -notin $runs.name) { continue }
  & python scripts/compare_mc.py `
    "runs/fsc-scaling/$($referencePair.base)/analysis/fsc-pilot-v1-mc.json" `
    "runs/reference/$($referencePair.name)/fsc-pilot-v1-mc.json" `
    --seed 20261020 --draws 20000 --out "runs/fsc-scaling/comparisons/$($referencePair.base)-vs-$($referencePair.name).json"
  if ($LASTEXITCODE -ne 0) { throw "Reference comparison failed: $($referencePair.name)" }
}
& python scripts/render_fsc_benchmark_chart.py --out docs/fsc-scaling-benchmark-comparison.svg
if ($LASTEXITCODE -ne 0) { throw "Benchmark comparison chart could not be rendered" }
Write-Host "SCALING EVALUATIONS AND PAIRED COMPARISONS COMPLETE"
