# Evaluate a size-near public SLM with the exact SLM-Tavenel multiple-choice harness.
# By default, wait for the project's CPU queue to exit. Use -RunNow only when that queue is idle.
param([switch]$RunNow)
$ErrorActionPreference = "Stop"
while (-not $RunNow) {
  $queue = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'resume_queue\.ps1.*-Cpu' }
  if (-not $queue) { break }
  Start-Sleep -Seconds 30
}

$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$model = "EleutherAI/pythia-14m"
$revision = "d31f659f077f9ea500c4d7faa999893d7f28cbdf"
$outDir = "runs/reference/pythia-14m"
$outFile = "$outDir/mc.json"
New-Item -ItemType Directory -Force $outDir | Out-Null

if (Test-Path -LiteralPath $outFile) {
  try {
    $value = Get-Content -LiteralPath $outFile -Raw | ConvertFrom-Json -ErrorAction Stop
    if ($null -ne $value) { exit 0 }
  }
  catch { Remove-Item -LiteralPath $outFile -Force }
}

python -m slm eval mc --hf-model $model --hf-revision $revision --device cpu `
  --benchmarks data/benchmarks/arc_easy.jsonl data/benchmarks/hellaswag.jsonl `
    data/benchmarks/belebele_en.jsonl data/benchmarks/belebele_fr.jsonl `
  --out $outFile
if ($LASTEXITCODE -ne 0) { throw "Pythia-14M evaluation failed with exit code $LASTEXITCODE" }
