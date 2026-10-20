# Controlled 1M / 10M / 25M scaling series on the frozen FSC pilot token corpus.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
$env:PATH = "$torchLib;$env:PATH"

$root = "runs/fsc-scaling"
New-Item -ItemType Directory -Force $root | Out-Null
$queueLog = Join-Path $root "queue.log"
function Log([string]$message) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
  Add-Content -LiteralPath $queueLog -Value $line
  Write-Host $line
}
function Train([string]$name, [string]$config, [string]$finalCheckpoint) {
  if (Test-Path -LiteralPath $finalCheckpoint) { Log "completed checkpoint exists; skip $name"; return }
  Log "starting $name"
  $log = Join-Path $root "$name.console.log"
  & python -m slm train $config 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { Log "FAILED $name exit=$LASTEXITCODE"; throw "$name failed" }
  if (-not (Test-Path -LiteralPath $finalCheckpoint)) { Log "FAILED $name missing final checkpoint"; throw "$name returned without final checkpoint" }
  Log "COMPLETED $name checkpoint=$finalCheckpoint"
}

$manifest = "data/tokens/fsc-pilot-v1-scaling/manifest.json"
if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing frozen FSC token manifest: $manifest" }
$trainManifest = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
if ($trainManifest.splits.train.tokens -lt 500000000) { throw "FSC train split is too small for the preregistered 491.52M-token budget" }

Train "1m-gated-s1234" "configs/train/fsc-pilot-v1-1m.yaml" "$root/1m-gated-s1234/checkpoints/step_00015000.pt"
Train "10m-gated-s1234" "configs/train/fsc-pilot-v1-10m.yaml" "$root/10m-gated-s1234/checkpoints/step_00015000.pt"

$grown = "$root/grown/25m-gated-from-10m.pt"
if (-not (Test-Path -LiteralPath $grown)) {
  Log "growing the completed FSC 10M checkpoint to 25M"
  & python -m slm grow --checkpoint "$root/10m-gated-s1234/checkpoints/step_00015000.pt" `
    --model configs/model/slm-25m-gated.yaml --out $grown 2>&1 | Tee-Object -FilePath (Join-Path $root "grow.console.log") -Append
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $grown)) { Log "FAILED 25M growth"; throw "25M growth failed" }
  Log "COMPLETED 25M growth checkpoint=$grown"
}

Train "25m-gated-grown-s1234" "configs/train/fsc-pilot-v1-25m-grown.yaml" "$root/25m-gated-grown-s1234/checkpoints/step_00015000.pt"
Log "SCALING TRAINING QUEUE COMPLETE"
