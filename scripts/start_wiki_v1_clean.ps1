# Start the registered GPU training queue and its CPU evaluation companion.
$ErrorActionPreference = "Stop"
foreach ($path in @(
  "data/processed/wiki-v1/decontamination-audit.json",
  "data/processed/wiki-v1/bpb-holdout-manifest.json",
  "data/processed/wiki-v1/tokenizer-train-sample-manifest.json",
  "data/processed/wiki-v1/tokenizer-train-120k.jsonl",
  "data/tokenizer/wiki-v1-clean/manifest.json",
  "data/tokens/wiki-v1-clean/manifest.json"
)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required artifact is missing: $path" }
}
New-Item -ItemType Directory -Force "runs/wiki-v1-clean" | Out-Null
$workingDirectory = (Get-Location).Path
$eval = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -WorkingDirectory $workingDirectory `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/eval_wiki_v1_clean.ps1")
$train = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -WorkingDirectory $workingDirectory `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/queue_wiki_v1_clean.ps1")
$state = @{ started = (Get-Date).ToString("o"); gpu_queue_pid = $train.Id; cpu_evaluator_pid = $eval.Id }
$state | ConvertTo-Json | Set-Content -LiteralPath "runs/wiki-v1-clean/processes.json" -Encoding utf8
Write-Host "Started clean wiki-v1 replication; GPU queue PID $($train.Id), CPU evaluator PID $($eval.Id)"
