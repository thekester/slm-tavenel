# Wait for the sequential scaling queue, then run all same-harness evaluations once.
$ErrorActionPreference = "Stop"
$queueLog = "runs/fsc-scaling/queue.log"
while ($true) {
  if (Test-Path -LiteralPath $queueLog) {
    $text = Get-Content -LiteralPath $queueLog -Raw
    if ($text -match "SCALING TRAINING QUEUE COMPLETE") { break }
    if ($text -match "FAILED") { throw "Scaling queue failed; evaluation watcher will stop." }
  }
  Start-Sleep -Seconds 60
}
$evalLog = "runs/fsc-scaling/evaluation-queue.log"
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_fsc_scaling.ps1 *>> $evalLog
if ($LASTEXITCODE -ne 0) { throw "Scaling evaluation queue failed with exit code $LASTEXITCODE" }
Set-Content -LiteralPath "runs/fsc-scaling/evaluation-complete.txt" -Value "All EXP-0009 local and reference evaluations completed at $(Get-Date -Format o)"
