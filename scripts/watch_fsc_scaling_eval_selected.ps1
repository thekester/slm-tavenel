# Stop the already-running legacy queue after 25M-grown, then evaluate only selected runs.
$ErrorActionPreference = "Stop"
$queueLog = "runs/fsc-scaling/queue.log"
$deadline = [DateTime]::UtcNow.AddHours(16)
while ([DateTime]::UtcNow -lt $deadline) {
  if (Test-Path -LiteralPath $queueLog) {
    $text = Get-Content -LiteralPath $queueLog -Raw
    if ($text -match "FAILED") { throw "Scaling queue failed before the selected 25M-grown run completed." }
    if ($text -match "COMPLETED 25M-gated-grown-s1234") { break }
  }
  Start-Sleep -Milliseconds 300
}
if ([DateTime]::UtcNow -ge $deadline) { throw "Timed out waiting for the 25M-grown checkpoint." }

# The original queue process has the scratch run in memory. Stop that queue at this boundary.
$queueProcesses = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match "-File scripts/run_fsc_scaling\.ps1"
}
foreach ($process in $queueProcesses) {
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 300
$scratchProcesses = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match "slm train configs/train/fsc-pilot-v1-25m-scratch\.yaml"
}
foreach ($process in $scratchProcesses) {
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
Add-Content -LiteralPath $queueLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 25M scratch deferred by user; evaluating selected runs"

$evalLog = "runs/fsc-scaling/evaluation-queue.log"
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_fsc_scaling.ps1 *>> $evalLog
if ($LASTEXITCODE -ne 0) { throw "Selected scaling evaluations failed with exit code $LASTEXITCODE" }
Set-Content -LiteralPath "runs/fsc-scaling/evaluation-complete.txt" -Value "Selected EXP-0009 evaluations completed at $(Get-Date -Format o); 25M scratch deferred"
