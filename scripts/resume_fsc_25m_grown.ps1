# Resume the selected 25M-grown run with separate stdout/stderr capture.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
if ($LASTEXITCODE -ne 0) { throw "Could not locate PyTorch runtime" }
$env:PATH = "$torchLib;$env:PATH"

$outDir = "runs/fsc-scaling/25m-gated-grown-s1234"
$checkpoint = Join-Path $outDir "checkpoints/step_00015000.pt"
if (Test-Path -LiteralPath $checkpoint) {
  Add-Content -LiteralPath "runs/fsc-scaling/queue.log" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') COMPLETED 25M-gated-grown-s1234 checkpoint=$checkpoint"
  exit 0
}

$savedCheckpoints = @(Get-ChildItem -LiteralPath (Join-Path $outDir "checkpoints") -Filter "step_*.pt" -File -ErrorAction SilentlyContinue)
if ($savedCheckpoints.Count -eq 0) { throw "No saved checkpoint to resume in $outDir" }
$stdout = Join-Path "runs/fsc-scaling" "25m-gated-grown-s1234.resume.stdout.log"
$stderr = Join-Path "runs/fsc-scaling" "25m-gated-grown-s1234.resume.stderr.log"
$process = Start-Process -FilePath "python.exe" `
  -ArgumentList @("-m", "slm", "train", "configs/train/fsc-pilot-v1-25m-grown.yaml") `
  -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Set-Content -LiteralPath (Join-Path $outDir "active-train-pid.txt") -Value $process.Id
Add-Content -LiteralPath "runs/fsc-scaling/queue.log" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') resuming 25M-gated-grown-s1234 from latest checkpoint"
$process.WaitForExit()
$exitCode = $process.ExitCode
Remove-Item -LiteralPath (Join-Path $outDir "active-train-pid.txt") -ErrorAction SilentlyContinue
if ($exitCode -ne 0) {
  Add-Content -LiteralPath "runs/fsc-scaling/queue.log" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FAILED 25M-gated-grown-s1234 exit=$exitCode; inspect $stderr"
  throw "25M-grown resume failed with exit code $exitCode; see $stderr"
}
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "25M-grown training exited without final checkpoint $checkpoint" }
Add-Content -LiteralPath "runs/fsc-scaling/queue.log" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') COMPLETED 25M-gated-grown-s1234 checkpoint=$checkpoint"
