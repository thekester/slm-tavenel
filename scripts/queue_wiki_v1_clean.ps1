# Preregistered 10M replication: three single-change arms, paired seeds, one GPU job at a time.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$root = "runs/wiki-v1-clean"
New-Item -ItemType Directory -Force $root | Out-Null
$queueLog = Join-Path $root "queue.log"
function Log([string]$message) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
  Add-Content -LiteralPath $queueLog -Value $line
  Write-Host $line
}

$snapshotHash = Join-Path $root "source-snapshot.sha256"
& python scripts/audit_wiki_v1_decontamination.py 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Could not write decontamination audit" }
$snapshotValue = & python scripts/snapshot_clean_run_sources.py 2>$null
if ($LASTEXITCODE -ne 0) { throw "Could not write source snapshot" }
Set-Content -LiteralPath $snapshotHash -Value $snapshotValue -Encoding utf8

$jobs = @(
  @{ name = "adamw-s1234"; config = "configs/train/slm-10m-wiki.yaml"; seed = 1234 },
  @{ name = "gate-s1234";  config = "configs/train/slm-10m-wiki-gated.yaml"; seed = 1234 },
  @{ name = "muon-s1234"; config = "configs/train/slm-10m-wiki-muon.yaml"; seed = 1234 },
  @{ name = "adamw-s2";    config = "configs/train/slm-10m-wiki.yaml"; seed = 2 },
  @{ name = "gate-s2";     config = "configs/train/slm-10m-wiki-gated.yaml"; seed = 2 },
  @{ name = "muon-s2";     config = "configs/train/slm-10m-wiki-muon.yaml"; seed = 2 },
  @{ name = "adamw-s3";    config = "configs/train/slm-10m-wiki.yaml"; seed = 3 },
  @{ name = "gate-s3";     config = "configs/train/slm-10m-wiki-gated.yaml"; seed = 3 },
  @{ name = "muon-s3";     config = "configs/train/slm-10m-wiki-muon.yaml"; seed = 3 }
)

foreach ($job in $jobs) {
  $outDir = "$root/$($job.name)"
  $finalCheckpoint = "$outDir/checkpoints/step_00015000.pt"
  if (Test-Path -LiteralPath $finalCheckpoint) {
    Log "$($job.name) already has final checkpoint; skipping"
    continue
  }
  $errFile = "$root/$($job.name).err"
  Log "starting $($job.name) from step 0 on wiki-v1"
  & python -m slm train $job.config --set "data_dir=data/tokens/wiki-v1-clean" "out_dir=$outDir" "seed=$($job.seed)" 2>> $errFile
  if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath "$root/queue-failed.txt" -Value "$($job.name) exit code $LASTEXITCODE at $(Get-Date -Format o)"
    throw "$($job.name) failed with exit code $LASTEXITCODE"
  }
  if (-not (Test-Path -LiteralPath $finalCheckpoint)) {
    Set-Content -LiteralPath "$root/queue-failed.txt" -Value "$($job.name) returned without the final checkpoint at $(Get-Date -Format o)"
    throw "$($job.name) did not create its final checkpoint"
  }
  Log "completed $($job.name): 15,000 steps"
}

Set-Content -LiteralPath "$root/queue-complete.txt" -Value "All nine wiki-v1 10M runs completed at $(Get-Date -Format o)"
Log "all nine runs complete"
