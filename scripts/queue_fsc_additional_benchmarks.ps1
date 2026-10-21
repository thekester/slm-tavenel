# Native Python libraries emit informational warnings on stderr; keep those in the queue log and
# use explicit exit-code checks below to stop on actual Python failures.
$ErrorActionPreference = "Continue"
while (Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.CommandLine -match "scripts/evaluate_fsc_scaling\.ps1" }) {
  Start-Sleep -Seconds 10
}
foreach ($path in @(
  "runs/reference/pythia-14m/fsc-pilot-v1-mc.json",
  "runs/reference/smollm2-135m/fsc-pilot-v1-mc.json",
  "runs/reference/tiny1m/fsc-pilot-v1-mc.json"
)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "The prior evaluation ended without $path; stopping before the next GPU phase." }
}

$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
if ($LASTEXITCODE -ne 0) { throw "Python environment lookup failed with exit code $LASTEXITCODE" }
$env:PATH = "$torchLib;$env:PATH"
New-Item -ErrorAction Stop -ItemType Directory -Force runs/fsc-additional | Out-Null
Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log -Value "$(Get-Date -Format o) Prior EXP-0009 references complete; starting supplementary benchmark phase"

& python scripts/evaluate_fsc_additional.py --fetch-only 2>&1 | Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log
if ($LASTEXITCODE -ne 0) { throw "Supplementary benchmark fetch failed with exit code $LASTEXITCODE" }
& python scripts/audit_fsc_additional_contamination.py 2>&1 | Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log
if ($LASTEXITCODE -ne 0) { throw "Exact-overlap audit failed with exit code $LASTEXITCODE" }
& python scripts/evaluate_fsc_additional.py --device cuda --skip-fetch 2>&1 | Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log
if ($LASTEXITCODE -ne 0) { throw "Supplementary model evaluations failed with exit code $LASTEXITCODE" }

$pairs = @(
  @{ base = "slm-tavenel-1m"; ref = "tiny1m" },
  @{ base = "slm-tavenel-1m"; ref = "slm-tavenel-10m" },
  @{ base = "slm-tavenel-10m"; ref = "pythia-14m" },
  @{ base = "slm-tavenel-10m"; ref = "pythia-31m" },
  @{ base = "slm-tavenel-10m"; ref = "smollm2-135m" }
)
New-Item -ErrorAction Stop -ItemType Directory -Force runs/fsc-additional/comparisons | Out-Null
foreach ($pair in $pairs) {
  & python scripts/compare_mc.py `
    "runs/fsc-additional/$($pair.base)/fsc-pilot-v2-mc.json" `
    "runs/fsc-additional/$($pair.ref)/fsc-pilot-v2-mc.json" `
    --seed 20261020 --draws 20000 `
    --out "runs/fsc-additional/comparisons/$($pair.base)-vs-$($pair.ref).json" `
    2>&1 | Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log
  if ($LASTEXITCODE -ne 0) { throw "Paired bootstrap failed for $($pair.base) vs $($pair.ref)" }
}
& python scripts/render_fsc_additional_benchmarks.py 2>&1 | Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log
if ($LASTEXITCODE -ne 0) { throw "Supplementary chart and README update failed" }
Set-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/complete.txt -Value "Completed at $(Get-Date -Format o)"
Add-Content -ErrorAction Stop -LiteralPath runs/fsc-additional/queue.log -Value "$(Get-Date -Format o) Supplementary benchmark phase complete"
