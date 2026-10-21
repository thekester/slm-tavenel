# Resume the original FSC evaluation queue, then run the supplementary benchmark wave.
# Completed evaluations are reused when their signatures match. Incomplete model evaluations
# are recomputed; the active model is not checkpointed during evaluation.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
if ($LASTEXITCODE -ne 0) { throw "Python environment lookup failed with exit code $LASTEXITCODE" }
$env:PATH = "$torchLib;$env:PATH"

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_fsc_scaling.ps1
if ($LASTEXITCODE -ne 0) { throw "The original FSC evaluation queue failed with exit code $LASTEXITCODE" }

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/queue_fsc_additional_benchmarks.ps1
if ($LASTEXITCODE -ne 0) { throw "The supplementary FSC benchmark queue failed with exit code $LASTEXITCODE" }
