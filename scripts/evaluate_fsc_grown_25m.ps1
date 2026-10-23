# Run the final 25M FSC held-out BPB and six-task benchmark evaluation.
$ErrorActionPreference = "Stop"
$checkpoint = "runs/fsc-scaling/25m-gated-grown-s1234/checkpoints/step_00015000.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
  throw "The final 25M checkpoint is not present yet: $checkpoint"
}

$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$runtime = python -c "import sys, torch; print(sys.version.split()[0] + '|' + torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the active Python/PyTorch environment" }
if ($runtime -ne "3.11.9|2.10.0+cu130") {
  throw "Use the recorded resume runtime Python 3.11.9 / PyTorch 2.10.0+cu130; found $runtime"
}

$torchLib = python -c "import site; from pathlib import Path; print(Path(site.getusersitepackages()) / 'torch' / 'lib')"
if ($LASTEXITCODE -ne 0) { throw "Could not locate the PyTorch runtime libraries" }
$env:PATH = "$torchLib;$env:PATH"
$log = "runs/fsc-scaling/25m-gated-grown-s1234/analysis/evaluation-console.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
python scripts/evaluate_fsc_grown_25m.py 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "25M final evaluation failed with exit code $LASTEXITCODE" }
