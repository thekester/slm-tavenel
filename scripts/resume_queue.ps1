# Resume the experiment queue after a shutdown. Run at most one instance per mode:
# completed outputs are skipped, interrupted training resumes from its latest checkpoint,
# and a failed Python command stops the queue. CPU and GPU modes may run concurrently.
#   powershell -ExecutionPolicy Bypass -File scripts/resume_queue.ps1          # GPU queue
#   powershell -ExecutionPolicy Bypass -File scripts/resume_queue.ps1 -Cpu     # CPU jobs (run in parallel)
param([switch]$Cpu)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
function Log($m) { "$(Get-Date -Format 'HH:mm') $m" }
function Done($run) { Test-Path "runs/$run/checkpoints/step_00015000.pt" }
function Invoke-Python {
  & python @args
  if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}
function ValidJson($path) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
  try {
    $value = Get-Content -LiteralPath $path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    return $null -ne $value
  } catch { return $false }
}
function Evaluated($run) {
  foreach ($name in "bpb-en.json", "bpb-fr.json", "sinks-seed1.json", "mc.json") {
    if (-not (ValidJson "runs/$run/analysis/$name")) { return $false }
  }
  return $true
}

function Train($run, $config, $extra = @()) {
  if (Done $run) { Log "$run already finished"; return }
  $t = Measure-Command { Invoke-Python -m slm train $config @extra 2>>"runs-$run.err" | Out-Host }
  Log "$run wall: $([math]::Round($t.TotalMinutes,1)) min"
}

$benchmarks = @("data/benchmarks/arc_easy.jsonl", "data/benchmarks/hellaswag.jsonl",
                "data/benchmarks/belebele_en.jsonl", "data/benchmarks/belebele_fr.jsonl")

function EvalCpu($run) {
  # bits per byte on fixed documents, benchmarks and sink analysis, on CPU with 4 threads
  if (-not (Done $run)) { return }
  if (Evaluated $run) { return }
  $env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
  $ck = "runs/$run/checkpoints/step_00015000.pt"
  New-Item -ItemType Directory -Force "runs/$run/analysis" | Out-Null
  foreach ($l in "en", "fr") {
    if (ValidJson "runs/$run/analysis/bpb-$l.json") { continue }
    Invoke-Python -m slm eval bpb --device cpu --checkpoint $ck --tokenizer data/tokenizer/wiki-v0 `
      --data "data/processed/wiki-v0/val-$l.jsonl" --limit 500 2>$null | Out-File -Encoding utf8 "runs/$run/analysis/bpb-$l.json"
  }
  if (-not (ValidJson "runs/$run/analysis/sinks-seed1.json")) {
    Invoke-Python scripts/analyze_sinks.py $ck --data data/tokens/wiki-v0 --windows 128 --seq-len 256 --seed 1 `
      --out "runs/$run/analysis/sinks-seed1.json" 2>$null | Out-Null
  }
  if (-not (ValidJson "runs/$run/analysis/mc.json")) {
    Invoke-Python -m slm eval mc --device cpu --checkpoint $ck --tokenizer data/tokenizer/wiki-v0 `
      --benchmarks @benchmarks --out "runs/$run/analysis/mc.json" 2>$null | Out-Null
  }
  Log "$run evaluated"
}

if ($Cpu) {
  if (-not (ValidJson data/tokens/wiki-v1/manifest.json)) {
    Invoke-Python -m slm data run configs/data/pipeline-wiki-v1.yaml --allow-unapproved 2>data/pipeline-wiki-v1.err | Out-Null
    Invoke-Python -m slm data encode --tokenizer data/tokenizer/wiki-v0 --inputs data/processed/wiki-v1/documents.jsonl `
      --out data/tokens/wiki-v1 --val-fraction 0.01 2>$null | Out-Null
    Log "wiki-v1 built"
  }
  # Evaluate every finished run as soon as it exists; loop until the GPU queue's last run is evaluated.
  $runs = "slm-10m-wiki-docmask", "slm-10m-wiki-muon-s2", "slm-10m-wiki-gated-muon",
          "slm-1m-wiki-gated", "slm-25m-wiki-grown", "slm-25m-wiki-scratch"
  while ($true) {
    foreach ($r in $runs) { EvalCpu $r }
    if (Evaluated "slm-25m-wiki-scratch") { break }
    Start-Sleep -Seconds 300
  }
  exit 0
}

# ---- GPU queue, in order ----
Train "slm-10m-wiki-muon-s2" "configs/train/slm-10m-wiki-muon.yaml" @("--set", "seed=2", "out_dir=runs/slm-10m-wiki-muon-s2")
Train "slm-10m-wiki-gated-muon" "configs/train/slm-10m-wiki-gated-muon.yaml"

$ref = "HuggingFaceTB/SmolLM2-135M"; $rev = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
New-Item -ItemType Directory -Force runs/reference/smollm2-135m | Out-Null
if (-not (ValidJson runs/reference/smollm2-135m/mc.json)) {
  Invoke-Python -m slm eval mc --hf-model $ref --hf-revision $rev --device cuda --out runs/reference/smollm2-135m/mc.json `
    --benchmarks @benchmarks 2>$null | Out-Null
}
foreach ($l in "en", "fr") {
  if (-not (ValidJson "runs/reference/smollm2-135m/bpb-$l.json")) {
    Invoke-Python -m slm eval bpb --hf-model $ref --hf-revision $rev --device cuda --data "data/processed/wiki-v0/val-$l.jsonl" `
      --limit 500 2>$null | Out-File -Encoding utf8 "runs/reference/smollm2-135m/bpb-$l.json"
  }
}
Log "reference evaluated"

New-Item -ItemType Directory -Force runs/grown | Out-Null
if (-not (Test-Path runs/grown/slm-25m-gated-from-10m.pt)) {
  Invoke-Python -m slm grow --checkpoint runs/slm-10m-wiki-gated/checkpoints/step_00015000.pt `
    --model configs/model/slm-25m-gated.yaml --out runs/grown/slm-25m-gated-from-10m.pt 2>$null | Out-Null
}
if (-not (ValidJson runs/grown/bench-25m.json)) {
  Invoke-Python -m slm bench configs/train/slm-25m-wiki-scratch.yaml --steps 15 --warmup 3 2>$null | Out-File -Encoding utf8 runs/grown/bench-25m.json
}
$mem = (Get-Content runs/grown/bench-25m.json -Raw | ConvertFrom-Json).max_mem_gb
$override = if ($mem -gt 6.5) { @("--set", "micro_batch_size=4", "grad_accum_steps=8") } else { @() }
Log "25M bench: $mem GB"

Train "slm-1m-wiki-gated" "configs/train/slm-1m-wiki-gated.yaml"
Train "slm-25m-wiki-grown" "configs/train/slm-25m-wiki-grown.yaml" $override
Train "slm-25m-wiki-scratch" "configs/train/slm-25m-wiki-scratch.yaml" $override
Log "GPU queue finished"
