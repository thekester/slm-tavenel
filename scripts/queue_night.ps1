# Sequential GPU queue: one job at a time (concurrent jobs share 8 GB and slow each other down ~8x).
# Waits for the gated+Muon run, then: reference model evaluation, 25M throughput benchmark,
# 1M run, 25M grown run, 25M scratch run.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
function Log($m) { "$(Get-Date -Format 'HH:mm') $m" }

while (-not (Test-Path runs/slm-10m-wiki-gated-muon/checkpoints/step_00015000.pt)) { Start-Sleep -Seconds 60 }
Start-Sleep -Seconds 30

# 1. Reference model, same harness (GPU, a few minutes)
$ref = "HuggingFaceTB/SmolLM2-135M"; $rev = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
New-Item -ItemType Directory -Force runs/reference/smollm2-135m | Out-Null
python -m slm eval mc --hf-model $ref --hf-revision $rev --device cuda --out runs/reference/smollm2-135m/mc.json `
  --benchmarks data/benchmarks/arc_easy.jsonl data/benchmarks/hellaswag.jsonl data/benchmarks/belebele_en.jsonl data/benchmarks/belebele_fr.jsonl 2>$null | Out-Null
foreach ($l in "en", "fr") {
  python -m slm eval bpb --hf-model $ref --hf-revision $rev --device cuda --data "data/processed/wiki-v0/val-$l.jsonl" --limit 500 2>$null |
    Out-File -Encoding utf8 "runs/reference/smollm2-135m/bpb-$l.json"
}
Log "reference evaluated"

# 2. 25M throughput and memory; fall back to a smaller micro-batch if 8 does not fit comfortably
$bench = python -m slm bench configs/train/slm-25m-wiki-scratch.yaml --steps 15 --warmup 3 2>$null | Out-String
$bench | Out-File -Encoding utf8 runs/grown/bench-25m.json
$mem = ($bench | ConvertFrom-Json).max_mem_gb
$override = @()
if ($mem -gt 6.5) { $override = @("--set", "micro_batch_size=4", "grad_accum_steps=8") }
Log "25M bench: $mem GB, override: $override"

# 3-5. Training runs
foreach ($c in "slm-1m-wiki-gated", "slm-25m-wiki-grown", "slm-25m-wiki-scratch") {
  if (Test-Path "runs/$c") { Log "runs/$c exists, skipping"; continue }
  $args25 = if ($c -like "slm-25m*") { $override } else { @() }
  $t = Measure-Command { python -m slm train "configs/train/$c.yaml" @args25 2>"runs-$c.err" | Out-Host }
  Log "$c wall: $([math]::Round($t.TotalMinutes,1)) min"
}
