# Build the tokenizer and token shards only from the filtered wiki-v1 document set.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$docs = "data/processed/wiki-v1/documents.jsonl"
$tokenizerSample = "data/processed/wiki-v1/tokenizer-train-120k.jsonl"
$tokenizerManifest = "data/tokenizer/wiki-v1-clean/manifest.json"
$tokensManifest = "data/tokens/wiki-v1-clean/manifest.json"
if (-not (Test-Path -LiteralPath "data/processed/wiki-v1/decontamination-audit.json")) {
  throw "Run scripts/audit_wiki_v1_decontamination.py first"
}
if (-not (Test-Path -LiteralPath "data/processed/wiki-v1/bpb-holdout-manifest.json")) {
  throw "Prepare the fixed BPB holdout first"
}
if (-not (Test-Path -LiteralPath "data/processed/wiki-v1/tokenizer-train-sample-manifest.json")) {
  python scripts/prepare_wiki_v1_tokenizer_sample.py
  if ($LASTEXITCODE -ne 0) { throw "wiki-v1 tokenizer sample preparation failed" }
}
if (-not (Test-Path -LiteralPath $tokenizerManifest)) {
  python -m slm tokenizer train --inputs $tokenizerSample --out data/tokenizer/wiki-v1-clean --vocab-size 8192 --seed 0
  if ($LASTEXITCODE -ne 0) { throw "wiki-v1 tokenizer training failed" }
}
if (-not (Test-Path -LiteralPath $tokensManifest)) {
  python -m slm data encode --tokenizer data/tokenizer/wiki-v1-clean --inputs $docs --out data/tokens/wiki-v1-clean --val-fraction 0.01
  if ($LASTEXITCODE -ne 0) { throw "wiki-v1 token encoding failed" }
}
Write-Host "wiki-v1 clean tokenizer and shards are ready"
