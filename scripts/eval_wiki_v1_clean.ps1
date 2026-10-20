# CPU evaluation companion for queue_wiki_v1_clean.ps1.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$root = "runs/wiki-v1-clean"
trap {
  Set-Content -LiteralPath "$root/eval-failed.txt" -Value "$(Get-Date -Format o) $_"
  Write-Error $_
  exit 1
}
$jobs = @("adamw-s1234", "gate-s1234", "muon-s1234", "adamw-s2", "gate-s2", "muon-s2",
          "adamw-s3", "gate-s3", "muon-s3")
$benchmarks = @("data/benchmarks/arc_easy.jsonl", "data/benchmarks/hellaswag.jsonl",
                "data/benchmarks/belebele_en.jsonl", "data/benchmarks/belebele_fr.jsonl")

function ValidJson([string]$path) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
  try {
    $value = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -ErrorAction Stop
    return $null -ne $value
  }
  catch { return $false }
}
function Log([string]$message) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
  Add-Content -LiteralPath "$root/eval.log" -Value $line
  Write-Host $line
}

while ($true) {
  if (Test-Path -LiteralPath "$root/queue-failed.txt") { throw (Get-Content -LiteralPath "$root/queue-failed.txt" -Raw) }
  $didWork = $false
  foreach ($name in $jobs) {
    $runDir = "$root/$name"
    $ck = "$runDir/checkpoints/step_00015000.pt"
    if (-not (Test-Path -LiteralPath $ck)) { continue }
    $analysis = "$runDir/analysis"
    New-Item -ItemType Directory -Force $analysis | Out-Null
    foreach ($language in @("en", "fr")) {
      $out = "$analysis/bpb-$language.json"
      if (ValidJson $out) { continue }
      Log "starting BPB $language for $name"
      & python -m slm eval bpb --device cpu --checkpoint $ck --tokenizer data/tokenizer/wiki-v1-clean `
        --data "data/processed/wiki-v1/bpb-val-$language.jsonl" 2>$null |
        Out-File -LiteralPath $out -Encoding utf8
      if ($LASTEXITCODE -ne 0 -or -not (ValidJson $out)) { throw "BPB $language failed for $name" }
      $didWork = $true
    }
    $mc = "$analysis/mc.json"
    if (-not (ValidJson $mc)) {
      Log "starting item-level benchmark evaluation for $name"
      & python -m slm eval mc --device cpu --checkpoint $ck --tokenizer data/tokenizer/wiki-v1-clean `
        --benchmarks @benchmarks --out $mc 2>$null | Out-Null
      if ($LASTEXITCODE -ne 0 -or -not (ValidJson $mc)) { throw "benchmark evaluation failed for $name" }
      $didWork = $true
    }
    $sinks = "$analysis/sinks-wiki-v1.json"
    if (-not (ValidJson $sinks)) {
      Log "starting sink analysis for $name"
      & python scripts/analyze_sinks.py $ck --data data/tokens/wiki-v1-clean --windows 128 --seq-len 256 --seed 1 `
        --out $sinks 2>$null | Out-Null
      if ($LASTEXITCODE -ne 0 -or -not (ValidJson $sinks)) { throw "sink analysis failed for $name" }
      $didWork = $true
    }
    $ready = (ValidJson "$analysis/bpb-en.json") -and (ValidJson "$analysis/bpb-fr.json") -and
             (ValidJson "$analysis/mc.json") -and (ValidJson "$analysis/sinks-wiki-v1.json")
    if ($ready -and -not (Test-Path -LiteralPath "$analysis/evaluation-complete.txt")) {
      Set-Content -LiteralPath "$analysis/evaluation-complete.txt" -Value "Completed $(Get-Date -Format o)"
      Log "completed all evaluations for $name"
    }
  }

  $allEvaluated = $true
  foreach ($name in $jobs) {
    if (-not (Test-Path -LiteralPath "$root/$name/analysis/evaluation-complete.txt")) { $allEvaluated = $false; break }
  }
  if ($allEvaluated -and (Test-Path -LiteralPath "$root/queue-complete.txt")) { break }
  Start-Sleep -Seconds 90
}

foreach ($seedSuffix in @("s1234", "s2", "s3")) {
  foreach ($treatment in @("gate", "muon")) {
    $baseline = "$root/adamw-$seedSuffix/analysis/mc.json"
    $treated = "$root/$treatment-$seedSuffix/analysis/mc.json"
    $out = "$root/paired-$treatment-$seedSuffix.json"
    & python scripts/compare_mc.py $baseline $treated --out $out 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "paired comparison failed for $treatment $seedSuffix" }
  }
}
Set-Content -LiteralPath "$root/evaluation-complete.txt" -Value "All paired evaluations completed at $(Get-Date -Format o)"
Log "all nine runs and paired benchmark comparisons complete"
