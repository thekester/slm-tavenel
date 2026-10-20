# Offline checks for queue helpers. Does not execute the queue or launch Python.
$ErrorActionPreference = "Stop"
$queuePath = Join-Path $PSScriptRoot "../scripts/resume_queue.ps1"
$queueTokens = $null
$queueErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $queuePath, [ref]$queueTokens, [ref]$queueErrors)
if ($queueErrors.Count) { throw ($queueErrors | Out-String) }
foreach ($name in "Invoke-Python", "ValidJson", "Evaluated") {
  $definition = $ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
  }, $true) | Where-Object Name -eq $name
  if (-not $definition) { throw "Missing helper: $name" }
  . ([scriptblock]::Create($definition.Extent.Text))
}

$testFile = [System.IO.Path]::GetTempFileName()
try {
  if (ValidJson $testFile) { throw "Empty output marked complete" }
  Set-Content -LiteralPath $testFile -Value '{"partial":' -Encoding utf8
  if (ValidJson $testFile) { throw "Truncated JSON marked complete" }
  Set-Content -LiteralPath $testFile -Value '{"items": 12}' -Encoding utf8
  if (-not (ValidJson $testFile)) { throw "Valid output rejected" }
} finally {
  Remove-Item -LiteralPath $testFile
}
if (ValidJson $testFile) { throw "Missing output marked complete" }

function ValidJson($path) { return $script:completed.ContainsKey($path) }
$script:completed = @{ "runs/demo/analysis/mc.json" = $true }
if (Evaluated "demo") { throw "Benchmark output alone marked the run complete" }
foreach ($name in "bpb-en.json", "bpb-fr.json", "sinks-seed1.json") {
  $script:completed["runs/demo/analysis/$name"] = $true
}
if (-not (Evaluated "demo")) { throw "Complete evaluation rejected" }

function python { $global:LASTEXITCODE = $script:mockExitCode; "stub-output" }
$script:mockExitCode = 0
if ((Invoke-Python -m slm) -ne "stub-output") { throw "Successful command output lost" }
$script:mockExitCode = 7
$caught = $false
try { Invoke-Python -m slm 2>$null | Out-Null } catch { $caught = $true }
if (-not $caught) { throw "Failed command did not stop execution" }
$global:LASTEXITCODE = 0
"Queue helper checks passed (no jobs launched)"
