param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

Push-Location $repoRoot
try {
    if (Test-Path $venvPython) {
        & $venvPython -m ollama_code.cli @CliArgs
        exit $LASTEXITCODE
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        & $py.Source -3 -m ollama_code.cli @CliArgs
        exit $LASTEXITCODE
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        & $python.Source -m ollama_code.cli @CliArgs
        exit $LASTEXITCODE
    }

    throw "Python was not found. Install Python or create .venv first."
}
finally {
    Pop-Location
}
