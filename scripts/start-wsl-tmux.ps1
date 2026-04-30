param(
    [string]$Session = "ollama-code",
    [string]$Model = ""
)

$repo = Split-Path -Parent $PSScriptRoot
$repoWsl = (wsl.exe wslpath -a "$repo").Trim()

$escapedSession = $Session.Replace("'", "'\"'\"'")
$escapedModel = $Model.Replace("'", "'\"'\"'")
$escapedRepo = $repoWsl.Replace("'", "'\"'\"'")

wsl.exe -e bash -lc "cd '$escapedRepo' && ./scripts/start-wsl-tmux.sh '$escapedSession' '$escapedModel' '$escapedRepo'"
