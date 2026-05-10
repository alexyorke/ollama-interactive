#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

workspace="${OLLAMA_SELF_WORKSPACE:-/workspace}"
state_dir="${OLLAMA_SELF_STATE_DIR:-${workspace}/.ollama-code/self-host}"
cli_loop="${OLLAMA_SELF_LOOP:-0}"
max_restarts="${OLLAMA_SELF_MAX_RESTARTS:-8}"
restart_delay="${OLLAMA_SELF_RESTART_DELAY:-2}"
rollback_on_crash="${OLLAMA_SELF_ROLLBACK:-1}"
checkpoint_on_success="${OLLAMA_SELF_CHECKPOINT_ON_SUCCESS:-0}"
health_check_cmd="${OLLAMA_SELF_HEALTH_CMD:-}"

good_head_file="${state_dir}/good_head.sha"
attempt_file="${state_dir}/attempt_count"
log_prefix="[self-host]"

mkdir -p "${state_dir}"
trap 'echo "${log_prefix} received signal, exiting"' EXIT

if ! git -C "${workspace}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "${log_prefix} workspace is not a git repository: ${workspace}" >&2
  exit 1
fi

if [ ! -f "${good_head_file}" ]; then
  git -C "${workspace}" rev-parse HEAD > "${good_head_file}"
fi

if [ "$#" -gt 0 ]; then
  cli_args=("$@")
else
  cli_args=(python -m ollama_code.cli --cwd "${workspace}")
fi

if ! command -v "${cli_args[0]}" >/dev/null 2>&1 && [ "${cli_args[0]}" != "python" ] && [ "${cli_args[0]}" != "python3" ]; then
  echo "${log_prefix} first CLI arg '${cli_args[0]}' is not executable" >&2
  exit 1
fi

run_health_check() {
  if [ -z "${health_check_cmd}" ]; then
    return 0
  fi

  bash -lc "${health_check_cmd}"
}

checkpoint_if_needed() {
  if [ "${checkpoint_on_success}" != "1" ]; then
    return 0
  fi

  if git -C "${workspace}" diff --quiet --ignore-submodules --; then
    return 0
  fi

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  git -C "${workspace}" add -A
  git -C "${workspace}" -c user.name="ollama self host" -c user.email="self-host@localhost" commit -m "self-host checkpoint ${timestamp}"
  git -C "${workspace}" rev-parse HEAD > "${good_head_file}"
  echo "0" > "${attempt_file}"
}

rollback_to_good() {
  if [ "${rollback_on_crash}" != "1" ]; then
    return 0
  fi

  good_head="$(cat "${good_head_file}")"
  current_head="$(git -C "${workspace}" rev-parse HEAD)"
  if [ "${good_head}" != "${current_head}" ]; then
    git -C "${workspace}" reset --hard "${good_head}"
    git -C "${workspace}" clean -fd
    echo "${log_prefix} rolled back ${current_head} -> ${good_head}"
  fi
}

restarter() {
  if [ "${cli_loop}" != "1" ]; then
    return 1
  fi

  if [ ! -f "${attempt_file}" ]; then
    echo "0" > "${attempt_file}"
  fi

  attempt="$(cat "${attempt_file}")"
  if [ "${attempt}" -ge "${max_restarts}" ]; then
    echo "${log_prefix} max restart attempts reached (${attempt}/${max_restarts})"
    return 1
  fi

  attempt=$((attempt + 1))
  echo "${attempt}" > "${attempt_file}"
  sleep "${restart_delay}"
  return 0
}

run_once() {
  echo "${log_prefix} starting: ${cli_args[*]}"
  set +e
  "${cli_args[@]}"
  cli_rc=$?
  set -e

  if [ "${cli_rc}" -ne 0 ]; then
    echo "${log_prefix} cli exited with ${cli_rc}"
    rollback_to_good
    if restarter; then
      return 0
    fi
    return 1
  fi

  echo "${log_prefix} cli exited cleanly"
  if ! run_health_check; then
    echo "${log_prefix} health check failed"
    rollback_to_good
    if restarter; then
      return 0
    fi
    return 1
  fi

  checkpoint_if_needed
  echo "0" > "${attempt_file}"
  return 1
}

while true; do
  run_once
  if [ $? -eq 1 ]; then
    break
  fi
done
