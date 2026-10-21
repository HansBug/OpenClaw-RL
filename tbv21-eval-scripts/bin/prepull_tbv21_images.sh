#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

TASKS_DIR="${1:-${TBV21_TASKS_DIR}}"
RETRIES="${TBV21_PREPULL_RETRIES:-5}"
RETRY_DELAY_S="${TBV21_PREPULL_RETRY_DELAY_S:-15}"
PULL_TIMEOUT_S="${TBV21_PREPULL_TIMEOUT_S:-900}"
REQUIRED="${TBV21_PREPULL_REQUIRED:-1}"
MIRROR_PREFIX="${TBV21_DOCKERHUB_MIRROR_PREFIX:-}"
MIRROR_FIRST="${TBV21_DOCKERHUB_MIRROR_FIRST:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
IMAGE_LIST="${TBV21_STATE_DIR}/tbv21_task_images_${TS}.tsv"
FAILED_LIST="${TBV21_STATE_DIR}/tbv21_task_images_failed_${TS}.tsv"
LOG_FILE="${TBV21_LOG_DIR}/prepull_tbv21_images_${TS}.log"

mkdir -p "${TBV21_STATE_DIR}" "${TBV21_LOG_DIR}"
touch "${LOG_FILE}" "${FAILED_LIST}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

if [ ! -d "${TASKS_DIR}" ]; then
  log "[ERROR] tasks dir missing: ${TASKS_DIR}"
  exit 1
fi

python3 - "${TASKS_DIR}" "${IMAGE_LIST}" <<'PY'
from pathlib import Path
import re
import sys

tasks_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for task_toml in sorted(tasks_dir.glob("*/task.toml")):
    text = task_toml.read_text(errors="replace")
    match = re.search(r'^\s*docker_image\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if match:
        rows.append((match.group(1), task_toml.parent.name))

seen = {}
for image, task in rows:
    seen.setdefault(image, []).append(task)

with out.open("w") as f:
    for image in sorted(seen):
        f.write(f"{image}\t{','.join(sorted(seen[image]))}\n")

print(f"task_images={len(rows)} unique_images={len(seen)} image_list={out}")
PY

log "docker_host=${DOCKER_HOST:-unset}"
if ! docker info >/dev/null 2>&1; then
  log "[ERROR] docker is not reachable; run bin/use_worker_rootless_docker.sh start first"
  exit 1
fi

docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}} images={{.Images}}'

mirror_image_for() {
  local image="$1"
  local first="${image%%/*}"
  if [ -z "${MIRROR_PREFIX}" ]; then
    return 1
  fi
  if [ "${image}" != "${first}" ] && { [[ "${first}" == *.* ]] || [[ "${first}" == *:* ]] || [ "${first}" = "localhost" ]; }; then
    return 1
  fi
  printf '%s/%s\n' "${MIRROR_PREFIX}" "${image}"
}

pull_one() {
  local image="$1"
  local mirror_image=""
  local attempt=1

  if docker image inspect "${image}" >/dev/null 2>&1; then
    log "[OK] already present: ${image}"
    return 0
  fi

  while [ "${attempt}" -le "${RETRIES}" ]; do
    if [ "${MIRROR_FIRST}" = "1" ] && mirror_image="$(mirror_image_for "${image}")"; then
      log "[INFO] mirror-first pull attempt ${attempt}/${RETRIES}: ${mirror_image}"
      if timeout "${PULL_TIMEOUT_S}" docker pull "${mirror_image}"; then
        docker tag "${mirror_image}" "${image}"
        log "[OK] pulled via mirror and tagged: ${mirror_image} -> ${image}"
        return 0
      fi
    fi

    log "[INFO] pull attempt ${attempt}/${RETRIES}: ${image}"
    if timeout "${PULL_TIMEOUT_S}" docker pull "${image}"; then
      log "[OK] pulled: ${image}"
      return 0
    fi

    if [ "${MIRROR_FIRST}" != "1" ] && mirror_image="$(mirror_image_for "${image}")"; then
      log "[INFO] mirror pull attempt ${attempt}/${RETRIES}: ${mirror_image}"
      if timeout "${PULL_TIMEOUT_S}" docker pull "${mirror_image}"; then
        docker tag "${mirror_image}" "${image}"
        log "[OK] pulled via mirror and tagged: ${mirror_image} -> ${image}"
        return 0
      fi
    fi

    if [ "${attempt}" -lt "${RETRIES}" ]; then
      sleep "${RETRY_DELAY_S}"
    fi
    attempt=$((attempt + 1))
  done

  return 1
}

total=0
ok=0
failed=0
while IFS=$'\t' read -r image tasks; do
  [ -n "${image}" ] || continue
  total=$((total + 1))
  if pull_one "${image}"; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    printf '%s\t%s\n' "${image}" "${tasks}" >> "${FAILED_LIST}"
    log "[ERROR] failed to pull: ${image} tasks=${tasks}"
  fi
done < "${IMAGE_LIST}"

log "prepull summary: total=${total} ok=${ok} failed=${failed}"
log "image_list=${IMAGE_LIST}"
log "failed_list=${FAILED_LIST}"

if [ "${failed}" -gt 0 ] && [ "${REQUIRED}" = "1" ]; then
  exit 1
fi
