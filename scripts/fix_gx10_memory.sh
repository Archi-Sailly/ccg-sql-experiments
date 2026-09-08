#!/usr/bin/env bash
# =====================================================================
# fix_gx10_memory.sh — GX10(DGX Spark, CPU/GPU 통합 메모리) RAM 정리
#
# 하는 일 (순서대로):
#   1) vLLM(vllm-dept): --gpu-memory-utilization 을 모델 크기에 맞게 낮춰 재기동.
#      compose 파일을 직접 수정(백업 생성) → docker compose config 로 검증 → 재기동 →
#      기동 성공 로그 확인. 실패하면 백업으로 자동 롤백해 서비스를 원상복구.
#   2) Dify(/opt/dify/docker): 최근 7일 사용 흔적(nginx 요청, 채팅 메시지) 확인 →
#      없으면 docker compose down. 데이터는 /opt/dify/docker/volumes 에 그대로 남음.
#   3) compose 프로젝트에 속하지 않은 고아 open-webui 컨테이너: 사용 흔적 없으면
#      정지 + 자동재시작 해제 (삭제하지 않음 → docker start 로 복구 가능)
#   4) 스왑에 밀려난 페이지를 RAM 으로 되돌림 (swapoff/swapon, sudo)
#
# 사용법 (GX10 에서, docker 그룹 사용자):
#   bash fix_gx10_memory.sh                 # 전체 실행 (시작 전 1회 확인 질문)
#   DRY_RUN=1 bash fix_gx10_memory.sh       # 아무것도 바꾸지 않고 계획/판정만 출력
#   UTIL=0.45 bash fix_gx10_memory.sh       # vLLM utilization 값을 직접 지정
#   MAX_MODEL_LEN=16384 bash fix_gx10_memory.sh   # (선택) 최대 컨텍스트도 축소
#   FORCE_DIFY=1  사용 흔적이 있어도 Dify 내림 / SKIP_VLLM=1 SKIP_DIFY=1 SKIP_ORPHAN=1 / YES=1 확인 생략
# =====================================================================
set -uo pipefail

VLLM_CONTAINER="${VLLM_CONTAINER:-vllm-dept}"
DIFY_DIR="${DIFY_DIR:-/opt/dify/docker}"
ORPHAN="${ORPHAN:-open-webui}"
ORPHAN_REPLACEMENT="${ORPHAN_REPLACEMENT:-open-webui-dept}"
KV_GIB="${KV_GIB:-24}"            # KV 캐시에 남겨줄 메모리 (GiB)
OVERHEAD_GIB="${OVERHEAD_GIB:-6}" # 활성화값/CUDA graph/프로파일링 여유 (GiB)
ACTIVITY_DAYS="${ACTIVITY_DAYS:-7}"
ACTIVITY_THRESHOLD="${ACTIVITY_THRESHOLD:-20}"   # 7일간 실제 요청 수가 이보다 많으면 '사용 중'
WAIT_SECS="${WAIT_SECS:-900}"
DRY_RUN="${DRY_RUN:-0}"; YES="${YES:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"; SKIP_DIFY="${SKIP_DIFY:-0}"; SKIP_ORPHAN="${SKIP_ORPHAN:-0}"; FORCE_DIFY="${FORCE_DIFY:-0}"
UTIL="${UTIL:-}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$HOME/fix_gx10_memory_${TS}.log"
WORK="$(mktemp -d)"
exec > >(tee -a "$LOG") 2>&1

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
hr(){ printf '\n%s== %s ==%s\n' "$B" "$1" "$N"; }
info(){ printf '%s\n' "$*"; }
ok(){ printf '%s✓ %s%s\n' "$G" "$*" "$N"; }
warn(){ printf '%s! %s%s\n' "$Y" "$*" "$N"; }
die(){ printf '%s✗ %s%s\n' "$R" "$*" "$N"; exit 1; }
run(){ if [ "$DRY_RUN" = 1 ]; then printf '%s[DRY] %s%s\n' "$Y" "$*" "$N"; else printf '+ %s\n' "$*"; "$@"; fi; }
gib_from_kb(){ awk -v k="$1" 'BEGIN{printf "%.1f", k/1048576}'; }
mem_kb(){ awk -v key="$1" '$1==key":"{print $2}' /proc/meminfo; }
label(){ docker inspect --format "{{index .Config.Labels \"$2\"}}" "$1" 2>/dev/null; }

# ---- compose 파일 편집기 (python) ----
cat > "$WORK/edit_compose.py" <<'PYEOF'
#!/usr/bin/env python3
"""compose 파일에서 특정 서비스의 command 에 CLI 플래그 값을 설정(교체 또는 추가)한다.
사용: edit_compose.py <compose.yml> <service> <flag> <value>
- list 형태, 한 줄 문자열, 블록 스칼라(>, |) 모두 처리. 주석/서식은 보존.
- 성공 시 exit 0 + 'REPLACED' 또는 'APPENDED' 출력, 처리 불가 시 exit 2.
"""
import re
import sys


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def main() -> int:
    path, svc, flag, value = sys.argv[1:5]
    lines = open(path, encoding="utf-8").read().split("\n")

    # 1) services: 아래에서 서비스 블록 찾기
    svc_re = re.compile(rf"^(\s*){re.escape(svc)}\s*:\s*(#.*)?$")
    svc_idx = svc_ind = None
    for i, ln in enumerate(lines):
        m = svc_re.match(ln)
        if m and indent_of(ln) > 0:
            svc_idx, svc_ind = i, indent_of(ln)
            break
    if svc_idx is None:
        print(f"service '{svc}' not found", file=sys.stderr)
        return 2
    # 서비스 블록 끝
    end = len(lines)
    for j in range(svc_idx + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.lstrip().startswith("#") and indent_of(ln) <= svc_ind:
            end = j
            break

    # 2) 블록 안의 command: 찾기
    cmd_re = re.compile(r"^(\s*)command\s*:\s*(.*)$")
    cmd_idx = None
    child_ind_first = None
    for j in range(svc_idx + 1, end):
        if lines[j].strip() and not lines[j].lstrip().startswith("#"):
            child_ind_first = indent_of(lines[j]); break
    for j in range(svc_idx + 1, end):
        m = cmd_re.match(lines[j])
        if m and indent_of(lines[j]) == child_ind_first:
            cmd_idx, cmd_ind, rest = j, indent_of(lines[j]), m.group(2).strip()
            break
    if cmd_idx is None:
        # command 없음 → 서비스 블록 마지막에 리스트로 추가
        insert_at = end
        while insert_at > svc_idx + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        child_ind = None
        for j in range(svc_idx + 1, end):
            if lines[j].strip() and not lines[j].lstrip().startswith("#"):
                child_ind = indent_of(lines[j]); break
        child_ind = child_ind or svc_ind + 2
        pad = " " * child_ind
        lines[insert_at:insert_at] = [f"{pad}command:", f"{pad}  - {flag}={value}"]
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        print("APPENDED(new command)")
        return 0

    tok_re = re.compile(rf"(?<![\w-])({re.escape(flag)})([ =])(\"|')?[^\s\"']+(\"|')?")
    tok_re_val_only = re.compile(rf"(?<![\w-])({re.escape(flag)})(?![\w=-])")

    def sub_in_string(s: str):
        new, n = tok_re.subn(
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{value}{m.group(4) or ''}", s
        )
        return new, n

    # 2a) 한 줄 문자열 (command: --model x --port 8000 ...) 또는 flow list [..]
    if rest and rest[0] not in (">", "|"):
        new, n = sub_in_string(lines[cmd_idx])
        if n:
            lines[cmd_idx] = new; print("REPLACED(inline)")
        else:
            if rest.startswith("["):  # flow list
                if not rest.rstrip().endswith("]"):
                    print("multi-line flow list unsupported", file=sys.stderr); return 2
                body = rest.rstrip()[:-1]
                lines[cmd_idx] = lines[cmd_idx][: lines[cmd_idx].index(rest)] + body.rstrip().rstrip(",") + f', "{flag}={value}"]'
            else:
                # 따옴표로 감싼 문자열이면 닫는 따옴표 앞에 삽입
                stripped = lines[cmd_idx].rstrip()
                if stripped[-1] in ('"', "'") and rest[0] == stripped[-1]:
                    lines[cmd_idx] = stripped[:-1] + f" {flag} {value}" + stripped[-1]
                else:
                    lines[cmd_idx] = stripped + f" {flag} {value}"
            print("APPENDED(inline)")
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        return 0

    # 2b) 블록 스칼라 (> 또는 |) — 이어지는 더 깊은 들여쓰기 줄들이 본문
    if rest and rest[0] in (">", "|"):
        j = cmd_idx + 1
        last = cmd_idx
        body_ind = None
        while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > cmd_ind):
            if lines[j].strip():
                last = j
                body_ind = indent_of(lines[j]) if body_ind is None else body_ind
            j += 1
        replaced = 0
        for k in range(cmd_idx + 1, last + 1):
            new, n = sub_in_string(lines[k])
            if n:
                lines[k] = new; replaced += n
        if replaced:
            print("REPLACED(block)")
        else:
            body_ind = body_ind or cmd_ind + 2
            # 마지막 줄 끝에 백슬래시 연속 여부 확인
            uses_bs = any(lines[k].rstrip().endswith("\\") for k in range(cmd_idx + 1, last + 1))
            if uses_bs and not lines[last].rstrip().endswith("\\"):
                lines[last] = lines[last].rstrip() + " \\"
            lines.insert(last + 1, " " * body_ind + f"{flag} {value}")
            print("APPENDED(block)")
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        return 0

    # 2c) 블록 리스트 (다음 줄부터 '- ' 항목)
    j = cmd_idx + 1
    items = []
    while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("#") or indent_of(lines[j]) > cmd_ind):
        if lines[j].lstrip().startswith("- "):
            items.append(j)
        j += 1
    if not items:
        print("command has no value", file=sys.stderr); return 2
    item_ind = indent_of(lines[items[0]])
    replaced = 0
    for n_i, k in enumerate(items):
        ln = lines[k]
        val = ln.lstrip()[2:].strip().strip("\"'")
        if val == flag and n_i + 1 < len(items):  # ['--flag', '0.9'] 형태
            nxt = items[n_i + 1]
            q = '"' if '"' in lines[nxt] else ("'" if "'" in lines[nxt] else "")
            lines[nxt] = " " * item_ind + f"- {q}{value}{q}"
            replaced += 1
        else:
            new, n = sub_in_string(ln)
            if n:
                lines[k] = new; replaced += n
    if replaced:
        print("REPLACED(list)")
    else:
        lines.insert(items[-1] + 1, " " * item_ind + f"- {flag}={value}")
        print("APPENDED(list)")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF

cat > "$WORK/verify_cmd.py" <<'PYEOF'
import json, re, shlex, sys
svc, flag, value = sys.argv[1:4]
data = sys.stdin.read()
def has(tokens):
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens) and str(tokens[i + 1]) == value: return True
        if t == f"{flag}={value}": return True
    return False
try:
    cfg = json.loads(data)
    cmd = cfg["services"][svc].get("command")
    toks = cmd if isinstance(cmd, list) else shlex.split(cmd or "")
    sys.exit(0 if has([str(x) for x in toks]) else 1)
except Exception:
    flat = re.sub(r"\s+", " ", data)
    sys.exit(0 if re.search(re.escape(flag) + r"[ =\"'-]*" + re.escape(value) + r"(?![0-9])", flat) else 1)
PYEOF

# 컨테이너 라벨로 compose 호출 인자 구성 → 전역 CP_* / COMPOSE 배열
compose_from_container(){
  local c="$1"
  CP_PROJECT="$(label "$c" com.docker.compose.project)"
  CP_DIR="$(label "$c" com.docker.compose.project.working_dir)"
  CP_SERVICE="$(label "$c" com.docker.compose.service)"
  local files; files="$(label "$c" com.docker.compose.project.config_files)"
  IFS=',' read -r -a CP_FILES <<< "$files"
  [ -n "$CP_PROJECT" ] && [ -n "$CP_DIR" ] && [ ${#CP_FILES[@]} -gt 0 ] || return 1
  COMPOSE=(docker compose -p "$CP_PROJECT" --project-directory "$CP_DIR")
  local f; for f in "${CP_FILES[@]}"; do COMPOSE+=(-f "$f"); done
  return 0
}

# 최근 N일 로그에서 실제 HTTP 요청 수 (health/ping 제외, 최대 5000줄까지만 셈)
http_activity(){
  local c="$1" n
  n=$(docker logs --since "$((ACTIVITY_DAYS*24))h" "$c" 2>&1 \
      | grep -E '"(GET|POST|PUT|DELETE|PATCH) ' | grep -vE '/health|/ping|favicon|/api/system|/metrics' \
      | head -n 5000 | wc -l)
  echo "${n:-0}"
}

# ============================== 사전 점검 ==============================
hr "0. 사전 점검"
command -v docker >/dev/null || die "docker 명령이 없습니다"
docker info >/dev/null 2>&1 || die "docker 데몬에 접근할 수 없습니다 (docker 그룹 또는 sudo 필요)"
command -v python3 >/dev/null || die "python3 가 필요합니다"
TOTAL_KB=$(mem_kb MemTotal); TOTAL_GIB=$(gib_from_kb "$TOTAL_KB")
info "호스트: $(hostname) | 총 메모리 ${TOTAL_GIB} GiB | 로그: $LOG"
[ "$DRY_RUN" = 1 ] && warn "DRY_RUN=1: 실제 변경 없이 계획만 출력합니다"
info; info "-- 시작 시점 메모리 --"; free -h
BEFORE_USED_KB=$(( TOTAL_KB - $(mem_kb MemAvailable) ))

if [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  info
  info "계획: (1) $VLLM_CONTAINER 재기동(1~3분 중단) (2) Dify 사용 흔적 확인 후 down (3) 고아 $ORPHAN 정지 (4) 스왑 정리"
  read -r -p "계속할까요? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || die "사용자 취소"
fi

# ============================== 1. vLLM ==============================
VLLM_RESULT="건너뜀"
if [ "$SKIP_VLLM" = 1 ]; then
  hr "1. vLLM — SKIP_VLLM=1 로 건너뜀"
elif ! docker inspect "$VLLM_CONTAINER" >/dev/null 2>&1; then
  hr "1. vLLM — 컨테이너 '$VLLM_CONTAINER' 없음, 건너뜀"
else
  hr "1. vLLM ($VLLM_CONTAINER) — gpu-memory-utilization 조정"
  compose_from_container "$VLLM_CONTAINER" || die "compose 라벨을 읽을 수 없습니다 (compose 로 띄운 컨테이너가 아님?)"
  info "프로젝트=$CP_PROJECT 서비스=$CP_SERVICE 디렉터리=$CP_DIR 파일=${CP_FILES[*]}"
  VLLM_COMPOSE_STR="${COMPOSE[*]}"

  CMD_JSON="$(docker inspect --format '{{json .Config.Cmd}}' "$VLLM_CONTAINER")"
  CUR_UTIL="$(printf '%s' "$CMD_JSON" | python3 -c '
import json,sys
a=json.load(sys.stdin) or []
v="0.9(기본값)"
for i,t in enumerate(a):
    if t=="--gpu-memory-utilization" and i+1<len(a): v=a[i+1]
    elif t.startswith("--gpu-memory-utilization="): v=t.split("=",1)[1]
print(v)')"
  MODEL="$(printf '%s' "$CMD_JSON" | python3 -c '
import json,sys
a=json.load(sys.stdin) or []
m="?"
for i,t in enumerate(a):
    if t=="--model" and i+1<len(a): m=a[i+1]
    elif t.startswith("--model="): m=t.split("=",1)[1]
if m=="?":
    for t in a:
        if "/" in t and not t.startswith("-"): m=t; break
print(m)')"
  info "현재 command: $CMD_JSON"
  info "모델: $MODEL | 현재 utilization: $CUR_UTIL"

  info; info "-- vLLM 기동 로그 (메모리 관련) --"
  docker logs "$VLLM_CONTAINER" 2>&1 | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|max_model_len|gpu_memory_utilization' | tail -6
  WEIGHTS_GIB="$(docker logs "$VLLM_CONTAINER" 2>&1 | grep -oE 'Model loading took [0-9.]+ GiB' | tail -1 | grep -oE '[0-9.]+' | head -1 || true)"

  if [ -z "$UTIL" ]; then
    if [ -n "$WEIGHTS_GIB" ]; then
      UTIL="$(python3 -c "
import math
w,kv,ov,t=float('$WEIGHTS_GIB'),float('$KV_GIB'),float('$OVERHEAD_GIB'),float('$TOTAL_GIB')
u=math.ceil((w+kv+ov)/t*20)/20
print('%.2f'%min(max(u,0.25),0.85))")"
      info "가중치 ${WEIGHTS_GIB} GiB + KV ${KV_GIB} GiB + 여유 ${OVERHEAD_GIB} GiB → utilization ${UTIL} (총 ${TOTAL_GIB} GiB 기준)"
    else
      UTIL=0.5
      warn "로그에서 가중치 크기를 못 찾아 기본값 UTIL=0.5 사용. 기동 실패 시 UTIL=0.6 등으로 재실행하세요"
    fi
  else
    info "UTIL=$UTIL (사용자 지정)"
  fi
  BUDGET_GIB="$(python3 -c "print('%.0f'%(float('$UTIL')*float('$TOTAL_GIB')))")"
  info "→ vLLM 총 예산: 약 ${BUDGET_GIB} GiB (이전: $(python3 -c "
c='$CUR_UTIL'.split('(')[0]
try: print('%.0f GiB'%(float(c)*float('$TOTAL_GIB')))
except: print('?')"))"
  awk -v u="$UTIL" 'BEGIN{if (u+0>=0.8) print "! 모델이 커서 절감 폭이 작습니다. FP8/AWQ 양자화 모델로 바꾸거나 MAX_MODEL_LEN 축소를 권장합니다"}'

  # 편집 대상 파일: 서비스 정의가 있는 마지막 compose 파일 (override 가 있으면 그것)
  TARGET=""
  for (( i=${#CP_FILES[@]}-1; i>=0; i-- )); do
    if grep -qE "^[[:space:]]+${CP_SERVICE}[[:space:]]*:" "${CP_FILES[$i]}" 2>/dev/null; then TARGET="${CP_FILES[$i]}"; break; fi
  done
  [ -n "$TARGET" ] || die "서비스 '$CP_SERVICE' 정의를 compose 파일에서 찾지 못했습니다"
  BACKUP="${TARGET}.bak.${TS}"
  cp -p "$TARGET" "$BACKUP" || die "백업 실패: $BACKUP"
  ok "백업: $BACKUP"

  python3 "$WORK/edit_compose.py" "$TARGET" "$CP_SERVICE" --gpu-memory-utilization "$UTIL" \
    || { cp -p "$BACKUP" "$TARGET"; die "compose 파일 자동 편집 실패 — 수동으로 command 에 --gpu-memory-utilization $UTIL 추가 후 재기동하세요"; }
  if [ -n "$MAX_MODEL_LEN" ]; then
    python3 "$WORK/edit_compose.py" "$TARGET" "$CP_SERVICE" --max-model-len "$MAX_MODEL_LEN" || warn "max-model-len 편집 실패(무시)"
  fi
  info; info "-- 변경 diff --"; diff -u "$BACKUP" "$TARGET" || true

  # 렌더링 검증 (docker compose 가 실제로 읽는 결과 기준; JSON 우선, YAML 폴백)
  RENDERED="$("${COMPOSE[@]}" config --format json 2>"$WORK/config.err")" \
    || RENDERED="$("${COMPOSE[@]}" config 2>"$WORK/config.err")" \
    || { cat "$WORK/config.err"; cp -p "$BACKUP" "$TARGET"; die "docker compose config 실패 → 원본 복구함"; }
  if printf '%s' "$RENDERED" | python3 "$WORK/verify_cmd.py" "$CP_SERVICE" --gpu-memory-utilization "$UTIL"; then
    ok "docker compose config 검증 통과 (utilization=${UTIL})"
  else
    printf '%s\n' "$RENDERED" | grep -nE -- 'command|gpu-memory' | head; cp -p "$BACKUP" "$TARGET"
    die "렌더링된 설정에 새 값이 반영되지 않아 원본 복구함. compose 파일의 command 를 수동 확인하세요"
  fi

  if [ "$DRY_RUN" = 1 ]; then
    cp -p "$BACKUP" "$TARGET"; rm -f "$BACKUP"
    warn "[DRY] 여기서 재기동했을 것입니다 (파일은 원상복구함)"
    VLLM_RESULT="DRY: ${CUR_UTIL} → ${UTIL}"
  else
    run "${COMPOSE[@]}" stop "$CP_SERVICE"
    sleep 3; info; info "-- vLLM 정지 후 --"; free -h
    # 여유가 생긴 지금 스왑 복귀 + 캐시 정리 (vLLM 이 시작 시 '여유 메모리' 를 검사하므로)
    SWAP_USED_KB=$(awk '$1=="SwapTotal:"{t=$2}$1=="SwapFree:"{f=$2}END{print t-f}' /proc/meminfo)
    if [ "${SWAP_USED_KB:-0}" -gt 0 ] && [ "$(mem_kb MemAvailable)" -gt $((SWAP_USED_KB + 4*1048576)) ]; then
      info "스왑 $(gib_from_kb "$SWAP_USED_KB") GiB → RAM 복귀 (sudo)"; sudo swapoff -a && sudo swapon -a || warn "swapoff 실패(무시)"
    fi
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || warn "drop_caches 실패(무시)"

    START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run "${COMPOSE[@]}" up -d --force-recreate --no-deps "$CP_SERVICE" || { cp -p "$BACKUP" "$TARGET"; "${COMPOSE[@]}" up -d --force-recreate --no-deps "$CP_SERVICE"; die "up 실패 → 원본 설정으로 복구 기동함"; }
    NEWC="$(docker ps -aq --filter "label=com.docker.compose.project=$CP_PROJECT" --filter "label=com.docker.compose.service=$CP_SERVICE" | head -1)"
    info "새 컨테이너: $(docker inspect --format '{{.Name}}' "$NEWC" | sed 's#^/##') — 기동 대기 (최대 ${WAIT_SECS}s)"
    STATE=""; T0=$(date +%s)
    while :; do
      ST="$(docker inspect --format '{{.State.Status}}' "$NEWC" 2>/dev/null || echo missing)"
      LOGS="$(docker logs --since "$START_TS" "$NEWC" 2>&1)"
      if printf '%s' "$LOGS" | grep -qE 'Application startup complete|Uvicorn running on'; then STATE=ok; break; fi
      if [ "$ST" != running ]; then STATE=fail; break; fi
      if printf '%s' "$LOGS" | grep -qE 'Free memory on device|No available memory for the cache blocks|CUDA out of memory|Engine core initialization failed|EngineCore failed|Traceback \(most recent call last\)'; then
        sleep 15
        printf '%s' "$(docker logs --since "$START_TS" "$NEWC" 2>&1)" | grep -qE 'Application startup complete|Uvicorn running on' && { STATE=ok; break; }
        STATE=fail; break
      fi
      EL=$(( $(date +%s) - T0 )); [ $EL -ge "$WAIT_SECS" ] && { STATE=timeout; break; }
      [ $((EL % 30)) -eq 0 ] && info "  ... ${EL}s 경과 (상태: $ST)"
      sleep 5
    done
    if [ "$STATE" = ok ]; then
      ok "vLLM 기동 완료 (utilization ${CUR_UTIL} → ${UTIL})"
      docker logs --since "$START_TS" "$NEWC" 2>&1 | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency' | tail -4
      VLLM_RESULT="${CUR_UTIL} → ${UTIL} (예산 ~${BUDGET_GIB} GiB)"
    else
      warn "vLLM 기동 실패/타임아웃 (${STATE}). 마지막 로그:"; docker logs --since "$START_TS" "$NEWC" 2>&1 | tail -40
      cp -p "$BACKUP" "$TARGET"
      "${COMPOSE[@]}" up -d --force-recreate --no-deps "$CP_SERVICE"
      NEXT="$(python3 -c "print('%.2f'%min(float('$UTIL')+0.1,0.9))")"
      die "원본 설정(${CUR_UTIL})으로 롤백 기동했습니다. 예산이 부족했을 가능성이 크니 'UTIL=${NEXT} bash $0' 으로 재시도하세요"
    fi
    info; info "-- vLLM 재기동 후 --"; free -h
  fi
fi

# ============================== 2. Dify ==============================
DIFY_RESULT="건너뜀"
if [ "$SKIP_DIFY" = 1 ]; then
  hr "2. Dify — SKIP_DIFY=1 로 건너뜀"
else
  hr "2. Dify ($DIFY_DIR) — 사용 흔적 확인 후 down"
  mapfile -t DIFY_CS < <(docker ps -aq --filter "label=com.docker.compose.project.working_dir=$DIFY_DIR")
  if [ ${#DIFY_CS[@]} -eq 0 ]; then
    info "해당 디렉터리로 띄운 컨테이너가 없습니다 — 이미 내려간 상태"
    DIFY_RESULT="없음(이미 내려감)"
  else
    ANCHOR="${DIFY_CS[0]}"
    for c in "${DIFY_CS[@]}"; do [ "$(label "$c" com.docker.compose.service)" = api ] && ANCHOR="$c"; done
    compose_from_container "$ANCHOR" || die "Dify compose 라벨을 읽을 수 없습니다"
    info "프로젝트=$CP_PROJECT 컨테이너 ${#DIFY_CS[@]}개 (파일: ${CP_FILES[*]})"
    docker ps -a --filter "label=com.docker.compose.project.working_dir=$DIFY_DIR" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

    NGINX_C=""; PG_C=""
    for c in "${DIFY_CS[@]}"; do
      s="$(label "$c" com.docker.compose.service)"
      [ "$s" = nginx ] && NGINX_C="$c"
      { [ "$s" = db_postgres ] || [ "$s" = db ]; } && PG_C="$c"
    done
    REQ=0; RECENT_MSGS=0; LAST_MSG="?"; ACCOUNTS="?"
    if [ -n "$NGINX_C" ]; then REQ="$(http_activity "$NGINX_C")"; fi
    if [ -n "$PG_C" ] && [ "$(docker inspect --format '{{.State.Running}}' "$PG_C")" = true ]; then
      PGU="$(grep -E '^POSTGRES_USER=' "$DIFY_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed "s/[\"']//g")"; PGU="${PGU:-postgres}"
      PGD="$(grep -E '^POSTGRES_DB=' "$DIFY_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed "s/[\"']//g")"; PGD="${PGD:-dify}"
      LAST_MSG="$(docker exec "$PG_C" psql -U "$PGU" -d "$PGD" -tAc "select coalesce(max(created_at)::text,'없음') from messages" 2>/dev/null || echo '조회실패')"
      RECENT_MSGS="$(docker exec "$PG_C" psql -U "$PGU" -d "$PGD" -tAc "select count(*) from messages where created_at > now() - interval '${ACTIVITY_DAYS} days'" 2>/dev/null || echo 0)"
      ACCOUNTS="$(docker exec "$PG_C" psql -U "$PGU" -d "$PGD" -tAc "select count(*) from accounts" 2>/dev/null || echo '?')"
    fi
    info
    info "최근 ${ACTIVITY_DAYS}일 nginx 실제 요청: ${REQ}건 (기준 ${ACTIVITY_THRESHOLD}) | 최근 ${ACTIVITY_DAYS}일 채팅 메시지: ${RECENT_MSGS}건 | 마지막 메시지: ${LAST_MSG} | 계정 수: ${ACCOUNTS}"
    IN_USE=0; { [ "${REQ:-0}" -gt "$ACTIVITY_THRESHOLD" ] || [ "${RECENT_MSGS:-0}" -gt 0 ]; } 2>/dev/null && IN_USE=1
    if [ "$IN_USE" = 1 ] && [ "$FORCE_DIFY" != 1 ]; then
      warn "최근 사용 흔적이 있어 Dify 는 건너뜁니다. 그래도 내리려면 FORCE_DIFY=1 로 재실행하세요"
      DIFY_RESULT="사용 흔적 있음 → 유지"
    else
      [ "$IN_USE" = 1 ] && warn "FORCE_DIFY=1: 사용 흔적이 있지만 내립니다"
      run "${COMPOSE[@]}" down --remove-orphans
      # profile 밖 잔여 컨테이너 정리 (데이터는 bind mount 라 보존됨)
      for c in $(docker ps -aq --filter "label=com.docker.compose.project.working_dir=$DIFY_DIR"); do
        run docker stop "$c"; run docker rm "$c"
      done
      [ "$DRY_RUN" = 1 ] && DIFY_RESULT="DRY: down 대상 (${#DIFY_CS[@]}개)" || DIFY_RESULT="down 완료 (${#DIFY_CS[@]}개 제거, 데이터 보존)"
      [ -d "$DIFY_DIR/volumes" ] && info "데이터 보존 위치: $DIFY_DIR/volumes ($(du -sh "$DIFY_DIR/volumes" 2>/dev/null | cut -f1)) — 복구: cd $DIFY_DIR && docker compose up -d"
    fi
  fi
fi

# ============================== 3. 고아 open-webui ==============================
ORPHAN_RESULT="건너뜀"
if [ "$SKIP_ORPHAN" = 1 ]; then
  hr "3. 고아 컨테이너 — SKIP_ORPHAN=1 로 건너뜀"
elif ! docker inspect "$ORPHAN" >/dev/null 2>&1; then
  hr "3. 고아 컨테이너 '$ORPHAN' 없음, 건너뜀"; ORPHAN_RESULT="없음"
else
  hr "3. 고아 컨테이너 ($ORPHAN)"
  OP="$(label "$ORPHAN" com.docker.compose.project)"
  docker inspect --format 'image={{.Config.Image}} 생성={{.Created}} restart={{.HostConfig.RestartPolicy.Name}} 상태={{.State.Status}} ports={{json .HostConfig.PortBindings}}' "$ORPHAN"
  docker inspect --format '{{range .Mounts}}mount: {{.Type}} {{.Name}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' "$ORPHAN"
  if [ -n "$OP" ]; then
    warn "compose 프로젝트 '$OP' 소속이라 고아가 아닙니다 — 건너뜀"; ORPHAN_RESULT="compose 소속($OP) → 유지"
  else
    docker inspect "$ORPHAN_REPLACEMENT" >/dev/null 2>&1 && info "대체 서비스 $ORPHAN_REPLACEMENT 존재함"
    OREQ="$(http_activity "$ORPHAN")"
    info "최근 ${ACTIVITY_DAYS}일 실제 요청: ${OREQ}건 (기준 ${ACTIVITY_THRESHOLD})"
    if [ "${OREQ:-0}" -gt "$ACTIVITY_THRESHOLD" ]; then
      warn "사용 흔적이 있어 유지합니다"; ORPHAN_RESULT="사용 흔적 있음 → 유지"
    else
      run docker update --restart=no "$ORPHAN"
      run docker stop "$ORPHAN"
      ORPHAN_RESULT="정지 + 자동재시작 해제 (복구: docker update --restart=unless-stopped $ORPHAN && docker start $ORPHAN)"
    fi
  fi
fi

# ============================== 4. 나머지 프로젝트 활동 현황 (참고용, 변경 없음) ==============================
hr "4. 나머지 컨테이너 활동 현황 (변경하지 않음 — 판단 참고용)"
printf '%-28s %-16s %10s  %s\n' "컨테이너" "프로젝트" "메모리" "최근 ${ACTIVITY_DAYS}일 HTTP 요청"
for c in $(docker ps --format '{{.Names}}'); do
  p="$(label "$c" com.docker.compose.project)"
  case "$c" in "$VLLM_CONTAINER"|"$ORPHAN") continue;; esac
  [ "$(label "$c" com.docker.compose.project.working_dir)" = "$DIFY_DIR" ] && continue
  m="$(docker stats --no-stream --format '{{.MemUsage}}' "$c" 2>/dev/null | cut -d/ -f1 | tr -d ' ')"
  a="$(http_activity "$c")"; [ "$a" -ge 5000 ] && a="5000+"
  printf '%-28s %-16s %10s  %s\n' "$c" "${p:-(단독)}" "$m" "$a"
done
info "→ 요청 0건이 계속되는 것은 'docker compose -p <프로젝트> stop' 으로 내려도 됩니다 (DB/redis 는 앱과 함께 판단)"

# ============================== 5. 스왑 정리 ==============================
hr "5. 스왑 정리"
SWAP_USED_KB=$(awk '$1=="SwapTotal:"{t=$2}$1=="SwapFree:"{f=$2}END{print t-f}' /proc/meminfo)
if [ "${SWAP_USED_KB:-0}" -gt $((512*1024)) ]; then
  if [ "$(mem_kb MemAvailable)" -gt $((SWAP_USED_KB + 4*1048576)) ]; then
    info "스왑 $(gib_from_kb "$SWAP_USED_KB") GiB 사용 중 → RAM 으로 복귀 (sudo)"
    run sudo swapoff -a && run sudo swapon -a || warn "swapoff 실패(무시)"
  else
    warn "가용 메모리가 부족해 스왑 복귀는 생략 (스왑 $(gib_from_kb "$SWAP_USED_KB") GiB)"
  fi
else
  ok "스왑 사용량 미미 ($(gib_from_kb "${SWAP_USED_KB:-0}") GiB)"
fi

# ============================== 요약 ==============================
hr "요약"
free -h
AFTER_USED_KB=$(( TOTAL_KB - $(mem_kb MemAvailable) ))
info
info "실사용(총-가용): $(gib_from_kb "$BEFORE_USED_KB") GiB → $(gib_from_kb "$AFTER_USED_KB") GiB"
info "vLLM   : $VLLM_RESULT"
info "Dify   : $DIFY_RESULT"
info "고아    : $ORPHAN_RESULT"
info
docker compose ls -a
info
docker stats --no-stream --format '{{.MemUsage}}\t{{.Name}}' | sort -h -r | head -25
info
info "롤백 방법:"
[ -n "${BACKUP:-}" ] && [ -f "${BACKUP:-/nonexistent}" ] && info "  vLLM : cp '$BACKUP' '$TARGET' && ${VLLM_COMPOSE_STR:-docker compose} up -d --force-recreate"
info "  Dify : cd $DIFY_DIR && docker compose up -d"
info "  $ORPHAN: docker update --restart=unless-stopped $ORPHAN && docker start $ORPHAN"
info "로그 파일: $LOG"
rm -rf "$WORK"
