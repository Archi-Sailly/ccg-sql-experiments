#!/usr/bin/env bash
# ram_diag.sh — RAM 사용 원인 진단 (읽기 전용: 아무것도 중지/삭제/변경하지 않음)
#
# 사용법 (RAM이 높은 그 머신에서):
#   bash ram_diag.sh 2>&1 | tee ram_diag_$(hostname)_$(date +%Y%m%d_%H%M).txt
#
# Windows + Docker Desktop(WSL2) 환경이면:
#   1) PowerShell:  Get-Process | Sort WS -Desc | Select -First 15 Name,@{n='GB';e={[math]::Round($_.WS/1GB,2)}}
#      -> "vmmem" / "vmmemWSL" 이 크면 WSL2(Docker Desktop) 가 RAM을 먹는 것
#   2) 그 다음 WSL 안에서(wsl -d Ubuntu) 이 스크립트를 실행
set -u
hr(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1; }
col(){ if have column; then column -t -s $'\t'; else cat; fi; }
SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo && sudo -n true 2>/dev/null; then SUDO="sudo -n"; fi

# ---------- macOS 분기 ----------
if [ "$(uname -s)" = "Darwin" ]; then
  hr "macOS 메모리"
  sysctl -n hw.memsize | awk '{printf "총 메모리: %.1f GiB\n",$1/1073741824}'
  vm_stat | head -12
  hr "프로세스 메모리 Top 20"
  top -l 1 -o mem -n 20 -stats pid,command,mem,cpu | tail -22
  hr "Docker"
  have docker && docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' 2>&1
  have docker && docker compose ls -a 2>&1
  have ollama && { echo "-- ollama ps --"; ollama ps 2>&1; }
  exit 0
fi

# ---------- Linux ----------
hr "0. 호스트"
uname -a
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo ">> WSL2 환경입니다. Windows 쪽 'vmmem' 프로세스 크기와 %UserProfile%\\.wslconfig 확인 필요"
fi
have lscpu && lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Hypervisor)'
have systemd-detect-virt && echo "가상화: $(systemd-detect-virt 2>/dev/null || echo none)"
uptime

hr "1. 메모리 총괄  (핵심: 'available' 이 크면 대부분 파일 캐시 → 실제 문제 아님)"
free -h
echo
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|Shmem|SReclaimable|SUnreclaim|AnonPages|Mapped|PageTables|KernelStack|HugePages_Total|Hugepagesize|Committed_AS)' /proc/meminfo
echo
awk '/^MemTotal/{t=$2}/^MemFree/{f=$2}/^MemAvailable/{a=$2}/^Cached/{c=$2}/^SReclaimable/{s=$2}/^AnonPages/{an=$2}/^Shmem:/{sh=$2}
END{printf "총: %.1f GiB | 프로세스 실사용(AnonPages): %.1f GiB | 공유메모리/tmpfs(Shmem): %.1f GiB | 캐시(회수 가능): %.1f GiB | 실제 여유(available): %.1f GiB\n",
t/1048576,an/1048576,sh/1048576,(c+s-sh)/1048576,a/1048576;
printf ">> 단순 (총-free) 기준 사용률: %.0f%%  vs  실제 압박 지표 (총-available): %.0f%%  → 두 값 차이가 크면 대부분 캐시\n",(t-f)/t*100,(t-a)/t*100}' /proc/meminfo

hr "2. 프로세스 RSS Top 25"
ps -eo pid,ppid,user,rss,pmem,pcpu,etime,args --sort=-rss | head -26 \
  | awk 'NR==1{print;next}{$4=sprintf("%.2fG",$4/1048576);print}' | cut -c1-180

hr "2b. 프로세스명별 RSS 합계 Top 15"
ps -eo rss,comm --no-headers | awk '{a[$2]+=$1}END{for(k in a)printf "%8.2f GiB  %s\n",a[k]/1048576,k}' | sort -rn | head -15

hr "3. Docker"
if have docker && $SUDO docker info >/dev/null 2>&1; then
  $SUDO docker info --format 'Server {{.ServerVersion}} | 컨테이너 {{.Containers}} (실행중 {{.ContainersRunning}}) | 이미지 {{.Images}} | cgroup {{.CgroupVersion}}' 2>&1
  echo; echo "-- compose 프로젝트 목록 (어떤 스택이 올라가 있는지) --"
  $SUDO docker compose ls -a 2>&1
  echo; echo "-- 컨테이너별 메모리 사용량 (큰 순) --"
  $SUDO docker stats --no-stream --format '{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}\t{{.Name}}' 2>&1 | sort -h -r | col
  echo; echo "-- 전체 컨테이너 (중지 포함) / 상태 / 재시작정책 / 메모리제한 / 소속 프로젝트 --"
  for c in $($SUDO docker ps -aq 2>/dev/null); do
    $SUDO docker inspect --format '{{.Name}}	{{.State.Status}}	restart={{.HostConfig.RestartPolicy.Name}}	mem_limit={{if .HostConfig.Memory}}{{.HostConfig.Memory}}{{else}}없음{{end}}	project={{index .Config.Labels "com.docker.compose.project"}}	dir={{index .Config.Labels "com.docker.compose.project.working_dir"}}' "$c" 2>/dev/null
  done | sed 's#^/##' | col
  echo; echo "-- Docker 디스크 --"; $SUDO docker system df 2>&1
else
  echo "docker 없음 또는 데몬 접근 불가 (sudo 필요할 수 있음)"
fi

hr "4. 로컬 LLM 런타임 (GPU 없는 머신에서 RAM 수십 GB를 점유하는 1순위 후보)"
found=0
for p in ollama vllm llama-server llama_cpp lmstudio xinference localai koboldcpp text-generation tgi sglang; do
  if pgrep -fa "$p" >/dev/null 2>&1; then found=1; pgrep -fa "$p" | cut -c1-150 | head -3; fi
done
[ $found -eq 0 ] && echo "호스트 프로세스에서 LLM 런타임 미발견 (컨테이너 안에 있을 수 있음 → 3번 docker stats 확인)"
if have ollama; then
  echo "-- ollama ps (현재 RAM에 올라간 모델) --"; ollama ps 2>&1
  echo "-- ollama 환경변수 --"; have systemctl && systemctl show ollama -p Environment --no-pager 2>/dev/null
fi
have nvidia-smi && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1

hr "5. 기타 상주 서비스"
if have systemctl; then
  systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | awk '{print $1}' \
    | grep -Ev '^(systemd-|dbus|getty|ssh|cron|rsyslog|polkit|udev|user@|accounts-daemon|ModemManager|snapd|unattended)' | col
fi
have kubectl && { echo "-- kubernetes pods --"; kubectl get pods -A 2>&1 | head -30; }
n=$(pgrep -fc 'jupyter|ipykernel_launcher' 2>/dev/null); echo "jupyter/ipykernel 프로세스 수: ${n:-0}"
echo "-- DB/벡터DB/검색엔진 프로세스 --"
pgrep -fa 'postgres|redis-server|mysqld|mariadbd|mongod|elasticsearch|opensearch|weaviate|qdrant|milvus|minio|clickhouse' 2>/dev/null | cut -c1-120 | head -15

hr "6. ZFS ARC / tmpfs / swap"
if [ -f /proc/spl/kstat/zfs/arcstats ]; then
  awk '/^size /{printf "ZFS ARC 현재: %.1f GiB\n",$3/1073741824}/^c_max /{printf "ZFS ARC 상한: %.1f GiB\n",$3/1073741824}' /proc/spl/kstat/zfs/arcstats
else echo "ZFS 없음"; fi
df -h -t tmpfs 2>/dev/null | awk 'NR==1||$3+0>0'
swapon --show 2>/dev/null || echo "swap 없음"

hr "완료 — 이 출력 전체를 Claude에게 붙여넣어 주세요"
