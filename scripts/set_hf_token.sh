#!/usr/bin/env bash
# HF 토큰을 .env 에 넣고 그 자리에서 유효성까지 확인한다.
#
#   ./scripts/set_hf_token.sh hf_xxxxx      ← 끝에 붙여넣기 (제일 편함)
#   ./scripts/set_hf_token.sh               ← 인자 없이 실행하면 붙여넣고 Ctrl-D
#
# ⚠ 첫째 형태는 토큰이 명령줄에 남는다. `HISTCONTROL` 에 `ignorespace`(또는
#   `ignoreboth`)가 걸린 셸이면 **명령 앞 공백 한 칸**으로 히스토리를 건너뛴다.
#   확실히 하려면 둘째 형태를 쓸 것 — 명령줄을 아예 안 거친다.
set -u
umask 077          # .env 는 토큰을 담는다 — 만들어지는 순간부터 600 이어야 한다
cd "$(dirname "$0")/.." || exit 1

if [ $# -ge 1 ]; then
  TOK="$1"
else
  echo "토큰을 붙여넣고 Enter 후 Ctrl-D:"
  TOK="$(cat)"
fi

# 붙여넣기가 흔히 끌고 오는 것들을 턴다: 공백·따옴표·CR·줄바꿈
TOK="${TOK//[[:space:]]/}"; TOK="${TOK//\"/}"; TOK="${TOK//\'/}"
[ -n "$TOK" ] || { echo "✗ 토큰이 비었다"; exit 1; }

echo
echo "받은 값: ${#TOK}자 · 접두사 ${TOK:0:3}"

# 네트워크가 막히면 붙잡히지 않고 죽는다 — 사람이 기다리는 대화형 작업이다.
hf() { curl -s --connect-timeout 5 --max-time 20 -H "Authorization: Bearer $TOK" "$@"; }
name_of() { hf https://huggingface.co/api/whoami-v2 \
            | python3 -c "import json,sys;print(json.load(sys.stdin).get('name','?'))" 2>/dev/null; }

# --- 넣기 전에 먼저 물어본다. 틀린 토큰으로 .env 를 덮지 않기 위해서다 ---
# 게이트 응답 하나가 세 경우를 다 가른다: 401 무효 / 403 미승인 / 200 정상.
echo
echo "→ 넣기 전에 HuggingFace 에 먼저 확인한다"
GATE=$(hf -o /dev/null -w '%{http_code}' \
       https://huggingface.co/facebook/sam3/resolve/main/config.json)

# 계정 이름은 그것이 쓰이는 가지에서만 받는다. 이 왕복 하나는 **일부러 남겼다** —
# 공유 서버라 남의 토큰을 넣는 사고가 실제로 나고, 그때 이름이 유일한 단서다.
case "$GATE" in
  200) echo "✓ 토큰 유효 — 계정: $(name_of)"
       echo "✓ facebook/sam3 접근 가능"
       TAIL="✓ 준비 완료" ;;
  403) NAME=$(name_of)
       echo "✓ 토큰은 유효 — 계정: $NAME"
       echo "✗ facebook/sam3 게이트 미승인 (HTTP 403)"
       echo "  https://huggingface.co/facebook/sam3 에서 '$NAME' 계정으로 라이선스 동의할 것."
       echo "  fine-grained 토큰이면 'Read access to public gated repos' 체크도 확인."
       TAIL="△ 토큰은 됐지만 게이트가 남았다 (위 안내 참고)" ;;
  401) echo "✗ 토큰이 무효다 — .env 는 건드리지 않았다."
       echo "  https://huggingface.co/settings/tokens 에서 살아 있는지 확인하고 새로 발급할 것."
       exit 1 ;;
  000) echo "✗ HuggingFace 에 못 닿았다 (연결 실패 또는 시간초과) — .env 는 건드리지 않았다."
       exit 1 ;;
  *)   echo "⚠ 예상 밖 응답 HTTP $GATE — .env 는 건드리지 않았다."
       exit 1 ;;
esac

# --- 여기까지 왔으면 .env 에 넣는다. 다른 키는 보존한다 ---
{ grep -v '^[[:space:]]*#\?[[:space:]]*HF_TOKEN=' .env 2>/dev/null
  printf 'HF_TOKEN=%s\n' "$TOK"; } > .env.tmp && mv .env.tmp .env

echo
echo "✓ .env 에 기록 (권한 600, .gitignore 에 있어 커밋되지 않는다)"
echo "$TAIL"
