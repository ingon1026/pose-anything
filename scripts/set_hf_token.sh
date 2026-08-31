#!/usr/bin/env bash
# HF 토큰을 .env 에 넣고 그 자리에서 유효성까지 확인한다.
#
#   ./scripts/set_hf_token.sh hf_xxxxx      ← 끝에 붙여넣기 (제일 편함)
#   ./scripts/set_hf_token.sh               ← 인자 없이 실행하면 붙여넣고 Ctrl-D
#
# ⚠ 첫째 형태는 토큰이 명령줄에 남는다. **명령 앞에 공백 한 칸**을 넣으면
#   히스토리에 안 남는다(이 기계 ~/.bashrc 가 HISTCONTROL=ignoreboth 다).
#   둘째 형태는 명령줄을 아예 안 거치므로 그 걱정이 없다.
set -u
cd "$(dirname "$0")/.." || exit 1

if [ $# -ge 1 ]; then
  TOK="$1"
else
  echo "토큰을 붙여넣고 Enter 후 Ctrl-D:"
  TOK="$(cat)"
fi

# 붙여넣기가 흔히 끌고 오는 것들을 턴다: 공백·따옴표·CR·줄바꿈
TOK="$(printf '%s' "$TOK" | tr -d '[:space:]"'"'" )"

if [ -z "$TOK" ]; then echo "✗ 토큰이 비었다"; exit 1; fi

echo
echo "받은 값: ${#TOK}자 · 접두사 ${TOK:0:3} · 비영숫자 $(printf '%s' "$TOK" | tr -d 'A-Za-z0-9_' | wc -c)자"
case "$TOK" in
  hf_*) ;;
  *) echo "⚠ hf_ 로 시작하지 않는다. 그대로 진행은 하지만 십중팔구 오타다";;
esac

# --- 넣기 전에 먼저 물어본다. 틀린 토큰으로 .env 를 덮지 않기 위해서다 ---
echo
echo "→ 넣기 전에 HuggingFace 에 먼저 확인한다"
BODY=$(curl -s -H "Authorization: Bearer $TOK" https://huggingface.co/api/whoami-v2)
NAME=$(printf '%s' "$BODY" | python3 -c "import json,sys;print(json.load(sys.stdin).get('name',''))" 2>/dev/null)
if [ -z "$NAME" ]; then
  echo "✗ 토큰이 무효다 — .env 는 건드리지 않았다."
  echo "  https://huggingface.co/settings/tokens 에서 살아 있는지 확인하고 새로 발급할 것."
  exit 1
fi
echo "✓ 토큰 유효 — 계정: $NAME"

GATE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" \
       https://huggingface.co/facebook/sam3/resolve/main/config.json)
case "$GATE" in
  200) echo "✓ facebook/sam3 접근 가능";;
  403) echo "✗ facebook/sam3 게이트 미승인 (HTTP 403)"
       echo "  https://huggingface.co/facebook/sam3 에서 '$NAME' 계정으로 라이선스 동의할 것."
       echo "  fine-grained 토큰이면 'Read access to public gated repos' 체크도 확인.";;
  *)   echo "⚠ facebook/sam3 응답 HTTP $GATE — 예상 밖이다";;
esac

# --- 여기까지 왔으면 .env 에 넣는다 ---
touch .env
grep -v '^[[:space:]]*#\?[[:space:]]*HF_TOKEN=' .env > .env.tmp
printf 'HF_TOKEN=%s\n' "$TOK" >> .env.tmp
mv .env.tmp .env
chmod 600 .env    # 공유 서버다. 같은 그룹 7 명이 읽지 못하게 한다

echo
echo "✓ .env 에 기록 (권한 $(stat -c%a .env), .gitignore 에 있어 커밋되지 않는다)"
[ "$GATE" = "200" ] && echo "✓ 준비 완료" || echo "△ 토큰은 됐지만 게이트가 남았다 (위 안내 참고)"
