# Codex 활성화 체크리스트

## 현재 결론

이 저장소의 Codex 이관은 활성화되었다. 사용자 승인에 따라
`/home/rkim/.codex/config.toml`의 `[features]`에 `plugins = true`를 명시했고,
Rules, agent 정의, 사용자 수준 hook 등록은 추가하지 않았다. 적용 상태는 다음과 같다.

| 사용자 수준 항목 | 초안 | 상태 |
| --- | --- | --- |
| `/home/rkim/.codex/config.toml` | `[features]`의 `plugins = true` | 활성, strict config 파싱과 새 세션 smoke 확인 |
| `/home/rkim/.codex/rules/` | 새 Rule을 추가하지 않고 현재 상태를 유지 | 비활성, 변경 없음 |
| `/home/rkim/.codex/agents/` | agent 정의를 추가하거나 변경하지 않음 | 비활성, 변경 없음 |
| custom commit hook | 등록하지 않음 | 제거 결정 유지 |
| project custom reviewer | 등록하지 않음 | runtime 자동 발견 실패로 제거 |
| 기본 도구 전환 | 수행하지 않음 | 범위 밖 |
| Claude 자산 삭제 | 수행하지 않음 | 범위 밖 |

프로젝트의 16개 `AGENTS.md`가 저장소 지침 자산이다. project custom reviewer 파일은
두 runtime 경로에서 자동 발견되지 않아 제거했다. 다섯 주요 경로와 최종 root
작업 추적성 loader smoke가 통과했다. 전역 플러그인 활성화 후 debug skill과
actual-diff smoke도 통과했으며, 사용자는 automatic commit-time enforcement 상실을
포함한 잔여 차이를 검토한 뒤 활성화 진행을 승인했다.

## 읽기 전용 사전 조사와 백업

다음 명령은 사용자 설정을 원본으로만 읽고 별도의 임시 디렉터리에 사본과 지문을
만든다. 활성화 직전에 다시 실행하고, 출력된 임시 경로를 검토 기록에 남긴다.

```bash
codex_activation_audit=$(mktemp -d)
install -m 600 /home/rkim/.codex/config.toml \
  "$codex_activation_audit/config.toml.before"
if test -d /home/rkim/.codex/rules; then
  cp -a /home/rkim/.codex/rules "$codex_activation_audit/rules.before"
fi
sha256sum /home/rkim/.codex/config.toml \
  "$codex_activation_audit/config.toml.before"
find /home/rkim/.codex/rules -type f -print0 2>/dev/null \
  | sort -z \
  | xargs -0 -r sha256sum
```

이번 활성화에서는 다음 비교가 `[features]` 아래의 `plugins = true` 한 줄만 보여야
한다. 후속 설정 변경에서도 같은 명령으로 실제 차이를 검토한다.

```bash
diff -u "$codex_activation_audit/config.toml.before" \
  /home/rkim/.codex/config.toml
if test -d "$codex_activation_audit/rules.before"; then
  diff -ruN "$codex_activation_audit/rules.before" \
    /home/rkim/.codex/rules
fi
```

## 영향 범위와 잔여 차이

[OpenAI 공식 Rules 문서](https://developers.openai.com/codex/rules/)에 따르면 Rules는
샌드박스 밖에서 실행하려는 명령의 인자 prefix에 `allow`, `prompt`, `forbidden`
판정을 적용한다. Rules를 경로 기반 파일 쓰기 권한이나 diff 기반 커밋 검증으로
해석하지 않는다. 특히 다음 보장은 Rules에 맡기지 않는다.

- `certified/**/*.json`과 기준 파일에 대한 Write/Edit 경로 차단
- 현재 diff의 Ruff, 테스트 무결성, 기준 파일 변경, certified 인용 수치 검사
- shell indirection이나 다른 실행기를 통한 Git 호출의 완전 판별
- automatic commit-time deterministic enforcement

custom Codex commit hook은 Bash indirection bypass를 완전 판별할 수 없으므로
등록하지 않는다. project custom reviewer도 runtime 자동 발견이 확인되지 않아
등록하지 않는다. Codex에서는 `AGENTS.md`, native `code-reviewer` 또는 exact prompt를
받은 generic read-only subagent, `verification-before-completion`, 사용자가 hook 밖에서
커밋 전에 검증 결과와 diff를 확인하는 정책을 사용한다. native type이 없는 환경에서는
generic fallback의 역할 preset을 보장하지 못한다. 이 수단들은 automatic commit gate와
동등하지 않으며, 명시적 호출 또는 사람 절차가 누락될 수 있다.

## 활성화 전 판정 체크리스트

- [x] 활성화 후 migration 테스트와 기존 Claude hook 테스트를 함께 재실행해
  `97 passed`를 확인하고 Ruff와 `git diff --check`도 통과했다.
- [x] custom Codex commit hook 구현과 등록을 활성화 후보에서 제거했다.
- [x] project custom reviewer의 두 runtime 호출이 `unknown agent_type reviewer`를
  반환해 repository TOML을 제거했다.
- [x] native `code-reviewer` smoke가 PASS했고 작업 트리가 바뀌지 않았다. 사용할 수
  없는 환경의 generic read-only fallback 계약은 root `AGENTS.md`에 명시했다.
- [x] `.claude/**`와 모든 `CLAUDE.md`를 보존한다.
- [x] 루트, `src/ner`, `src/server`, `docker`, `tests/ner`의 read-only ephemeral
  Codex 세션에서 root common 규칙과 path-specific 규칙을 구분해 확인했다.
- [x] 프로젝트 스킬 부재 상태에서 전역 Superpowers 플러그인의
  `systematic-debugging` 통합과 actual-diff 설명 동작을 새 세션에서 확인했다.
- [x] 사용자가 automatic commit-time enforcement 상실과 보호 파일 영향 범위를
  검토한 뒤 남은 작업 진행을 지시하여 잔여 차이를 수용했다.
- [x] 사용자 승인에 따라 전역 `plugins = true`를 적용하고 플래그 없는 새 세션에서
  skill 로딩을 재검증했다.

활성화 판정에 필요한 loader, reviewer, debug skill, actual-diff smoke가 통과했다.
현재 채택 판정은 `채택`, 전환 상태는 `활성`이다. 직접 smoke하지 않은 세부 하위
경로와 전체 suite의 기존 EN fingerprint 불일치 2건은 별도 잔여 항목으로 유지한다.

## 후속 설정 변경 경계

이번 활성화에서 사용자 수준 변경은 `plugins = true` 한 줄뿐이다. 이후 계획에서
추가 `config.toml` 또는 Rules 변경이 필요해지면 실제 후보 파일, `diff`, Rule의
`match`와 `not_match`, `codex execpolicy check` 결과를 먼저 제시한다. 다음 형태의
설치 명령은 반드시 사용자 별도 승인 대상으로 유지한다.

```bash
# 사용자 별도 승인 후 실행
install -m 600 /tmp/codex-activation-candidate/config.toml \
  /home/rkim/.codex/config.toml

# 사용자 별도 승인 후 실행
install -d -m 700 /home/rkim/.codex/rules
install -m 600 /tmp/codex-activation-candidate/default.rules \
  /home/rkim/.codex/rules/default.rules
```

이 명령 예시는 승인 절차의 경계를 나타낼 뿐 현재 채택된 설정이 아니다. custom
commit hook 등록은 이후 후보에도 포함하지 않는다.

## 롤백

이번 활성화의 롤백 대상은 `[features]`의 `plugins = true` 한 줄뿐이다. 먼저 현재
설정에서 그 한 줄을 제외한 내용이 활성화 직전 사본과 같은지 확인한다. 다른 변경이
있으면 자동 롤백을 중단하고 diff를 검토한다.

```bash
test "$(rg -n '^plugins = true$' /home/rkim/.codex/config.toml | wc -l)" -eq 1
cmp \
  <(sed '/^plugins = true$/d' /home/rkim/.codex/config.toml) \
  "$codex_activation_audit/config.toml.before"
```

사전 검사가 통과한 뒤 다음 변경 단계는 사용자 별도 승인 후 실행한다. 변경 직전에도
같은 조건을 다시 확인하며, 실패하면 `sed -i`를 실행하지 않는다.

```bash
# 사용자 별도 승인 후 실행
if test "$(rg -n '^plugins = true$' /home/rkim/.codex/config.toml | wc -l)" -eq 1 \
  && cmp \
    <(sed '/^plugins = true$/d' /home/rkim/.codex/config.toml) \
    "$codex_activation_audit/config.toml.before"; then
  sed -i '/^plugins = true$/d' /home/rkim/.codex/config.toml
  cmp /home/rkim/.codex/config.toml \
    "$codex_activation_audit/config.toml.before"
  codex --strict-config --version
else
  echo "Codex config changed after activation; rollback aborted." >&2
  exit 1
fi
```

두 지문이 일치하면 Codex를 재시작한다. 이번 활성화에서는 Rules를 변경하지 않았으므로
롤백도 Rules를 수정하지 않는다. 향후 Rule 파일을 만들면 해당 변경에 정확한 파일별
롤백을 별도로 기록하며 넓은 디렉터리 삭제나 custom hook 제거 명령은 사용하지 않는다.
