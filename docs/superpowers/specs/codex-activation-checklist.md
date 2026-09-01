# Codex 활성화 체크리스트

## 현재 결론

이 저장소의 Codex 이관은 아직 활성화 후보이며 전환은 보류한다. 이 Task에서는
`/home/rkim/.codex/config.toml`, `/home/rkim/.codex/rules/`, 사용자 수준 hook
등록을 수정하지 않는다. 정확한 설정 초안은 다음과 같다.

| 사용자 수준 항목 | 초안 | 상태 |
| --- | --- | --- |
| `/home/rkim/.codex/config.toml` | 현재 파일을 그대로 유지 | 비활성, 변경 없음 |
| `/home/rkim/.codex/rules/` | 새 Rule을 추가하지 않고 현재 상태를 유지 | 비활성, 변경 없음 |
| custom commit hook | 등록하지 않음 | 제거 결정 유지 |
| 기본 도구 전환 | 수행하지 않음 | 범위 밖 |
| Claude 자산 삭제 | 수행하지 않음 | 범위 밖 |

프로젝트의 16개 `AGENTS.md`와 `.codex/agents/reviewer.toml`은 저장소 자산이다.
다섯 주요 경로의 새 세션 loader smoke는 통과했다. 스킬·actual-diff smoke와 잔여
차이 수용이 끝나기 전에는 이 자산을 전역 전환 완료로 간주하지 않는다.

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

현재 초안은 변경 없음이므로 다음 비교는 출력이 없어야 한다. 별도 활성화 계획에서
후보 파일을 만들면 같은 명령으로 실제 차이를 검토한다.

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
등록하지 않는다. Codex에서는 `AGENTS.md`, 명시적으로 호출하는 읽기 전용 reviewer,
`verification-before-completion`, 사용자가 hook 밖에서 커밋 전에 검증 결과와 diff를
확인하는 정책을 사용한다. 이 네 수단은 automatic commit gate와 동등하지 않으며,
명시적 호출 또는 사람 절차가 누락될 수 있다.

## 활성화 전 판정 체크리스트

- [x] migration 테스트 46개와 기존 Claude hook 테스트 51개의 이전 PASS 근거가 있다.
- [x] custom Codex commit hook 구현과 등록을 활성화 후보에서 제거했다.
- [x] 프로젝트 reviewer의 정적 계약과 controller smoke가 PASS했고 작업 트리가
  바뀌지 않았다.
- [x] `.claude/**`와 모든 `CLAUDE.md`를 보존한다.
- [x] 루트, `src/ner`, `src/server`, `docker`, `tests/ner`의 read-only ephemeral
  Codex 세션에서 root common 규칙과 path-specific 규칙을 구분해 확인했다.
- [ ] 프로젝트 스킬 부재, `systematic-debugging` 통합, actual-diff 설명 동작을
  새 세션에서 확인한다.
- [ ] 사용자가 automatic commit-time enforcement 상실과 보호 파일 영향 범위를
  검토하고 잔여 차이의 수용 여부를 명시한다.
- [ ] 사용자가 전역 활성화를 별도로 승인한다.

loader smoke는 통과했지만 위 미완료 항목이 남아 있으므로 현재 채택 판정은 `후보`,
전환 상태는 `보류`이다.

## 별도 승인 후 실행 경계

현재 초안에는 사용자 수준 파일을 수정하는 활성화 명령이 없다. 이후 계획에서
`config.toml` 또는 Rules 변경이 필요해지면 실제 후보 파일, `diff`, Rule의
`match`와 `not_match`, `codex execpolicy check` 결과를 먼저 제시한다. 다음 형태의
설치 명령은 반드시 `사용자 별도 승인 후 실행` 대상으로 표시하고 이 Task에서는
실행하지 않는다.

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

현재 초안은 사용자 파일을 바꾸지 않으므로 현재 Task의 롤백은 새 Codex 세션을
종료하고 Claude 병행 운영을 계속하는 것이다. 이후 별도 승인으로 사용자 파일을
바꾼 경우에는 활성화 직전 사본의 지문과 대상을 다시 확인한 뒤 다음 복구를 쓴다.

```bash
# 사용자 별도 승인 후 실행
install -m 600 "$codex_activation_audit/config.toml.before" \
  /home/rkim/.codex/config.toml

# 사용자 별도 승인 후 실행
if test -d "$codex_activation_audit/rules.before"; then
  cp -a "$codex_activation_audit/rules.before/." /home/rkim/.codex/rules/
fi
```

복구 후 Codex를 재시작하고 `diff`와 `sha256sum`으로 원본과 일치하는지 확인한다.
별도 활성화에서 새 Rule 파일을 만들었다면 복구 계획에는 그 정확한 파일만 제거하는
명령을 추가한다. 넓은 디렉터리 삭제나 custom hook 제거 명령은 사용하지 않는다.
