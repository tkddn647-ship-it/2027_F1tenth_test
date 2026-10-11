# AI Office (이 레포용)

이슈에 `ai-office` 라벨을 붙이면 Claude(기획·리뷰)와 Cursor(구현)가 작업하고,
**`feat/mapless40-asym-sac` 브랜치로** PR을 올립니다. 자동 머지는 하지 않습니다.

```
이슈 + 라벨 "ai-office"
  → Planner  (Claude, 읽기 전용)  이슈·댓글 읽고 최대 3개 작업으로 나눔
       └ 막히는 질문이 있으면 → 이슈에 질문 댓글 + 라벨 자동 제거 → 멈춤 (작업 안 함)
  → Builder  (Cursor)             작업마다 구현
  → CI 테스트 (오케스트레이터가 직접 실행)  tests.py가 있는 패키지를 자동으로 찾아 전부 실행
  → Reviewer (Claude, 읽기 전용)  diff + Builder 보고 + CI 결과로 판정
  → PR → feat/mapless40-asym-sac
       ├ APPROVE + 테스트 통과 → 일반 PR
       └ 수정 요청 / 테스트 실패 / 작업 실패 → Draft PR + `needs-human` 라벨 + 경고 배너
```

## 처음 한 번 설정

1. **Settings → Secrets and variables → Actions**에 등록 (테스트 레포와 별개라 다시 넣어야 함)
   - `CLAUDE_CODE_OAUTH_TOKEN` — `claude setup-token`으로 발급
   - `CURSOR_API_KEY` — https://cursor.com/dashboard/api
2. **Settings → Actions → General → Workflow permissions**
   - Read and write permissions
   - Allow GitHub Actions to create and approve pull requests 체크
3. 워크플로는 **기본 브랜치(main)에 있는 것**만 이슈에 반응합니다. AI Office 변경은 main에 머지돼야 적용됩니다.

## 진행 상황 알림

- 라벨을 붙이면 1분 안에 이슈에 **"🏃 작업을 시작합니다 + 진행 상황 링크"** 댓글이 달립니다. 이 댓글이 안 달리면 실행이 안 된 것입니다 (라벨 이름, main에 워크플로가 있는지 확인).
- 실행이 실패하면 **"❌ 실패 + 로그 링크"** 댓글이 달립니다.

## PR 고치기: `/ai-office fix`

PR이 "확인 필요"로 왔거나 고칠 점이 있으면, **새 이슈를 만들지 말고 그 PR에 댓글**을 다세요.

```
/ai-office fix
/ai-office fix reset마다 트랙 만들지 말고 random이 뽑혔을 때만 만들어줘
```

- 명령 뒤에 지시를 덧붙이면 리뷰 지적보다 **사람 지시를 우선**합니다. 비워 두면 직전 리뷰 지적을 고칩니다.
- 같은 PR 브랜치에 커밋 → CI 테스트 → PR 전체 diff 재리뷰 → 결과를 PR 댓글로 남깁니다. 통과하면 `needs-human` 라벨과 제목의 "확인 필요"가 자동으로 빠집니다.
- PR 하나당 최대 2번 (`max_fix_rounds`). 그 이상은 범위를 좁힌 새 이슈로.
- 저장소 소유자·협업자의 댓글만 실행됩니다.

## AI가 질문을 올렸을 때

이슈가 모호하면 AI는 추측해서 작업하지 않고 **질문 댓글을 남긴 뒤 멈춥니다** (Pro 한도도 아낌).

1. 이슈에 **댓글로 답**합니다. 선택지가 있으면 번호만 써도 됩니다.
2. `ai-office` 라벨을 **다시 붙입니다** (AI가 멈출 때 자동으로 뗐습니다).
3. AI가 답을 읽고 이어서 진행합니다.

질문을 무시하고 그냥 진행하려면 댓글에 `/ai-office go`라고 쓰고 라벨을 붙이세요.

## PR이 "확인 필요"로 왔을 때

제목에 `[AI Office · 확인 필요]`, 맨 위에 노란 경고, `needs-human` 라벨이 붙습니다.
이유는 셋 중 하나입니다: 리뷰어 수정 요청 / CI 테스트 실패 / 작업 실패.
Draft PR이라 바로 머지되지 않습니다. 확인 후 머지하려면 **Ready for review**를 누른 뒤 머지하세요.
(Draft를 못 만드는 경우엔 일반 PR로 오고 경고와 라벨만 붙습니다.)

## 이슈 잘 쓰는 법

AI는 이슈 본문과 댓글만 보고 일합니다. 대상 폴더, 원하는 결과, 완료 기준을 적을수록 결과가 좋습니다.

> 좋은 예: `camera/depthfly/`의 관측 정규화에서 7 m 밖을 0으로 자르는 부분을 상수로 빼 주세요. 완료 기준: `python -m camera.depthfly.tests` 통과
> 나쁜 예: depthfly 개선해줘

의견을 묻는 질문("어케 생각해?")은 AI Office 말고 채팅으로 먼저 정리한 뒤, 할 일이 정해지면 이슈로 만드세요.

## 규칙과 한계

- 프로젝트 규칙은 `office/config.yaml`의 `project_notes`에 있습니다.
- CI 테스트는 `__init__.py`가 있는 패키지 안의 `tests.py`를 자동으로 찾습니다 (`test_discover`). 새 패키지를 만들면 `tests.py`(실패 시 exit 1)를 같이 두세요. 실차 폴더는 `test_exclude`로 제외됩니다.
- AI에게 전달되는 댓글은 **저장소 소유자·협업자의 것만**입니다. 공개 저장소라 외부인이 댓글로 AI에 지시를 끼워넣는 것을 막기 위함입니다.
- CI에는 GPU·ROS·실차가 없습니다. **학습 결과와 실차 동작은 검증되지 않습니다.** 테스트 통과는 "코드가 깨지지 않았다"는 뜻이지 "더 잘 달린다"는 뜻이 아닙니다.
- 작업 브랜치가 main이 아니라서 PR이 머지돼도 **이슈는 자동으로 닫히지 않습니다.** 직접 닫아 주세요.
- 한도: 실행 1번 = Claude 2회 + Cursor 최대 3회. 질문으로 멈추면 Claude 1회, 수정 라운드 1번 = Cursor 1회 + Claude 1회. 내 Pro/Cursor 구독 한도에서 빠집니다.
- 라벨은 저장소 쓰기 권한이 있는 사람만 붙일 수 있습니다. 팀원에게 쓰기 권한이 있으면 그 사람이 붙여도 **내 구독 한도**가 쓰입니다.
