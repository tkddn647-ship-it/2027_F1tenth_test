# AI Office (이 레포용)

이슈에 `ai-office` 라벨을 붙이면 Claude(기획·리뷰)와 Cursor(구현)가 작업하고,
**`feat/mapless40-asym-sac` 브랜치로** PR을 올립니다. 자동 머지는 하지 않습니다.

```
이슈 + 라벨 "ai-office"
  → Planner  (Claude, 읽기 전용)  최대 3개 작업으로 나눔
  → Builder  (Cursor)             작업마다 구현 + 테스트 실행
  → Reviewer (Claude, 읽기 전용)  diff 리뷰, 프로젝트 규칙 위반은 "높음"
  → PR → feat/mapless40-asym-sac
```

## 처음 한 번 설정

1. **Settings → Secrets and variables → Actions**에 등록 (테스트 레포와 별개라 다시 넣어야 함)
   - `CLAUDE_CODE_OAUTH_TOKEN` — `claude setup-token`으로 발급
   - `CURSOR_API_KEY` — https://cursor.com/dashboard/api
2. **Settings → Actions → General → Workflow permissions**
   - Read and write permissions
   - Allow GitHub Actions to create and approve pull requests 체크
3. 이 설정 PR(`ai-office-setup` → `main`)을 머지. 이슈 트리거는 **기본 브랜치(main)에 있는 워크플로**만 실행됩니다.

## 이슈 잘 쓰는 법

AI는 이슈 본문만 보고 일합니다. 대상 폴더, 원하는 결과, 완료 기준을 적을수록 결과가 좋습니다.

> 좋은 예: `camera/depthfly/`의 관측 정규화에서 7 m 밖을 0으로 자르는 부분을 상수로 빼 주세요. 완료 기준: `python -m camera.depthfly.tests` 통과
> 나쁜 예: depthfly 개선해줘

## 규칙과 한계

- 프로젝트 규칙은 `office/config.yaml`의 `project_notes`에 있습니다 (실차 코드 보호, 레거시 금지, 학습 실행 금지 등).
- CI에는 GPU·ROS·실차가 없습니다. **학습 결과와 실차 동작은 검증되지 않습니다.** 테스트 통과는 "코드가 깨지지 않았다"는 뜻이지 "더 잘 달린다"는 뜻이 아닙니다.
- 한도: 실행 1번 = Claude 2회 + Cursor 최대 3회. 내 Pro/Cursor 구독 한도에서 빠집니다.
- 라벨은 저장소 쓰기 권한이 있는 사람만 붙일 수 있습니다. 팀원에게 쓰기 권한이 있으면 그 사람이 붙여도 **내 구독 한도**가 쓰입니다.
