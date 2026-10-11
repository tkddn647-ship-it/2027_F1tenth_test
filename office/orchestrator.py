#!/usr/bin/env python3
"""AI Office 오케스트레이터: 이슈 → 기획(Planner) → 구현(Builder) → 리뷰(Reviewer) → PR.

사용법:
  # GitHub Actions에서 (워크플로가 환경변수를 넣어줌)
  python office/orchestrator.py

  # 로컬 테스트: 커밋/푸시/PR 없이 기획만 확인
  python office/orchestrator.py --issue-title "로그인 버튼 색 변경" --issue-body "..." --plan-only

  # 로컬 테스트: 구현·리뷰까지 하되 push/PR은 안 함
  python office/orchestrator.py --issue-title "..." --issue-body "..." --no-push
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML이 필요합니다: pip install pyyaml")

OFFICE_DIR = Path(__file__).resolve().parent
# 워크플로는 office/를 임시 폴더로 복사해 실행하므로(작업 브랜치에 office/가 없어도 되게) 경로를 환경변수로 받는다
REPO_ROOT = Path(os.environ.get("AI_OFFICE_REPO_ROOT") or OFFICE_DIR.parent).resolve()
LOG_DIR = Path(os.environ.get("AI_OFFICE_LOG_DIR") or REPO_ROOT / ".ai-office-logs")


# ---------- 유틸 ----------

def log(msg: str) -> None:
    print(f"[ai-office {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fill(template: str, **values) -> str:
    """{key}만 치환. 이슈 본문의 중괄호 때문에 str.format이 깨지는 걸 피한다."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def sh(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check, **kw)


def save_log(name: str, content: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / name).write_text(content, encoding="utf-8")


# ---------- 엔진 호출 ----------

def run_claude(prompt: str, role: dict) -> str:
    """Claude Code CLI 헤드리스 실행. 인증은 CLAUDE_CODE_OAUTH_TOKEN(구독) 또는 ANTHROPIC_API_KEY."""
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--max-turns", str(role.get("max_turns", 8))]
    if role.get("model"):
        cmd += ["--model", role["model"]]
    if role.get("allowed_tools"):
        cmd += ["--allowedTools", ",".join(role["allowed_tools"])]
    return _run_engine(cmd, role, "claude")


def run_cursor(prompt: str, role: dict) -> str:
    """Cursor Agent CLI 헤드리스 실행. 인증은 CURSOR_API_KEY."""
    cmd = ["cursor-agent", "-p", prompt, "--output-format", "json"]
    if role.get("model"):
        cmd += ["--model", role["model"]]
    cmd += role.get("extra_args", [])
    return _run_engine(cmd, role, "cursor")


def _run_engine(cmd: list[str], role: dict, engine: str) -> str:
    timeout = role.get("timeout_sec", 900)
    try:
        p = sh(cmd, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{engine}: {timeout}초 안에 끝나지 않았습니다")
    except FileNotFoundError:
        raise RuntimeError(f"{cmd[0]} 명령을 찾을 수 없습니다. 설치/PATH를 확인하세요")

    out = p.stdout.strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        # 일부 버전은 JSON 대신 텍스트를 줄 수 있음
        if p.returncode != 0:
            raise RuntimeError(f"{engine} 실패 (exit {p.returncode}):\n{p.stderr[-2000:]}")
        return out

    if p.returncode != 0 or data.get("is_error"):
        raise RuntimeError(
            f"{engine} 실패 (exit {p.returncode}, subtype={data.get('subtype')}):\n"
            f"{data.get('result') or p.stderr[-2000:]}"
        )
    # 사용량 기록 (Cursor는 0으로 나올 수 있음)
    usage = data.get("usage") or {}
    cost = data.get("total_cost_usd")
    log(f"  {engine} 완료 · usage={usage} · cost={cost}")
    return data.get("result", "")


ENGINES = {"claude": run_claude, "cursor": run_cursor}


def call(role_name: str, prompt: str, cfg: dict) -> str:
    role = cfg["roles"][role_name]
    engine = role["engine"]
    if engine not in ENGINES:
        raise ValueError(f"{role_name}.engine '{engine}'은 지원하지 않습니다 (claude|cursor)")
    log(f"{role_name} → {engine} 호출")
    result = ENGINES[engine](prompt, role)
    save_log(f"{role_name}.out.md", result)
    return result


# ---------- 단계 ----------

def notes(cfg: dict) -> str:
    return (cfg.get("project_notes") or "").strip() or "(없음)"


def extract_json(text: str) -> dict:
    """모델이 코드블록이나 설명을 섞어도 첫 JSON 객체를 뽑는다."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Planner 출력에서 JSON을 찾지 못했습니다:\n" + text[:1500])
    return json.loads(text[start:end + 1])


def plan(issue: dict, cfg: dict) -> dict:
    tpl = (OFFICE_DIR / "roles/planner.md").read_text(encoding="utf-8")
    prompt = fill(tpl, project_notes=notes(cfg), max_tasks=cfg["max_tasks"], issue_number=issue["number"],
                  issue_title=issue["title"], issue_body=issue["body"] or "(본문 없음)")
    result = extract_json(call("planner", prompt, cfg))
    tasks = result.get("tasks") or []
    if len(tasks) > cfg["max_tasks"]:
        log(f"  작업이 {len(tasks)}개라 {cfg['max_tasks']}개로 자릅니다")
        result["tasks"] = tasks[: cfg["max_tasks"]]
    save_log("plan.json", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build(plan_data: dict, cfg: dict, commit: bool) -> list[dict]:
    tpl = (OFFICE_DIR / "roles/builder.md").read_text(encoding="utf-8")
    tasks = plan_data["tasks"]
    reports = []
    for t in tasks:
        prompt = fill(tpl, project_notes=notes(cfg), summary=plan_data.get("summary", ""), task_id=t.get("id"),
                      task_count=len(tasks), title=t.get("title", ""),
                      files=", ".join(t.get("files", [])) or "(미지정)",
                      instructions=t.get("instructions", ""), done_when=t.get("done_when", ""))
        try:
            report = call("builder", prompt, cfg)
            ok = True
        except RuntimeError as e:
            report, ok = str(e), False
            log(f"  작업 {t.get('id')} 실패: {e}")
        changed = bool(sh(["git", "status", "--porcelain"]).stdout.strip())
        if commit and changed:
            sh(["git", "add", "-A"])  # 로그 폴더는 .gitignore로 제외됨
            sh(["git", "commit", "-m", f"ai-office: task {t.get('id')} - {t.get('title', '')}"])
        reports.append({"task": t, "ok": ok, "changed": changed, "report": report})
    return reports


def review(issue: dict, plan_data: dict, base: str, cfg: dict) -> str:
    diff = sh(["git", "diff", f"{base}...HEAD", "--", ".", ":!.ai-office-logs"], check=False).stdout
    if not diff.strip():  # 커밋 안 하는 로컬 모드: 새 파일도 diff에 보이게 intent-to-add
        sh(["git", "add", "-A", "-N"], check=False)
        diff = sh(["git", "diff", "--", ".", ":!.ai-office-logs"], check=False).stdout
    if not diff.strip():
        return "## 판정\n변경 사항이 없어 리뷰를 건너뜁니다."
    limit = cfg["max_diff_chars"]
    note = f" (앞 {limit}자만 표시, 전체 {len(diff)}자)" if len(diff) > limit else ""
    tpl = (OFFICE_DIR / "roles/reviewer.md").read_text(encoding="utf-8")
    prompt = fill(tpl, project_notes=notes(cfg), issue_number=issue["number"], issue_title=issue["title"],
                  issue_body=issue["body"] or "", plan_json=json.dumps(plan_data, ensure_ascii=False, indent=2),
                  truncated_note=note, diff=diff[:limit])
    try:
        return call("reviewer", prompt, cfg)
    except RuntimeError as e:
        return f"## 판정\n리뷰 실패 — 사람이 직접 확인하세요.\n\n```\n{e}\n```"


def pr_body(issue: dict, plan_data: dict, reports: list[dict], review_md: str) -> str:
    lines = [f"Closes #{issue['number']}" if issue["number"] else "", "",
             f"**요약:** {plan_data.get('summary', '')}", "", "### 작업 결과"]
    for r in reports:
        mark = "✅" if r["ok"] and r["changed"] else ("⚠️ 변경 없음" if r["ok"] else "❌ 실패")
        lines.append(f"- {mark} **{r['task'].get('id')}. {r['task'].get('title', '')}**")
        lines.append("  <details><summary>Builder 보고</summary>\n\n" + r["report"][:3000] + "\n\n  </details>")
    if plan_data.get("questions"):
        lines += ["", "### Planner가 남긴 질문"] + [f"- {q}" for q in plan_data["questions"]]
    lines += ["", "### 자동 리뷰", review_md, "", "---",
              "_AI Office가 만든 PR입니다. 자동 머지되지 않으니 직접 확인 후 머지하세요._"]
    return "\n".join(lines)


# ---------- 메인 ----------

def load_issue(args) -> dict:
    if args.issue_title:
        return {"number": args.issue_number or 0, "title": args.issue_title, "body": args.issue_body or ""}
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        ev = json.loads(Path(event_path).read_text(encoding="utf-8"))
        if "issue" in ev:
            i = ev["issue"]
            return {"number": i["number"], "title": i["title"], "body": i.get("body") or ""}
    num = args.issue_number or os.environ.get("ISSUE_NUMBER")
    if num:  # workflow_dispatch: gh로 이슈 가져오기
        i = json.loads(sh(["gh", "issue", "view", str(num), "--json", "number,title,body"]).stdout)
        return {"number": i["number"], "title": i["title"], "body": i.get("body") or ""}
    sys.exit("이슈 정보가 없습니다. --issue-title 또는 --issue-number를 주세요.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue-number", type=int)
    ap.add_argument("--issue-title")
    ap.add_argument("--issue-body")
    ap.add_argument("--plan-only", action="store_true", help="기획만 하고 종료")
    ap.add_argument("--no-push", action="store_true", help="커밋/푸시/PR 없이 로컬에서만 실행")
    args = ap.parse_args()

    cfg = yaml.safe_load((OFFICE_DIR / "config.yaml").read_text(encoding="utf-8"))
    issue = load_issue(args)
    log(f"이슈 #{issue['number']}: {issue['title']}")

    base_branch = cfg.get("base_branch")
    if base_branch:
        cur = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        if cur != base_branch:
            log(f"작업 브랜치로 전환: {cur} → {base_branch}")
            sh(["git", "fetch", "origin", f"{base_branch}:refs/remotes/origin/{base_branch}"], check=False)
            sh(["git", "checkout", "-B", base_branch, f"origin/{base_branch}"])

    if not args.plan_only and sh(["git", "status", "--porcelain"]).stdout.strip():
        sys.exit("커밋 안 된 변경이 있습니다. 내 작업이 AI 커밋에 섞이지 않도록 "
                 "먼저 커밋하거나 `git stash` 한 뒤 다시 실행하세요.")

    plan_data = plan(issue, cfg)
    log(f"계획: {len(plan_data['tasks'])}개 작업 · 질문 {len(plan_data.get('questions', []))}개")
    if args.plan_only:
        print(json.dumps(plan_data, ensure_ascii=False, indent=2))
        return 0
    if not plan_data["tasks"]:
        log("작업이 없습니다. Planner 질문을 확인하세요.")
        return 0

    push = not args.no_push
    base = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    branch = f"ai-office/issue-{issue['number']}-{int(time.time())}"
    if push:
        sh(["git", "config", "user.name", "ai-office[bot]"], check=False)
        sh(["git", "config", "user.email", "ai-office@users.noreply.github.com"], check=False)
        sh(["git", "checkout", "-b", branch])

    reports = build(plan_data, cfg, commit=push)
    review_md = review(issue, plan_data, base, cfg)
    body = pr_body(issue, plan_data, reports, review_md)
    save_log("pr_body.md", body)

    if not push:
        log(f"로컬 모드 종료. 결과: {LOG_DIR}/pr_body.md")
        return 0
    if not any(r["changed"] for r in reports):
        log("변경된 파일이 없어 PR을 만들지 않습니다.")
        if issue["number"]:
            sh(["gh", "issue", "comment", str(issue["number"]), "--body",
                "AI Office: 변경 사항 없이 끝났습니다.\n\n" + body[:60000]], check=False)
        return 1

    sh(["git", "push", "-u", "origin", branch])
    pr = sh(["gh", "pr", "create", "--base", base, "--head", branch,
             "--title", f"[AI Office] {issue['title']}", "--body", body[:60000]])
    log(f"PR 생성: {pr.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
