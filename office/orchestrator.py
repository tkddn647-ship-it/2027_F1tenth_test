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


# ---------- GitHub (REST만 사용: gh api) ----------

QUESTION_MARKER = "<!-- ai-office:questions -->"
GO_COMMAND = "/ai-office go"
FIX_COMMAND = "/ai-office fix"
FIX_MARKER = "<!-- ai-office:fix-round -->"


def gh_repo() -> str:
    # Actions에선 GITHUB_REPOSITORY, 로컬에선 gh가 git remote로 {owner}/{repo}를 채운다
    return os.environ.get("GITHUB_REPOSITORY") or "{owner}/{repo}"


def gh_api(path: str, *fields: str, method: str = "GET", check: bool = True):
    cmd = ["gh", "api", "-X", method, f"repos/{gh_repo()}" + (f"/{path}" if path else "")]
    for f in fields:
        cmd += ["-F" if f.split("=", 1)[1] in ("true", "false") else "-f", f]
    p = sh(cmd, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"gh api {method} {path} 실패: {(p.stderr or p.stdout)[-800:]}")
    try:
        return json.loads(p.stdout) if p.stdout.strip() else None
    except json.JSONDecodeError:
        return None


TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}


def fetch_comments(number: int) -> list[dict]:
    """저장소 소유자·협업자와 AI Office 자신의 댓글만 쓴다.
    공개 저장소에선 아무나 댓글을 달 수 있어서, 남의 댓글로 AI에 지시를 끼워넣는 걸 막기 위함."""
    if not number:
        return []
    try:
        data = gh_api(f"issues/{number}/comments?per_page=100") or []
    except RuntimeError as e:
        log(f"  댓글을 못 가져왔습니다 (계속 진행): {e}")
        return []
    out, skipped = [], 0
    for c in data:
        body = c.get("body") or ""
        is_bot = (c.get("user") or {}).get("type") == "Bot"
        trusted = c.get("author_association") in TRUSTED
        if not (is_bot or trusted):
            skipped += 1
            continue
        out.append({"human": trusted and not is_bot and QUESTION_MARKER not in body and FIX_MARKER not in body,
                    "author": (c.get("user") or {}).get("login", "?"), "body": body})
    if skipped:
        log(f"  권한 없는 사용자의 댓글 {skipped}개는 무시합니다")
    return out


def format_comments(comments: list[dict]) -> str:
    if not comments:
        return "(댓글 없음)"
    out = []
    for c in comments[-20:]:
        who = f"사람({c['author']})" if c["human"] else "AI Office"
        out.append(f"[{who}]\n{c['body'][:2000]}")
    return "\n\n".join(out)


def question_state(comments: list[dict]) -> tuple[bool, bool]:
    """(이전에 질문을 올렸는가, 그 뒤에 사람이 답했는가 또는 /ai-office go)"""
    asked_idx = max((i for i, c in enumerate(comments) if QUESTION_MARKER in c["body"]), default=-1)
    force = any(c["human"] and GO_COMMAND in c["body"] for c in comments)
    answered = asked_idx >= 0 and any(c["human"] for c in comments[asked_idx + 1:])
    return asked_idx >= 0, answered or force


def ensure_label(name: str, color: str, desc: str) -> None:
    gh_api("labels", f"name={name}", f"color={color}", f"description={desc}", method="POST", check=False)


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
                  issue_title=issue["title"], issue_body=issue["body"] or "(본문 없음)",
                  issue_comments=format_comments(issue.get("comments", [])))
    result = extract_json(call("planner", prompt, cfg))
    result.setdefault("tasks", [])
    result.setdefault("questions", [])
    result.setdefault("notes", [])
    if len(result["tasks"]) > cfg["max_tasks"]:
        log(f"  작업이 {len(result['tasks'])}개라 {cfg['max_tasks']}개로 자릅니다")
        result["tasks"] = result["tasks"][: cfg["max_tasks"]]
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
            sh(["git", "add", "-A"])
            sh(["git", "commit", "-m", f"ai-office: task {t.get('id')} - {t.get('title', '')}"])
        reports.append({"task": t, "ok": ok, "changed": changed, "report": report})
    return reports


SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", ".ai-office-logs"}


def discover_tests(cfg: dict) -> list[str]:
    """__init__.py가 있는 패키지 안의 tests.py → `python -m 패키지.tests`. 새로 만든 패키지도 자동으로 검사된다."""
    if not cfg.get("test_discover"):
        return []
    exclude = {e.strip("/") for e in cfg.get("test_exclude") or []}
    found = []
    for root, dirs, files in os.walk(REPO_ROOT):
        rel = Path(root).relative_to(REPO_ROOT)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and (rel / d).as_posix() not in exclude)
        if "tests.py" in files and "__init__.py" in files and rel.parts:
            if all(part.isidentifier() for part in rel.parts):
                found.append(f"python -m {'.'.join(rel.parts)}.tests")
    return found


def test_command_list(cfg: dict) -> list[str]:
    cmds = list(cfg.get("test_commands") or [])
    for c in discover_tests(cfg):
        if c not in cmds:
            cmds.append(c)
    return cmds


def run_tests(cfg: dict) -> list[dict]:
    """Builder와 별개로 오케스트레이터가 직접 테스트를 돌린다 (Builder 보고를 그대로 믿지 않기 위해)."""
    results = []
    timeout = cfg.get("test_timeout_sec", 600)
    for cmd in test_command_list(cfg):
        log(f"테스트: {cmd}")
        t0 = time.time()
        try:
            p = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout)
            code, out = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired:
            code, out = -1, f"{timeout}초 시간 초과"
        r = {"cmd": cmd, "ok": code == 0, "code": code, "secs": round(time.time() - t0, 1),
             "tail": "\n".join(out.strip().splitlines()[-15:])}
        log(f"  {'통과' if r['ok'] else '실패'} (exit {code}, {r['secs']}s)")
        results.append(r)
    save_log("tests.json", json.dumps(results, ensure_ascii=False, indent=2))
    return results


def format_tests(tests: list[dict], detail: bool = True) -> str:
    if not tests:
        return "(실행한 테스트 없음 — config.yaml의 test_commands / test_discover)"
    lines = ["| 명령 | 결과 | 시간 |", "|---|---|---|"]
    for t in tests:
        result = "✅ 통과" if t["ok"] else "❌ 실패 (exit %s)" % t["code"]
        lines.append(f"| `{t['cmd']}` | {result} | {t['secs']}s |")
    if detail:
        for t in tests:
            lines.append(f"\n<details><summary>{t['cmd']} 출력 (끝 15줄)</summary>\n\n```\n{t['tail']}\n```\n</details>")
    return "\n".join(lines)


def format_reports(reports: list[dict]) -> str:
    out = []
    for r in reports:
        state = "실패" if not r["ok"] else ("변경 있음" if r["changed"] else "변경 없음")
        out.append(f"[작업 {r['task'].get('id')}: {r['task'].get('title', '')} — {state}]\n{r['report'][:2500]}")
    return "\n\n".join(out) or "(없음)"


def review(issue: dict, plan_data: dict, base: str, reports: list[dict], tests: list[dict], cfg: dict) -> str:
    """base: diff 기준 (브랜치 이름 또는 origin/브랜치)."""
    diff = sh(["git", "diff", f"{base}...HEAD", "--", ".", ":!.ai-office-logs"], check=False).stdout
    if not diff.strip():  # 커밋 안 하는 로컬 모드: 새 파일도 diff에 보이게 intent-to-add
        sh(["git", "add", "-A", "-N"], check=False)
        diff = sh(["git", "diff", "--", ".", ":!.ai-office-logs"], check=False).stdout
    if not diff.strip():
        return "VERDICT: REQUEST_CHANGES\n## 판정 이유\n변경 사항이 없습니다."
    limit = cfg["max_diff_chars"]
    note = f" (앞 {limit}자만 표시, 전체 {len(diff)}자)" if len(diff) > limit else ""
    tpl = (OFFICE_DIR / "roles/reviewer.md").read_text(encoding="utf-8")
    prompt = fill(tpl, project_notes=notes(cfg), issue_number=issue["number"], issue_title=issue["title"],
                  issue_body=issue["body"] or "", plan_json=json.dumps(plan_data, ensure_ascii=False, indent=2),
                  builder_reports=format_reports(reports), test_results=format_tests(tests),
                  truncated_note=note, diff=diff[:limit])
    try:
        return call("reviewer", prompt, cfg)
    except RuntimeError as e:
        return f"VERDICT: REQUEST_CHANGES\n## 판정 이유\n리뷰 실패 — 사람이 직접 확인하세요.\n\n```\n{e}\n```"


def parse_verdict(review_md: str) -> str:
    m = re.search(r"VERDICT:\s*(APPROVE|REQUEST_CHANGES)", review_md)
    if m:
        return m.group(1)
    head = review_md[:400]  # 형식을 안 지켰을 때 대비
    if "REQUEST_CHANGES" in head:
        return "REQUEST_CHANGES"
    if "APPROVE" in head:
        return "APPROVE"
    return "UNKNOWN"


def status_banner(verdict: str, reports: list[dict], tests: list[dict]) -> tuple[bool, str]:
    problems = []
    if verdict == "REQUEST_CHANGES":
        problems.append("리뷰어가 **수정 요청**")
    elif verdict == "UNKNOWN":
        problems.append("리뷰 판정을 **읽지 못함**")
    failed_tests = [t for t in tests if not t["ok"]]
    if failed_tests:
        problems.append(f"CI 테스트 **{len(failed_tests)}/{len(tests)}개 실패**")
    failed_tasks = [r for r in reports if not r["ok"]]
    if failed_tasks:
        problems.append(f"작업 **{len(failed_tasks)}/{len(reports)}개 실패**")
    if problems:
        return True, ("> [!WARNING]\n> **사람 확인 필요:** " + " · ".join(problems) +
                      "\n> 아래 리뷰와 테스트 결과를 보고 판단하세요. 이대로 머지해도 되지만, 그건 사람의 결정입니다.")
    tests_note = f"CI 테스트 {len(tests)}개 통과" if tests else "CI 테스트 미설정"
    return False, f"> [!NOTE]\n> 리뷰 **APPROVE** · {tests_note}. 그래도 머지 전에 diff는 한 번 보세요."


def pr_body(issue: dict, plan_data: dict, reports: list[dict], tests: list[dict],
            review_md: str, banner: str, base: str) -> str:
    n = issue["number"]
    lines = [banner, ""]
    if n:
        lines.append(f"Closes #{n}" if base == default_branch() else
                     f"이슈 #{n} (작업 브랜치 `{base}`로 머지돼도 이슈는 자동으로 닫히지 않으니 직접 닫아 주세요)")
    lines += ["", f"**요약:** {plan_data.get('summary', '')}", "", "### 작업 결과"]
    for r in reports:
        mark = "✅" if r["ok"] and r["changed"] else ("⚠️ 변경 없음" if r["ok"] else "❌ 실패")
        lines.append(f"- {mark} **{r['task'].get('id')}. {r['task'].get('title', '')}**")
        lines.append("  <details><summary>Builder 보고</summary>\n\n" + r["report"][:3000] + "\n\n  </details>")
    lines += ["", "### CI 테스트 (오케스트레이터가 직접 실행)", format_tests(tests)]
    if plan_data.get("questions"):
        lines += ["", "### 남은 질문"] + [f"- {q}" for q in plan_data["questions"]]
    if plan_data.get("notes"):
        lines += ["", "### Planner 참고 의견"] + [f"- {q}" for q in plan_data["notes"]]
    lines += ["", "### 자동 리뷰", re.sub(r"^VERDICT:.*\n?", "", review_md, count=1), "", "---",
              "_AI Office가 만든 PR입니다. 자동 머지되지 않으니 직접 확인 후 머지하세요._"]
    return "\n".join(lines)


_default_branch_cache: str | None = None


def default_branch() -> str:
    global _default_branch_cache
    if _default_branch_cache is None:
        try:
            _default_branch_cache = (gh_api("") or {}).get("default_branch") or "main"
        except RuntimeError:
            _default_branch_cache = "main"
    return _default_branch_cache


def ask_questions(issue: dict, plan_data: dict) -> None:
    n = issue["number"]
    qs = "\n".join(f"{i}. {q}" for i, q in enumerate(plan_data["questions"], 1))
    notes_md = ("\n\n**참고 의견**\n" + "\n".join(f"- {x}" for x in plan_data["notes"])) if plan_data.get("notes") else ""
    body = (f"{QUESTION_MARKER}\n### 🤔 AI Office: 작업 전에 확인이 필요합니다\n\n"
            f"**이해한 내용:** {plan_data.get('summary', '')}\n\n"
            f"추측으로 진행하면 결과가 틀어질 수 있어서 멈췄습니다. 이 이슈에 댓글로 답해 주세요.\n\n{qs}{notes_md}\n\n"
            f"---\n**이어서 진행하려면:** 답 댓글을 단 뒤 `ai-office` 라벨을 다시 붙이세요 (방금 자동으로 뗐습니다).\n"
            f"질문을 무시하고 그냥 진행하려면 댓글에 `{GO_COMMAND}` 라고 쓰고 라벨을 붙이세요.")
    gh_api(f"issues/{n}/comments", f"body={body}", method="POST")
    gh_api(f"issues/{n}/labels/ai-office", method="DELETE", check=False)
    log(f"질문 {len(plan_data['questions'])}개를 이슈 #{n}에 올리고 멈춥니다.")


def create_pr(issue: dict, branch: str, base: str, body: str, needs_human: bool) -> str:
    title = ("[AI Office · 확인 필요] " if needs_human else "[AI Office] ") + issue["title"]
    fields = [f"title={title}", f"head={branch}", f"base={base}", f"body={body[:60000]}"]
    pr = None
    if needs_human:  # 초안(Draft)으로: 실수로 머지 버튼 누르는 것 방지
        try:
            pr = gh_api("pulls", *fields, "draft=true", method="POST")
        except RuntimeError as e:  # 무료 요금제 비공개 레포 등 Draft 미지원
            log(f"  Draft PR을 못 만들어 일반 PR로 만듭니다: {str(e)[:200]}")
    if pr is None:
        pr = gh_api("pulls", *fields, method="POST")
    if needs_human:
        set_needs_human(pr["number"], True)
    return pr.get("html_url", "")


# ---------- 수정 라운드 (/ai-office fix) ----------

def extract_section(md: str, heading: str) -> str:
    """'### heading'부터 다음 '### '나 '---' 구분선 전까지."""
    m = re.search(rf"^###[^\n]*{re.escape(heading)}[^\n]*\n(.*?)(?=^### |^---\s*$|\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def set_needs_human(number: int, needs_human: bool) -> None:
    if needs_human:
        ensure_label("needs-human", "d93f0b", "AI Office: 사람 확인 필요")
        gh_api(f"issues/{number}/labels", "labels[]=needs-human", method="POST", check=False)
    else:
        gh_api(f"issues/{number}/labels/needs-human", method="DELETE", check=False)


def run_fix(pr_number: int, cfg: dict, push: bool) -> int:
    pr = gh_api(f"pulls/{pr_number}")
    if pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]:
        gh_api(f"issues/{pr_number}/comments", "body=AI Office: 포크에서 온 PR은 고치지 않습니다.", method="POST", check=False)
        return 1
    head, base = pr["head"]["ref"], pr["base"]["ref"]
    comments = fetch_comments(pr_number)
    max_rounds = int(cfg.get("max_fix_rounds", 2))
    done_rounds = sum(1 for c in comments if FIX_MARKER in c["body"])
    log(f"PR #{pr_number} 수정 라운드 {done_rounds + 1}/{max_rounds} ({head} → {base})")
    if done_rounds >= max_rounds:
        if push:
            gh_api(f"issues/{pr_number}/comments", f"body={FIX_MARKER}\nAI Office: 수정 라운드를 이미 {max_rounds}번 했습니다 "
                   f"(`max_fix_rounds`). 더 고치려면 사람이 직접 고치거나, 새 이슈로 범위를 좁혀 주세요.", method="POST", check=False)
        log("최대 라운드에 도달해 멈춥니다.")
        return 0

    # 사람의 지시: 가장 최근 '/ai-office fix' 댓글에서 명령 뒤 텍스트
    req = next((c["body"].split(FIX_COMMAND, 1)[1].strip() for c in reversed(comments)
                if c["human"] and FIX_COMMAND in c["body"]), "")
    human_request = req or "(추가 지시 없음 — 아래 리뷰 지적을 고칠 것)"
    # 가장 최근 리뷰/테스트: 마지막 수정 라운드 댓글, 없으면 PR 본문
    last_round = next((c["body"] for c in reversed(comments) if FIX_MARKER in c["body"]), "")
    src = last_round or (pr.get("body") or "")
    latest_review = extract_section(src, "자동 리뷰") or "(리뷰 없음)"
    latest_tests = extract_section(src, "CI 테스트") or "(기록 없음)"
    m = re.search(r"(?:Closes |이슈 )#(\d+)", pr.get("body") or "")
    issue = {"number": int(m.group(1)), "title": "", "body": ""} if m else {"number": 0, "title": pr["title"], "body": ""}
    if m:
        i = gh_api(f"issues/{issue['number']}", check=False) or {}
        issue.update(title=i.get("title", ""), body=i.get("body") or "")

    sh(["git", "fetch", "origin", f"{head}:refs/remotes/origin/{head}", f"{base}:refs/remotes/origin/{base}"], check=False)
    sh(["git", "checkout", "-B", head, f"origin/{head}"])
    if sh(["git", "status", "--porcelain"]).stdout.strip():
        sys.exit("커밋 안 된 변경이 있습니다. 먼저 커밋하거나 `git stash` 하세요.")
    if push:
        sh(["git", "config", "user.name", "ai-office[bot]"], check=False)
        sh(["git", "config", "user.email", "ai-office@users.noreply.github.com"], check=False)

    tpl = (OFFICE_DIR / "roles/fixer.md").read_text(encoding="utf-8")
    prompt = fill(tpl, project_notes=notes(cfg), issue_number=issue["number"], issue_title=issue["title"],
                  issue_body=issue["body"] or "(없음)", human_request=human_request,
                  latest_review=latest_review[:6000], latest_tests=latest_tests[:3000])
    task = {"id": f"fix{done_rounds + 1}", "title": f"수정 라운드 {done_rounds + 1}"}
    try:
        report, ok = call("builder", prompt, cfg), True
    except RuntimeError as e:
        report, ok = str(e), False
    changed = bool(sh(["git", "status", "--porcelain"]).stdout.strip())
    if push and changed:
        sh(["git", "add", "-A"])
        sh(["git", "commit", "-m", f"ai-office: 수정 라운드 {done_rounds + 1} (PR #{pr_number})"])
    reports = [{"task": task, "ok": ok, "changed": changed, "report": report}]

    tests = run_tests(cfg)
    plan_data = {"mode": "PR 수정 라운드 — 아래 '이전 리뷰' 지적과 사람 지시를 고쳤는지 확인할 것",
                 "human_request": human_request, "previous_review": latest_review[:4000]}
    review_md = review(issue, plan_data, f"origin/{base}", reports, tests, cfg)
    verdict = parse_verdict(review_md)
    needs_human, banner = status_banner(verdict, reports, tests)
    if not changed:
        needs_human = True
    log(f"판정: {verdict} · 변경: {changed} · 사람 확인 필요: {needs_human}")

    mark = "✅" if ok and changed else ("⚠️ 변경 없음" if ok else "❌ 실패")
    body = "\n".join([
        FIX_MARKER, f"## 🔧 수정 라운드 {done_rounds + 1}/{max_rounds}", "", banner, "",
        f"**지시:** {human_request}", "",
        f"{mark} <details><summary>Builder 보고</summary>\n\n{report[:3000]}\n\n</details>", "",
        "### CI 테스트 (오케스트레이터가 직접 실행)", format_tests(tests), "",
        "### 자동 리뷰 (PR 전체 diff 기준)", re.sub(r"^VERDICT:.*\n?", "", review_md, count=1), "",
        "---",
        ("_더 고치려면 다시 `/ai-office fix` (지시를 뒤에 덧붙일 수 있음). "
         "Draft PR이면 확인 후 **Ready for review**를 누르세요._"),
    ])
    save_log("fix_comment.md", body)
    if not push:
        log(f"로컬 모드 종료. 결과: {LOG_DIR}/fix_comment.md")
        return 0
    if changed:
        sh(["git", "push", "origin", f"HEAD:{head}"])
    gh_api(f"issues/{pr_number}/comments", f"body={body[:60000]}", method="POST")
    set_needs_human(pr_number, needs_human)
    title = re.sub(r"^\[AI Office[^\]]*\]\s*", "", pr["title"])
    gh_api(f"pulls/{pr_number}", f"title={'[AI Office · 확인 필요] ' if needs_human else '[AI Office] '}{title}",
           method="PATCH", check=False)
    log("수정 라운드 결과를 PR에 남겼습니다.")
    return 0


# ---------- 메인 ----------

def load_issue(args) -> dict:
    issue = None
    if args.issue_title:
        issue = {"number": args.issue_number or 0, "title": args.issue_title, "body": args.issue_body or ""}
    else:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and Path(event_path).exists():
            ev = json.loads(Path(event_path).read_text(encoding="utf-8"))
            if "issue" in ev:
                i = ev["issue"]
                issue = {"number": i["number"], "title": i["title"], "body": i.get("body") or ""}
        num = args.issue_number or os.environ.get("ISSUE_NUMBER")
        if issue is None and num:  # workflow_dispatch
            i = gh_api(f"issues/{num}")
            issue = {"number": i["number"], "title": i["title"], "body": i.get("body") or ""}
    if issue is None:
        sys.exit("이슈 정보가 없습니다. --issue-title 또는 --issue-number를 주세요.")
    issue["comments"] = fetch_comments(issue["number"]) if not args.issue_title else []
    return issue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue-number", type=int)
    ap.add_argument("--issue-title")
    ap.add_argument("--issue-body")
    ap.add_argument("--plan-only", action="store_true", help="기획만 하고 종료")
    ap.add_argument("--no-push", action="store_true", help="커밋/푸시/PR 없이 로컬에서만 실행")
    ap.add_argument("--force", action="store_true", help="Planner 질문이 있어도 진행")
    ap.add_argument("--fix-pr", type=int, help="PR 수정 라운드 (/ai-office fix)")
    args = ap.parse_args()

    cfg = yaml.safe_load((OFFICE_DIR / "config.yaml").read_text(encoding="utf-8"))
    fix_pr = args.fix_pr or int(os.environ.get("AI_OFFICE_FIX_PR") or 0)
    if fix_pr:
        return run_fix(fix_pr, cfg, push=not args.no_push)
    issue = load_issue(args)
    log(f"이슈 #{issue['number']}: {issue['title']} (댓글 {len(issue['comments'])}개)")

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
    log(f"계획: 작업 {len(plan_data['tasks'])}개 · 막는 질문 {len(plan_data['questions'])}개 · 참고 {len(plan_data['notes'])}개")
    if args.plan_only:
        print(json.dumps(plan_data, ensure_ascii=False, indent=2))
        return 0

    push = not args.no_push
    asked_before, answered = question_state(issue["comments"])
    proceed_anyway = args.force or answered
    if plan_data["questions"] and not proceed_anyway:
        if push and issue["number"]:
            ask_questions(issue, plan_data)
        else:
            log("막는 질문이 있어 멈춥니다 (--force로 무시 가능):\n  - " + "\n  - ".join(plan_data["questions"]))
        return 0
    if plan_data["questions"]:
        log("질문이 남았지만 사람이 답했거나 강제 진행이라 계속합니다.")
    if not plan_data["tasks"]:
        log("작업이 없습니다.")
        return 0

    base = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    branch = f"ai-office/issue-{issue['number']}-{int(time.time())}"
    if push:
        sh(["git", "config", "user.name", "ai-office[bot]"], check=False)
        sh(["git", "config", "user.email", "ai-office@users.noreply.github.com"], check=False)
        sh(["git", "checkout", "-b", branch])

    reports = build(plan_data, cfg, commit=push)
    tests = run_tests(cfg)
    review_md = review(issue, plan_data, base, reports, tests, cfg)
    verdict = parse_verdict(review_md)
    needs_human, banner = status_banner(verdict, reports, tests)
    log(f"판정: {verdict} · 사람 확인 필요: {needs_human}")
    body = pr_body(issue, plan_data, reports, tests, review_md, banner, base) if push else \
        pr_body_local(issue, plan_data, reports, tests, review_md, banner, base)
    save_log("pr_body.md", body)

    if not push:
        log(f"로컬 모드 종료. 결과: {LOG_DIR}/pr_body.md")
        return 0
    if not any(r["changed"] for r in reports):
        log("변경된 파일이 없어 PR을 만들지 않습니다.")
        if issue["number"]:
            gh_api(f"issues/{issue['number']}/comments",
                   f"body=AI Office: 변경 사항 없이 끝났습니다.\n\n{body[:60000]}", method="POST", check=False)
        return 1

    sh(["git", "push", "-u", "origin", branch])
    url = create_pr(issue, branch, base, body, needs_human)
    log(f"PR 생성: {url}")
    return 0


def pr_body_local(*a) -> str:
    # 로컬 모드에선 GitHub 조회(default_branch) 없이 만든다
    global _default_branch_cache
    _default_branch_cache = _default_branch_cache or "main"
    return pr_body(*a)


if __name__ == "__main__":
    sys.exit(main())
