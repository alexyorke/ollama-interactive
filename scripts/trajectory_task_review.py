from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_profile
from scripts import trajectory_transcript_sampler as sampler


DEFAULT_DATA_ROOT = trajectory_profile.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT_JSON = DEFAULT_DATA_ROOT / "trajectory-task-review.json"
DEFAULT_OUTPUT_MD = DEFAULT_DATA_ROOT / "trajectory-task-review.md"

REAL_WORLD_FOCUSED_DATASETS = (
    "trace-commons-agent-traces",
    "agent-race-traces",
    "terminalbench-trajectories",
    "cc-bench-trajectories",
    "thoughtworks-agentic-coding-trajectories",
)

TASK_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("planning", ("project.md", "plan", "milestone", "roadmap", "stage 1", "architecture", "code quality", "maintainability", "style review", "inconsistent patterns", "poor naming", "poorly structured", "legacy debt", "compile a report", "security vulnerabilities", "@todo", "todos in the codebase", "inventory the @todos", "new release", "good to release", "release readiness", "review the .ts", "explain the behaviour", "understand how it works", "what does this code say", "does this code say", "preventing this codebase from being beautiful", "make recommendations", "run scan on this codebase")),
    ("bugfix", ("fix ", "failing", "broken", "bug", "error", "wrong", "regression", "debug", "properly appl", "sense-check", "why did this fail", "why this task failed", "not be able to use", "what's going on", "writes are done atomically", "locked for writing", "get 403", "403 when running", "reset back to queued", "keeps getting reset", "doesn't seem to work", "click handlers")),
    ("refactor", ("refactor", "restructure", "rename", "migrate", "update call", "call sites", "multi-file", "move the entire function", "move function", "package instead of", "crate instead of", "library instead of")),
    ("docs", ("readme", "docs", "documentation", "claude.md", "agents.md", "update the discord link")),
    ("setup-build", ("install", "setup", "set up", "run this please locally", "run this repo", "pushed to github", "launching the claude interface", "docker", "dependency", "configure", "makefile", "nginx.conf", "forwarded calls", "workflow_dispatch", ".github/workflows", "github api", "integration", "upload sessions", "to hf")),
    ("tests", ("test", "pytest", "unittest", "test case", "testcase", "coverage", "failing suite")),
    ("frontend-ui", ("react", "font", "css", "tailwind", "ui", "landing page", "website", "site", "tooltip", "hover")),
    ("data-analysis", ("dataset", "analy", "aggregate", "csv", "parquet", "vote", "instances that fail", "pass when", "syntactically incorrect", "extract all api endpoints", "fetches data from", "api endpoints", "inspect the apk")),
    ("feature", ("implement", "build", "create", "add ", "feature", "support", "translation layer", "live update", "real-time update", "remove the search", "write a script", "write a simple python script", "write a very simple script", "write a coordination script", "make a python script", "playwright script", "adapt the script", "adapt the code", "adapt", "download script", "adapt the benchmark", "flamegraph", "optimize them away", "also work for", "3x faster", "make this faster", "speed up", "finish the job", "started adapting it", "default quality is maximum", "shown in the upscale section", "upgrade my code", "write a bun backend", "replace all occurrences")),
)


def _truncate(text: str, limit: int = 180) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _review_prompt_preview(adapter: str, row: dict[str, Any]) -> str:
    prompt_preview = sampler._prompt_preview(adapter, row)
    if adapter == "terminalbench" and prompt_preview.strip().lower() == "warmup":
        task = sampler._task_label(row).strip()
        if task:
            return task
    return prompt_preview


def _classification_prompt_text(adapter: str, row: dict[str, Any]) -> str:
    direct_prompt = row.get("prompt")
    if isinstance(direct_prompt, str) and direct_prompt.strip():
        return direct_prompt.strip()
    if adapter in {"openhands", "cc_bench", "trace_commons", "thoughtworks"}:
        return sampler._find_openhands_message(row, "user").strip()
    if adapter == "terminalbench":
        prompt_text = sampler._find_terminalbench_prompt(row).strip()
        if prompt_text.lower() == "warmup":
            task = sampler._task_label(row).strip()
            if task:
                return task
        return prompt_text
    if adapter == "agent_race":
        return sampler._find_agent_race_message(row, "user").strip()
    task_name = row.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name.strip()
    return ""


def classify_task_family(*, task: str, prompt_preview: str) -> str:
    task_text = str(task or "").strip().lower()
    prompt_text = str(prompt_preview or "").strip().lower()
    combined = f"{task_text}\n{prompt_text}".strip()
    if not combined:
        return "other"
    if prompt_text in {"hello", "hi", "hey", "continue", "/load"}:
        return "control"
    if prompt_text == "say hello":
        return "control"
    if "write a bash command" in combined:
        return "control"
    if "what has hf released lately" in combined:
        return "control"
    if "use claude tool" in combined:
        return "control"
    if "/hf-cli" in combined and "trending papers" in combined:
        return "control"
    if "use ra_diagnostics on src/main.rs" in combined:
        return "control"
    if "what repo is this" in combined:
        return "control"
    if "what coding agent is this" in combined or "what coding agent is thsi" in combined:
        return "control"
    if prompt_text == "pi update":
        return "control"
    if "taille d un dossier" in combined and "commande" in combined:
        return "control"
    if "just ls" in combined:
        return "control"
    if prompt_text in {"read @justfile", "read justfile"}:
        return "control"
    if prompt_text == "list files in this directory":
        return "control"
    if "pls do ls" in combined:
        return "control"
    if "please do ls" in combined:
        return "control"
    if "safely deleted" in combined:
        return "control"
    if "candidates for deletion" in combined:
        return "control"
    if "free up space" in combined:
        return "control"
    if "push everything to github" in combined:
        return "control"
    if "push to github" in combined:
        return "control"
    if "commit work till now and push it" in combined:
        return "control"
    if "commit the changes so far" in combined and "push to current branch" in combined:
        return "control"
    if "see the pr we have on this repo" in combined:
        return "control"
    if "re-launch the front end" in combined:
        return "control"
    if "pushe verything to github" in combined:
        return "control"
    if "push this" in combined and "github.com/" in combined:
        return "control"
    if "unpushed changes" in combined:
        return "control"
    if "most recent issue" in combined:
        return "control"
    if "await instructions" in combined:
        return "control"
    if "start working on the task now" in combined:
        return "control"
    if "finish where you left off" in combined:
        return "control"
    if "what dir are you in" in combined:
        return "control"
    if "what project are" in combined and "you in" in combined:
        return "control"
    if prompt_text == "where are you":
        return "control"
    if "generate some text please" in combined:
        return "control"
    if "based only on the shell output above" in combined and "what os am i on" in combined:
        return "control"
    if "reply with exactly: ok" in combined:
        return "control"
    if "reply with just the number" in combined:
        return "control"
    if "what is 2+2" in combined:
        return "control"
    if "where cursor rules are stored" in combined:
        return "control"
    if "bump up the version" in combined:
        return "control"
    if "unzip all the files" in combined and "in parallel" in combined:
        return "control"
    if "bump the patch version" in combined:
        return "control"
    if ("context file" in combined or "context files" in combined) and (
        "loaded" in combined or "just list paths" in combined or "just the path" in combined
    ):
        return "control"
    if "find the pi sessions traces" in combined and "~/.pi/agent/sessions" in combined:
        return "control"
    if "weighted relative to the mechanical" in combined and "subjective" in combined:
        return "planning"
    if "summarize the changes needed to implement" in combined:
        return "planning"
    if "search for" in combined and "related files" in combined and "brief descriptions" in combined:
        return "planning"
    if "plan for building" in combined and (
        "tool" in combined
        or "assistant" in combined
        or "app" in combined
        or "player" in combined
        or "reader" in combined
        or "calculator" in combined
        or "travel" in combined
    ):
        return "feature"
    if "query key consistency" in combined and "hardcoded query keys" in combined and "cleanup target" in combined:
        return "planning"
    if "template code" in combined and ("add " in combined or "update " in combined or "make these changes" in combined):
        return "feature"
    if "make a sound whenever" in combined and "finished a task and is idle" in combined:
        return "feature"
    if "extracts all white text" in combined and "jsonl" in combined and "print progress" in combined:
        return "feature"
    if "write a new extraction tool" in combined and "extracts all text from pdfs into a jsonl" in combined:
        return "feature"
    if "here's db credentials" in combined and "discord_messages" in combined and "users/channels names table" in combined:
        return "data-analysis"
    if "what does" in combined and "videoitem.tsx" in combined and "where/how is it used" in combined:
        return "planning"
    if "how does this project work" in combined:
        return "planning"
    if "how does the " in combined and " plugin work" in combined:
        return "planning"
    if "what does the " in combined and " parameter do" in combined:
        return "planning"
    if "how does the " in combined and " handle images" in combined:
        return "planning"
    if "find the buckets-related code" in combined:
        return "planning"
    if "describe the status line of pi" in combined:
        return "planning"
    if "agent's tooling" in combined and "inspect where we are" in combined:
        return "planning"
    if "iclr2026_pdfs_text.jsonl" in combined and "em-dashes" in combined and "stddev" in combined:
        return "data-analysis"
    if "review this pr" in combined and "do you agree with suggestions" in combined:
        return "planning"
    if "use subagents" in combined and "research of the codebase" in combined and "bloat" in combined:
        return "planning"
    if "compare " in combined and "before giving me your report" in combined:
        return "planning"
    if "compare @" in combined and "subagents" in combined:
        return "planning"
    if "should i be linking" in combined and "subagents" in combined:
        return "planning"
    if "justfile" in combined and "just update" in combined:
        return "feature"
    if "update @justfile" in combined or "update the justfile" in combined:
        return "feature"
    if " vs " in combined and "pi-" in combined:
        return "planning"
    if "don't touch any code" in combined and "delightful" in combined:
        return "planning"
    if "categorize all the any usage" in combined and "streamlined" in combined:
        return "planning"
    if "problem summary" in combined and "root cause" in combined and "solution: minimal fix" in combined:
        return "bugfix"
    if "problem statement:" in combined and "bug:" in combined and "task context:" in combined:
        return "bugfix"
    if "cancellationexception" in combined and "prevent coroutine cancellation from propagating" in combined:
        return "bugfix"
    if "pi-share-hf extension" in combined and "collect the session automatically when pi shutdown" in combined:
        return "bugfix"
    if "sync-hf-sessions.sh" in combined and "only remove files if upload command succeeded" in combined:
        return "bugfix"
    if "only seeing articles from april 1st" in combined and "are we not updating" in combined:
        return "bugfix"
    if "economist demo" in combined and "indented differently" in combined and "investigate why" in combined:
        return "bugfix"
    if "media lightbox" in combined and "on ipad" in combined and "shifts weirdly" in combined:
        return "bugfix"
    if "lightboxes on ipad" in combined and "positoning weirdness" in combined:
        return "bugfix"
    if "render loop logs" in combined and "click into a shot" in combined:
        return "bugfix"
    if "issue stll happening" in combined and "github.com/" in combined and "/issues/" in combined:
        return "bugfix"
    if "spring" in combined and "logs console" in combined and "colore" in combined:
        return "bugfix"
    if "logback-spring.xml" in combined and "logs sur ma console" in combined and "pas de couleur" in combined:
        return "bugfix"
    if "instances currently registered with eureka" in combined and "service-gateway" in combined:
        return "bugfix"
    if "mediagallery" in combined and "replace it with an item from the next page" in combined and "empty spots" in combined:
        return "bugfix"
    if "generation pane" in combined and "doesn't seem to be working" in combined and "show no images" in combined:
        return "bugfix"
    if "edit images page" in combined and "selectors to click into different tools" in combined and "forms seem to be inputable" in combined:
        return "bugfix"
    if "reference section on the image generation form" in combined and "references are loading" in combined:
        return "bugfix"
    if "share button" in combined and "videotraveltoolpage.tsx" in combined and "isn't working" in combined:
        return "bugfix"
    if "duplicate an item" in combined and "@src/shared/hooks/timeline/" in combined and "skeleton" in combined:
        return "bugfix"
    if "default tag" in combined and "prompt and negative prompt fields" in combined:
        return "bugfix"
    if "load the websiet" in combined and "connecton isn't snappy" in combined and "jumps in" in combined:
        return "bugfix"
    if "default valjohn" in combined and "move mode" in combined and "default prompt" in combined:
        return "bugfix"
    if "structure video" in combined and "on ipad" in combined and "grab an end point" in combined:
        return "bugfix"
    if "segmentsettingsform.tsx" in combined and "can't scroll inside" in combined and "text fields" in combined:
        return "bugfix"
    if "media lightbox" in combined and "text field" in combined and "left slash right keys" in combined:
        return "bugfix"
    if "replace/browse button" in combined and "flows over the right" in combined:
        return "bugfix"
    if "escape key" in combined and "media lightbox stopped working" in combined:
        return "bugfix"
    if "floating shot selector" in combined and "ipad sizes" in combined and "behind the header" in combined:
        return "bugfix"
    if "image generaton modal" in combined and "blac background thing" in combined:
        return "bugfix"
    if "media lightbox" in combined and "variant count" in combined and "edit m" in combined:
        return "bugfix"
    if "press save" in combined and "'move' tool" in combined and "hangs indefinitely" in combined:
        return "bugfix"
    if "medialightbox" in combined and "task pane is locked" in combined and "appears behind it" in combined:
        return "bugfix"
    if "ipad with the timeline" in combined and "select a single image" in combined and "only shows when i select two images" in combined:
        return "bugfix"
    if "editvideopage.tsx" in combined and "tool selector isn't showing" in combined:
        return "bugfix"
    if "@supabase/functions/ai-prompt/" in combined and "a lot more poetic" in combined:
        return "bugfix"
    if "yellow time notices" in combined and "overlap with the other notices" in combined:
        return "bugfix"
    if "regenerate button" in combined and "doens't seem to be working" in combined:
        return "bugfix"
    if "medialightbox" in combined and "left/right keys don't work" in combined and "edit mode" in combined:
        return "bugfix"
    if "single image" in combined and "duration inside the segment model" in combined and "updates the time in the timeline" in combined:
        return "bugfix"
    if "medialightbox.tsx" in combined and "videos don't display correctly" in combined and "poster appears" in combined:
        return "bugfix"
    if "structure videos" in combined and "doesn't overlap" in combined:
        return "bugfix"
    if "ondeleteallprompts is not defined" in combined:
        return "bugfix"
    if "started happening more since you added" in combined and "investigate it" in combined:
        return "bugfix"
    if (
        "per module/subdirectory score" in combined
        or "best way to run multiple direct" in combined
        or ("git ignore" in combined and "subdirectory score" in combined)
    ):
        return "planning"
    if "subjective review" in combined and "code" in combined:
        return "planning"
    if "component" in combined and "impeccably structured" in combined:
        return "planning"
    if "page dropdown" in combined and "why is that different" in combined:
        return "planning"
    if "oversized hooks" in combined and "make a list" in combined:
        return "planning"
    if "agent instructions" in combined and "succinct" in combined:
        return "planning"
    if "agent instructons" in combined and "succint" in combined:
        return "planning"
    if "quick wins" in combined and "if they seem sensible" in combined:
        return "planning"
    if "well-named" in combined and "help command" in combined and "commands available" in combined:
        return "planning"
    if "prompts we give to agents" in combined and "nudge" in combined:
        return "planning"
    if "objective score" in combined and "subjective score" in combined and "strict score" in combined:
        return "planning"
    if "state of play" in combined and "comments" in combined:
        return "planning"
    if "state of play" in combined and "commments" in combined:
        return "planning"
    if "desloppify score" in combined and "improve" in combined:
        return "planning"
    if "decruftify" in combined and "missing areas" in combined and "bad patterns" in combined:
        return "planning"
    if "scan score" in combined and "llm" in combined and "nice table" in combined:
        return "planning"
    if "data fetching duplication" in combined and "which hook to use" in combined and "two hooks max" in combined:
        return "planning"
    if "structure.md" in combined and "philosophy" in combined:
        return "planning"
    if "subjective detectors" in combined and "codebase" in combined:
        return "planning"
    if "subjectivedetectors" in combined and "code-base" in combined:
        return "planning"
    if "usages of 'any'" in combined and "which are necessary" in combined and "which are sloppy" in combined:
        return "planning"
    if "subjective questions we ask of each language" in combined and "where htey're houses" in combined:
        return "planning"
    if "if they're confused" in combined and "why are they" in combined:
        return "planning"
    if "@scripts/decruftify/" in combined and "structure of this as a whole is good" in combined:
        return "planning"
    if "simplerealtimemanager.ts" in combined and "make a recommendaton" in combined:
        return "planning"
    if "what's a better name for this" in combined and "desloppify/" in combined:
        return "planning"
    if "don't pay him a ransom" in combined and "access to all of the data" in combined:
        return "planning"
    if "different tables that we're using" in combined and "any vulnerabilities" in combined:
        return "planning"
    if "different kinds of edit tasks" in combined and "optimistically submits the task" in combined:
        return "planning"
    if "skills.md" in combined and ("elegant" in combined or "well-explained" in combined or "any gaps" in combined):
        return "planning"
    if "general principles" in combined and ("refacoring" in combined or "refactoring" in combined):
        return "planning"
    if "why do we have" in combined and "imagegenerationtoolpage.tsx" in combined:
        return "planning"
    if "pr" in combined and "subjective questions" in combined and "codebase" in combined:
        return "planning"
    if "subjective questions" in combined and "assess" in combined:
        return "planning"
    if "open issues" in combined and "which of them are fixed" in combined and "should be closed" in combined:
        return "planning"
    if "isn't being used" in combined and ("properly implemented" in combined or "properly impelmeneted" in combined):
        return "planning"
    if ".env" in combined and "password" in combined:
        return "planning"
    if "latest issue" in combined and "good solution" in combined:
        return "planning"
    if "pull request" in combined and "assess it critically" in combined:
        return "planning"
    if "github commits since sunday" in combined and "summarise everyhing that was done" in combined:
        return "planning"
    if "language specific" in combined and ("properly wired up" in combined or "specific languages" in combined):
        return "planning"
    if "codebase" in combined and "beautiful and elegant" in combined:
        return "planning"
    if "codebase" in combined and "holistically beautiful" in combined:
        return "planning"
    if "codebase" in combined and "beautiful?" in combined and "elegant?" in combined:
        return "planning"
    if "fundamentally beautiful and elegant" in combined:
        return "planning"
    if "beautiful" in combined and (
        "your own assessment" in combined
        or "no tools" in combined
        or "without using" in combined
        or "any tools" in combined
    ):
        return "planning"
    if "communicate the requirements well" in combined:
        return "planning"
    if "code-base beautiful" in combined or ("code-base" in combined and "preventing it from being beautiful" in combined):
        return "planning"
    if "repo beautiful" in combined:
        return "planning"
    if "score now" in combined and "how can we improve" in combined:
        return "planning"
    if ("trict score" in combined or "strict score" in combined) and "closer to 100" in combined:
        return "planning"
    if "deslopify" in combined and "score higher than 95" in combined:
        return "planning"
    if "temp clone" in combined and "prompt it uses to compact" in combined:
        return "planning"
    if "desloppify scan" in combined and "95+" in combined:
        return "planning"
    if "desloppify scan" in combined and "codebase" in combined:
        return "planning"
    if "desloppify scan" in combined and "share the score and issues" in combined:
        return "planning"
    if "code paste" in combined and "holistically beautiful and well engineered" in combined:
        return "planning"
    if "run a scan on this repo" in combined and "tackle all of the open issues" in combined:
        return "planning"
    if "open desloppify issues" in combined and "run it using this repo's own code" in combined:
        return "planning"
    if "run a desloppify scan" in combined and "let me know if you spot anything" in combined:
        return "planning"
    if "run a desloppify scan" in combined and "next command" in combined and "presenting information" in combined:
        return "planning"
    if "revavluate subjective iitems" in combined and "<90" in combined:
        return "planning"
    if "scan narrative" in combined and "rerun subjective analsis" in combined:
        return "planning"
    if "documents/desloppify" in combined and "improve th score" in combined:
        return "planning"
    if "documents/desloppify" in combined and "strict score" in combined and "close to 100%" in combined:
        return "planning"
    if "latest documents/desloppify" in combined and "share the strict score" in combined:
        return "planning"
    if "documnets/desloppify" in combined and "strict score" in combined and "close as possible to 100" in combined:
        return "planning"
    if "bugfix:" in combined and "how is this possible" in combined:
        return "bugfix"
    if "jump into the shot context" in combined and "similar thing with images" in combined:
        return "feature"
    if "trailing video thing on timeline" in combined and "always shows it" in combined:
        return "feature"
    if "segmentsettingsform.tsx" in combined and "can you not show these fields" in combined:
        return "feature"
    if "edit button" in combined and "media light box" in combined and "remove that" in combined:
        return "feature"
    if "chop the last vdieo uploaded in two vertically" in combined:
        return "feature"
    if "put those last 2 downloaded videos side by side" in combined:
        return "feature"
    if "head_swap_compare_00018-audio.mp4" in combined and "small enough so it works in a doc" in combined:
        return "feature"
    if "last two downloaded images and videos" in combined and "example image two fades in" in combined:
        return "feature"
    if "medialightbox" in combined and "excessively large" in combined and "reduce the line count and complexity" in combined:
        return "refactor"
    if "mobileimageitem.tsx" in combined and "dead code" in combined:
        return "refactor"
    if "dimensions of the last two downloaded files" in combined and "difference between them" in combined:
        return "data-analysis"
    if "voice box" in combined and "takes up two rows" in combined:
        return "frontend-ui"
    if "generation slider" in combined and "variant toggle" in combined and "80% of the width" in combined:
        return "frontend-ui"
    if "closes the lightbox" in combined and "continue drawing at the edge" in combined:
        return "bugfix"
    if "transitions between the videos" in combined and "video one to video two" in combined:
        return "bugfix"
    if "deleted all the structure videos on the timeline" in combined and "empty state" in combined:
        return "bugfix"
    if "batch mode" in combined and "0:0 slot empty" in combined:
        return "bugfix"
    if "layout" in combined and "looks weird" in combined:
        return "bugfix"
    if "recent git commits" in combined and "still seems to be showing" in combined:
        return "bugfix"
    if "image generation form" in combined and "optimistic placeholder" in combined and "task pane" in combined:
        return "bugfix"
    if "taskspane.tsx" in combined and "on ipad" in combined and "pane only closes when i tap outside" in combined:
        return "bugfix"
    if "select a preset" in combined and ("doesn't seem to actually use" in combined or "pass the preset id" in combined):
        return "bugfix"
    if "segmentregenerateform.tsx" in combined and "frames per pair" in combined and "update on the timeline" in combined:
        return "bugfix"
    if "upscaling" in combined and "medialightbox" in combined and "doesn't seem to track" in combined:
        return "bugfix"
    if "delete a timeline item" in combined and "leaves an extra item" in combined:
        return "bugfix"
    if "varant selector isn't showing" in combined and "medialightbox" in combined and "edit mode" in combined:
        return "bugfix"
    if "fill images with ai" in combined and "black outline border" in combined and "media lightbox" in combined:
        return "bugfix"
    if "click into the gallery" in combined and "co.paddingbottom is not a function" in combined:
        return "bugfix"
    if "why this task got set to the parent generaton too" in combined:
        return "bugfix"
    if "github issue" in combined and "don't make it into the next queue" in combined:
        return "bugfix"
    if "on mobile" in combined and "advanced mode container" in combined and "run over the right side" in combined:
        return "bugfix"
    if "one-to-one mapping versus fit to range" in combined and (
        "moves behind the action bar" in combined or "moves behind the actual timeline" in combined
    ):
        return "bugfix"
    if "add it to " in combined and ".json" in combined:
        return "feature"
    if "terminal tab status" in combined and "done replying" in combined:
        return "feature"
    if "snake extension" in combined and "space invaders" in combined:
        return "feature"
    if "round out our arcade" in combined and ("tetris" in combined or "pacman" in combined or "mario" in combined):
        return "feature"
    if "doom" in combined and "pi extension" in combined:
        return "feature"
    if "sft smollm2" in combined and "capybara" in combined:
        return "feature"
    if "train smollm3" in combined and "agenttrove" in combined:
        return "feature"
    if "code " in combined and "python" in combined and "brute-force approach" in combined:
        return "feature"
    if "default enhance prompt to false" in combined:
        return "feature"
    if "claude.md" in combined and (
        "complete " in combined
        or "implementation" in combined
        or "start building" in combined
        or "help you create" in combined
        or "help you start building" in combined
        or "bootstrap" in combined
        or "building the " in combined
        or "creating a " in combined
        or "development of the " in combined
    ) and "read-only mode" not in combined and "planning mode" not in combined:
        return "feature"
    if "implement commit" in combined and (
        "scope:" in combined
        or "requirements:" in combined
        or "add commands/modes" in combined
        or "add extension file" in combined
    ):
        return "feature"
    if "do not touch readme" in combined and (
        "implement commit" in combined
        or "add extension file" in combined
        or "register /" in combined
    ):
        return "feature"
    if "pick up " in combined and ".md" in combined:
        return "docs"
    if "clone the following repo" in combined and "into a page" in combined:
        return "feature"
    if "single-purpose edge function" in combined and "returns its current status" in combined:
        return "feature"
    if "media lightbox" in combined and "load settings" in combined and "load images" in combined:
        return "feature"
    if "finalvideosection.tsx" in combined and (
        "show it instead of the images" in combined or "shot it instead of the images" in combined
    ):
        return "feature"
    if "finalvideosection.tsx" in combined and "copy button" in combined and "doesn't show a sucess state" in combined:
        return "bugfix"
    if "finalvideosection.tsx" in combined and "dragging over that secton" in combined and "opens the lightbox" in combined:
        return "bugfix"
    if "timeline" in combined and "notification" in combined and "output strip" in combined:
        return "frontend-ui"
    if "loading screen" in combined and "design this" in combined:
        return "frontend-ui"
    if "left/right" in combined and "navigate between sections" in combined and "first/last page" in combined:
        return "frontend-ui"
    if "generation pane" in combined and "see all images on mobile" in combined and "shot drop down" in combined:
        return "frontend-ui"
    if "tap timeline to place" in combined and ("touch devices" in combined or "ipads" in combined):
        return "frontend-ui"
    if "ipad" in combined and "batch mode" in combined and "two images per line" in combined and "shot images editor" in combined:
        return "frontend-ui"
    if "assets that show on the page less heavy" in combined and "unoptimized for the web" in combined:
        return "frontend-ui"
    if "image gallery" in combined and "actual size of the screen" in combined:
        return "frontend-ui"
    if "few frontend things" in combined and "queue goes over the thinking text" in combined:
        return "frontend-ui"
    if "mediagallery" in combined and "edit button on each item" in combined:
        return "frontend-ui"
    if "medialightbox" in combined and "lineage gif thing" in combined and "more than 5 depth lineage" in combined:
        return "frontend-ui"
    if "travel segment" in combined and "taskspane" in combined and "segmentoutputstrip.tsx" in combined:
        return "frontend-ui"
    if "lines of code" in combined and "past three months" in combined and "every repo" in combined:
        return "data-analysis"
    if "playwright" in combined and "script" in combined:
        return "feature"
    if (
        "generate the source code" in combined
        or "implement the source code" in combined
        or ("source code" in combined and "test case" in combined)
        or ("source code" in combined and "documentation" in combined)
    ):
        return "feature"
    if "pypi" in combined and "register" in combined:
        return "setup-build"
    if "default model" in combined and "pi settings" in combined:
        return "setup-build"
    if "demo of the researcher subagent" in combined:
        return "setup-build"
    if "showcase the researcher agent" in combined or "showcase a researcher agent" in combined:
        return "setup-build"
    if "start android" in combined and "emulator" in combined and "run the android app" in combined:
        return "setup-build"
    if "watch emulator" in combined and "pair them" in combined and "adb commands" in combined:
        return "setup-build"
    if "prettier config" in combined and "format json" in combined:
        return "feature"
    if "make a sound whenever" in combined and "finished a task and is idle" in combined:
        return "feature"
    if "generate new wallets for older challenges" in combined and "less difficulty than new challenges" in combined:
        return "feature"
    if "rewrite the miner logic like this" in combined and "challenges.json" in combined:
        return "feature"
    if "session links" in combined and "harmonize them" in combined:
        return "frontend-ui"
    if "find/remove any shims" in combined and "update tests if need be" in combined:
        return "refactor"
    if "commit authorship" in combined:
        return "setup-build"
    if "deploy it to npm" in combined and "cli command" in combined:
        return "setup-build"
    if "deploy @supabase/functions/" in combined:
        return "setup-build"
    if "decruftify" in combined and "scan" in combined and "please" in combined:
        return "control"

    def _matches_keyword(keyword: str) -> bool:
        normalized = keyword.strip().lower()
        if not normalized:
            return False
        if any(char.isspace() for char in normalized):
            return normalized in combined
        if normalized in {"adapt", "plan", "test", "tests"}:
            pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
            return re.search(pattern, combined) is not None
        if len(normalized) <= 3 and normalized.isalpha():
            pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
            return re.search(pattern, combined) is not None
        return normalized in combined

    for family, keywords in TASK_FAMILY_KEYWORDS:
        if any(_matches_keyword(keyword) for keyword in keywords):
            return family
    return "other"


def _insight_flags(summary: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    context_loop = float(summary.get("context_loop_rows_pct", 0.0) or 0.0)
    edit_without_test = float(summary.get("edit_without_later_test_pct", 0.0) or 0.0)
    avg_tool_calls = float(summary.get("avg_tool_calls", 0.0) or 0.0)
    mechanical = float(summary.get("mechanical_turn_candidates_pct", 0.0) or 0.0)
    if context_loop >= 20:
        flags.append("search-or-read loops stay open too long before progress")
    if edit_without_test >= 25:
        flags.append("edits frequently land without later validation")
    if avg_tool_calls >= 60:
        flags.append("sessions are expensive enough that startup and retry cost matter")
    if mechanical >= 5:
        flags.append("a meaningful slice looks deterministic enough for direct routing")
    return flags


def _aggregate_recommendations(dataset_summaries: list[dict[str, Any]]) -> list[str]:
    flags: Counter[str] = Counter()
    for summary in dataset_summaries:
        for item in _insight_flags(summary):
            flags[item] += 1
    ordered = [name for name, _ in flags.most_common()]
    recommendations: list[str] = []
    if "search-or-read loops stay open too long before progress" in ordered:
        recommendations.append("Tighten repeated read/search loop compression before adding more optional tools.")
    if "edits frequently land without later validation" in ordered:
        recommendations.append("Make post-edit validation harder to skip, especially for shell-driven and non-Python tasks.")
    if "sessions are expensive enough that startup and retry cost matter" in ordered:
        recommendations.append("Trim startup prompt/tool exposure on greenfield and long-running sessions.")
    if "a meaningful slice looks deterministic enough for direct routing" in ordered:
        recommendations.append("Expand deterministic routing only for measured mechanical turns such as status, search, and validator flows.")
    return recommendations


def _family_examples(examples: dict[str, list[str]], *, limit: int) -> dict[str, list[str]]:
    return {
        family: values[:limit]
        for family, values in sorted(examples.items(), key=lambda item: (-len(item[1]), item[0]))
        if values
    }


def review_dataset(
    data_root: Path,
    dataset: str,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    adapter, rows = sampler._iter_dataset_rows(data_root, dataset, max_rows)
    cached_rows = list(rows)
    profile_summary = trajectory_profile._summarize_dataset(dataset, adapter, iter(cached_rows))
    family_counts: Counter[str] = Counter()
    family_examples: dict[str, list[str]] = defaultdict(list)
    for row in cached_rows:
        task = sampler._task_label(row)
        prompt_text = _classification_prompt_text(adapter, row)
        prompt_preview = _truncate(prompt_text) if prompt_text else _review_prompt_preview(adapter, row)
        family = classify_task_family(task=task, prompt_preview=prompt_text)
        family_counts[family] += 1
        display_prompt = prompt_preview if prompt_preview else ""
        if display_prompt and display_prompt not in family_examples[family] and len(family_examples[family]) < examples_per_family:
            family_examples[family].append(display_prompt)
    return {
        "dataset": dataset,
        "adapter": adapter,
        "rows_scanned": int(profile_summary.get("rows_profiled", 0) or 0),
        "avg_tool_calls": profile_summary.get("avg_tool_calls", 0.0),
        "context_loop_rows_pct": profile_summary.get("context_loop_rows_pct", 0.0),
        "edit_without_later_test_pct": profile_summary.get("edit_without_later_test_pct", 0.0),
        "mechanical_turn_candidates_pct": profile_summary.get("mechanical_turn_candidates_pct", 0.0),
        "top_repeated_loops": list(profile_summary.get("top_repeated_loops") or [])[:6],
        "task_family_counts": dict(family_counts.most_common()),
        "task_family_examples": _family_examples(family_examples, limit=examples_per_family),
        "insight_flags": _insight_flags(profile_summary),
    }


def build_review(
    *,
    data_root: Path,
    datasets: list[str],
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    dataset_summaries = [
        review_dataset(
            data_root,
            dataset,
            max_rows=max_rows,
            examples_per_family=examples_per_family,
        )
        for dataset in datasets
    ]
    aggregate_counts: Counter[str] = Counter()
    for summary in dataset_summaries:
        aggregate_counts.update(summary.get("task_family_counts", {}))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve(strict=False)),
        "datasets": dataset_summaries,
        "aggregate_task_family_counts": dict(aggregate_counts.most_common()),
        "top_recommendations": _aggregate_recommendations(dataset_summaries),
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Trajectory Task Review", ""]
    aggregate_counts = payload.get("aggregate_task_family_counts") or {}
    if isinstance(aggregate_counts, dict) and aggregate_counts:
        lines.append("## Aggregate Task Mix")
        lines.append("")
        for family, count in aggregate_counts.items():
            lines.append(f"- `{family}`: `{count}`")
        lines.append("")
    recommendations = payload.get("top_recommendations") or []
    if recommendations:
        lines.append("## Top Recommendations")
        lines.append("")
        for item in recommendations:
            lines.append(f"- {item}")
        lines.append("")
    for summary in payload.get("datasets", []):
        if not isinstance(summary, dict):
            continue
        lines.append(f"## {summary.get('dataset')}")
        lines.append("")
        lines.append(f"- adapter: `{summary.get('adapter')}`")
        lines.append(f"- rows_scanned: `{summary.get('rows_scanned')}`")
        lines.append(f"- avg_tool_calls: `{summary.get('avg_tool_calls')}`")
        lines.append(f"- context_loop_rows_pct: `{summary.get('context_loop_rows_pct')}`")
        lines.append(f"- edit_without_later_test_pct: `{summary.get('edit_without_later_test_pct')}`")
        flags = summary.get("insight_flags") or []
        if flags:
            lines.append("- insight_flags: " + "; ".join(flags))
        family_counts = summary.get("task_family_counts") or {}
        if isinstance(family_counts, dict) and family_counts:
            lines.append("- task_families: " + ", ".join(f"`{key}` ({value})" for key, value in family_counts.items()))
        lines.append("")
        family_examples = summary.get("task_family_examples") or {}
        for family, examples in family_examples.items():
            if not isinstance(examples, list) or not examples:
                continue
            lines.append(f"### {family}")
            lines.append("")
            for example in examples:
                lines.append(f"- {example}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_output_paths(
    *,
    output: Path | None,
    output_json: Path | None,
    output_md: Path | None,
) -> tuple[Path, Path]:
    if output is None:
        return output_json or DEFAULT_OUTPUT_JSON, output_md or DEFAULT_OUTPUT_MD
    suffix = output.suffix.lower()
    if suffix == ".json":
        return output_json or output, output_md or output.with_suffix(".md")
    if suffix == ".md":
        return output_json or output.with_suffix(".json"), output_md or output
    return output_json or output.with_suffix(".json"), output_md or output.with_suffix(".md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review coding-agent transcript datasets to summarize task types and token-efficiency gaps.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing local trajectory datasets.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(trajectory_profile.DATASET_SPECS), default=list(REAL_WORLD_FOCUSED_DATASETS))
    parser.add_argument("--max-rows", type=int, default=150, help="Maximum rows to scan per dataset. Use 0 for all rows.")
    parser.add_argument("--examples-per-family", type=int, default=2, help="Prompt examples to keep per task family.")
    parser.add_argument("--output", type=Path, default=None, help="Base output path or explicit .json/.md path; writes both JSON and Markdown siblings.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_json, output_md = _resolve_output_paths(
        output=args.output,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    max_rows = None if args.max_rows == 0 else args.max_rows
    payload = build_review(
        data_root=args.data_root,
        datasets=list(args.datasets),
        max_rows=max_rows,
        examples_per_family=args.examples_per_family,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_format_markdown(payload), encoding="utf-8")
    for summary in payload["datasets"]:
        print(
            "[trajectory-task-review] "
            + f"dataset={summary['dataset']} "
            + f"rows={summary['rows_scanned']} "
            + f"families={len(summary['task_family_counts'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
