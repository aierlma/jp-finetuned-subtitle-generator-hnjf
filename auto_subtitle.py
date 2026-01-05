#!/usr/bin/env python3
"""
Pipeline wrapper around whisper-large-v2-translate-zh-v0.2-st-ct2-v0.10.

Drag a folder (or file) onto run_auto_subs.bat to:
- copy videos from a source folder (e.g. network drive) into a local staging area,
- run the existing GPU translator (infer.exe) to create SRT subtitles,
- store subtitles in a dedicated output folder mirroring the source structure,
- remove staged videos after each translation.

Tuning:
- Update INFER_EXE below if you move the translation tool.
- Adjust WORK_ROOT if you want a different staging/output location.
- MAX_LOCAL caps how many videos can sit in the local staging folder at once.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# Path to the existing translator (GPU script).
INFER_EXE = Path(
    r"D:\Downloads\hnjf\whisper-large-v2-translate-zh-v0.2-st-ct2-v0.10\infer.exe"
)

# Where to store temporary copies and final subtitles.
WORK_ROOT = Path(__file__).resolve().parent / "auto_sub_work"
STAGING_DIR = WORK_ROOT / "staging"
OUTPUT_ROOT = WORK_ROOT / "subtitles"

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ogv",
    ".f4v",
}
SUB_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".sub", ".sup"}
MAX_LOCAL = 10


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class VideoTask:
    source: Path
    base_root: Path
    relative: Path
    output_dir: Path
    output_file: Path


def has_local_subtitle(video: Path) -> bool:
    return any(video.with_suffix(ext).exists() for ext in SUB_EXTS)


def build_relative(base_root: Path, file_path: Path) -> Path:
    """Include a slice of the source path to distinguish different folders."""
    parts = list(base_root.parts)
    # Drop drive/anchor (e.g., "C:\\" or "\\\\server\\share\\") and keep the last two folders.
    filtered = [
        p
        for p in parts
        if p not in {base_root.drive, base_root.anchor, base_root.root}
    ]
    prefix_parts = filtered[-2:] if filtered else ([base_root.name] if base_root.name else [])
    return Path(*prefix_parts) / file_path.relative_to(base_root)


def collect_tasks(inputs: Iterable[Path]) -> List[VideoTask]:
    tasks: List[VideoTask] = []
    skipped = 0

    for raw in inputs:
        src = raw.resolve()
        if not src.exists():
            log(f"[SKIP] Missing path: {src}")
            skipped += 1
            continue

        if src.is_file():
            candidates = [src]
            base_root = src.parent
        else:
            candidates = [p for p in src.rglob("*") if p.is_file()]
            base_root = src

        for file_path in candidates:
            if file_path.suffix.lower() not in VIDEO_EXTS:
                continue

            relative = build_relative(base_root, file_path)
            output_dir = OUTPUT_ROOT / relative.parent
            output_file = output_dir / (file_path.stem + ".srt")

            if has_local_subtitle(file_path) or output_file.exists():
                skipped += 1
                continue

            tasks.append(
                VideoTask(
                    source=file_path,
                    base_root=base_root,
                    relative=relative,
                    output_dir=output_dir,
                    output_file=output_file,
                )
            )

    log(f"[INFO] Planned {len(tasks)} video(s); skipped {skipped} already subtitled.")
    return tasks


def stage_file(task: VideoTask, semaphore: threading.Semaphore) -> Path:
    semaphore.acquire()
    staged_path = STAGING_DIR / task.relative
    staged_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"[COPY] {task.source} -> {staged_path}")
    shutil.copy2(task.source, staged_path)

    if staged_path.suffix.lower() == ".wmv":
        mp4_path = staged_path.with_suffix(".mp4")
        log(f"[TRANSCODE] {staged_path} -> {mp4_path}")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(staged_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    str(mp4_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            # Clean up partially written mp4 if transcode fails.
            if mp4_path.exists():
                mp4_path.unlink()
            raise

        try:
            staged_path.unlink()
        except FileNotFoundError:
            pass
        staged_path = mp4_path

    return staged_path


def run_infer(staged_file: Path, task: VideoTask) -> None:
    task.output_dir.mkdir(parents=True, exist_ok=True)

    audio_suffixes = ",".join(sorted({ext.lstrip(".") for ext in VIDEO_EXTS}))

    cmd = [
        str(INFER_EXE),
        f'--audio_suffixes="{audio_suffixes}"',
        "--sub_formats=srt",
        "--device=cuda",
        str(staged_file),
    ]

    log(f"[RUN ] {task.source} -> {task.output_file}")
    try:
        subprocess.run(cmd, cwd=INFER_EXE.parent, check=True)
    except subprocess.CalledProcessError as exc:
        # Some infer.exe builds may return non-zero even when an .srt is emitted.
        staged_sub = staged_file.with_suffix(".srt")
        if staged_sub.exists():
            log(f"[WARN] infer.exe exited with code {exc.returncode}, but subtitle was created; moving it.")
        else:
            raise

    staged_sub = staged_file.with_suffix(".srt")
    if not staged_sub.exists():
        raise RuntimeError("infer.exe finished but subtitle file was not created.")

    shutil.move(str(staged_sub), str(task.output_file))


def worker_copy_queue(
    copy_q: queue.Queue,
    ready_q: queue.Queue,
    semaphore: threading.Semaphore,
    stop_event: threading.Event,
) -> None:
    while True:
        if stop_event.is_set() and copy_q.empty():
            ready_q.put(None)
            break

        try:
            task = copy_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            ready_q.put(None)
            break

        try:
            staged = stage_file(task, semaphore)
        except Exception as exc:  # pylint: disable=broad-except
            log(f"[FAIL] Copy failed for {task.source}: {exc}")
            semaphore.release()
            continue
        ready_q.put((task, staged))


def worker_process(
    ready_q: queue.Queue,
    semaphore: threading.Semaphore,
    stats: dict,
    resume_event: threading.Event,
) -> None:
    while True:
        resume_event.wait()
        item = ready_q.get()
        if item is None:
            ready_q.put(None)
            break

        task, staged_path = item
        try:
            run_infer(staged_path, task)
            stats["done"] += 1
            log(f"[DONE] {task.output_file}")
        except Exception as exc:  # pylint: disable=broad-except
            stats["failed"].append((task.source, str(exc)))
            log(f"[FAIL] {task.source}: {exc}")
        finally:
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass
            try:
                staged_path.with_suffix(".srt").unlink()
            except FileNotFoundError:
                pass
            semaphore.release()


def prepare_workspace() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Clean up any leftovers in staging from earlier runs.
    for residual in STAGING_DIR.rglob("*"):
        if residual.is_file():
            residual.unlink()


def control_loop(
    copy_q: queue.Queue,
    resume_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    log("[CTRL] Commands: 'p' pause/resume after current video | 'a <path>' append folder/file | 'q' finish and exit.")
    while True:
        try:
            raw = input().strip()
        except EOFError:
            stop_event.set()
            resume_event.set()
            copy_q.put(None)
            break

        if not raw:
            continue

        lower = raw.lower()
        if lower == "p":
            if resume_event.is_set():
                resume_event.clear()
                log("[CTRL] Pause requested. Will pause after the current video.")
            else:
                resume_event.set()
                log("[CTRL] Resumed.")
            continue

        if lower == "q":
            log("[CTRL] Finish requested. Will stop after queued work.")
            stop_event.set()
            resume_event.set()
            copy_q.put(None)
            break

        # Treat anything else as an append path (supports 'a <path>' or just a path dragged in).
        if lower.startswith("a "):
            path_text = raw[2:].strip()
        else:
            path_text = raw

        if not path_text:
            continue

        new_inputs = [Path(path_text.strip('"'))]
        new_tasks = collect_tasks(new_inputs)
        if not new_tasks:
            log("[CTRL] No new videos to add (already subtitled or invalid).")
            continue

        for t in new_tasks:
            copy_q.put(t)

        log(f"[CTRL] Appended {len(new_tasks)} task(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy videos locally, run infer.exe to generate subtitles, and clean up."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Folder(s) or file(s) to process. Drag and drop a folder onto the .bat to fill this.",
    )
    return parser.parse_args()


def main() -> None:
    if not INFER_EXE.exists():
        raise FileNotFoundError(f"infer.exe not found at {INFER_EXE}")

    args = parse_args()
    inputs = [Path(p) for p in args.inputs]

    prepare_workspace()
    tasks = collect_tasks(inputs)
    if not tasks:
        log("[INFO] Nothing to do.")
        return

    ready_q: queue.Queue = queue.Queue(maxsize=MAX_LOCAL)
    copy_q: queue.Queue = queue.Queue()
    semaphore = threading.Semaphore(MAX_LOCAL)
    stats = {"done": 0, "failed": []}
    resume_event = threading.Event()
    stop_event = threading.Event()
    resume_event.set()

    for task in tasks:
        copy_q.put(task)

    copy_thread = threading.Thread(
        target=worker_copy_queue,
        args=(copy_q, ready_q, semaphore, stop_event),
        daemon=True,
    )
    process_thread = threading.Thread(
        target=worker_process,
        args=(ready_q, semaphore, stats, resume_event),
        daemon=True,
    )
    control_thread = threading.Thread(
        target=control_loop, args=(copy_q, resume_event, stop_event), daemon=True
    )

    copy_thread.start()
    process_thread.start()
    control_thread.start()

    copy_thread.join()
    process_thread.join()

    log(f"[SUMMARY] Completed: {stats['done']} | Failed: {len(stats['failed'])}")
    if stats["failed"]:
        for src, reason in stats["failed"]:
            log(f"         - {src}: {reason}")


if __name__ == "__main__":
    main()
