from __future__ import annotations

import argparse
import base64
import contextlib
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


DEFAULT_OPENCODE_URL = "http://127.0.0.1:4096"
DEFAULT_CLEANUP_PORTS = (4096, 5173, 5174, 3000, 3001, 8000, 8080)
DEFAULT_TIMEOUT_SECONDS = 1800
MAX_LOG_FIELD_LENGTH = 600
MAX_LOG_JSON_LENGTH = 1400
REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "my-app"
OUTPUT_DIR = Path(os.environ.get("UI_REPLICATE_OUTPUT_DIR", REPO_ROOT / "runs" / "local"))
DEFAULT_SCREENSHOT_PATH = OUTPUT_DIR / "target.png"
AI_GENERATED_SCREENSHOT_PATH = OUTPUT_DIR / "ai-generated.png"
LOCAL_APP_URL = "http://localhost:5173"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_collection_utilities.screenshot import capture_screenshot


Runner = Callable[[list[str]], object]


def build_message_payload(task: str, *, screenshot_path: Path) -> dict[str, Any]:
    parts = [{"type": "text", "text": task}]
    parts.append({"type": "file", "mime": image_mime_type(screenshot_path), "url": image_data_url(screenshot_path)})
    return {"parts": parts}


def image_mime_type(image_path: Path) -> str:
    return mimetypes.guess_type(image_path.name)[0] or "image/png"


def image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{image_mime_type(image_path)};base64,{encoded}"


def normalize_image_url(url: str) -> str:
    if "dropbox.com" not in url:
        return url

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.pop("raw", None)
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def download_reference_image(image_url: str, destination_path: Path) -> Path:
    normalized_url = normalize_image_url(image_url)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(normalized_url, headers={"User-Agent": "ui-replicate/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            destination_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to download reference image with HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to download reference image: {error.reason}") from error
    return destination_path


def unique_ports(ports: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for port in ports:
        if port not in seen:
            seen.add(port)
            ordered.append(port)
    return ordered


def default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def start_dev_server(directory: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["bun", "run", "dev"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def wait_for_url(url: str, *, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception as error:
            last_error = error
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def stop_dev_server(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def kill_listeners_on_ports(ports: Iterable[int], *, runner: Runner = default_runner) -> list[int]:
    killed: list[int] = []
    if shutil.which("lsof"):
        for port in unique_ports(ports):
            pids = listener_pids_for_port(port, runner=runner)
            terminate_pids(pids)
            if pids:
                wait_for_port_to_close(port, timeout=2)
                remaining = listener_pids_for_port(port, runner=runner)
                terminate_pids(remaining, sig=signal.SIGKILL)
            killed.append(port)
        return killed

    fuser_command_builder = fuser_kill_command_builder()
    if fuser_command_builder is None:
        print("warning: neither fuser nor lsof is available; skipping port cleanup", file=sys.stderr)
        return killed

    for port in unique_ports(ports):
        runner(fuser_command_builder(port))
        killed.append(port)
    return killed


def listener_pids_for_port(port: int, *, runner: Runner = default_runner) -> list[int]:
    result = runner(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"])
    stdout = getattr(result, "stdout", "") or ""
    pids: list[int] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            pids.append(int(line))
    return pids


def terminate_pids(pids: Iterable[int], *, sig: signal.Signals = signal.SIGTERM) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            print(f"warning: could not kill pid {pid}: {error}", file=sys.stderr)


def wait_for_port_to_close(port: int, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = default_runner(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"])
        if not (result.stdout or "").strip():
            return
        time.sleep(0.1)


def fuser_kill_command_builder() -> Callable[[int], list[str]] | None:
    if shutil.which("fuser"):
        return lambda port: ["fuser", "-k", f"{port}/tcp"]
    return None


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    directory: Path = APP_DIR,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    body = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers=build_headers(directory),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            print(f"\n--- OpenCode response: {method} {url} ---", file=sys.stderr)
            print(response_body, file=sys.stderr)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{method} {url} failed: {error.reason}") from error

    return json.loads(response_body) if response_body else {}


def log_runner(message: str) -> None:
    print(f"[runner] {message}", file=sys.stderr, flush=True)


def build_headers(directory: Path) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "x-opencode-directory": str(directory),
    }
    username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    password = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    return headers


def run_task(
    task: str,
    *,
    screenshot_path: Path,
    base_url: str = DEFAULT_OPENCODE_URL,
    directory: Path = APP_DIR,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    session_id: str | None = None
    stop_stream = Event()
    event_queue: Queue[dict[str, Any]] = Queue()
    stream_thread: Thread | None = None
    try:
        health = request_json("GET", f"{base_url}/global/health", directory=directory, timeout=30)
        if not health.get("healthy"):
            raise RuntimeError(f"OpenCode server is not healthy: {health}")

        session = request_json("POST", f"{base_url}/session", payload={}, directory=directory, timeout=30)
        session_id = session["id"]
        stream_thread = Thread(
            target=stream_session_events,
            args=(base_url, directory, event_queue, stop_stream),
            daemon=True,
        )
        stream_thread.start()

        request_json(
            "POST",
            f"{base_url}/session/{session_id}/prompt_async",
            payload=build_message_payload(task, screenshot_path=screenshot_path),
            directory=directory,
            timeout=timeout,
        )
        return wait_for_session_completion(session_id, base_url=base_url, directory=directory, queue=event_queue, stop_stream=stop_stream)
    finally:
        stop_stream.set()
        if stream_thread is not None:
            stream_thread.join(timeout=2)
        if session_id:
            safe_request("POST", f"{base_url}/session/{session_id}/abort", directory=directory, timeout=10)
        safe_request("POST", f"{base_url}/instance/dispose", directory=directory, timeout=10)
        kill_listeners_on_ports(DEFAULT_CLEANUP_PORTS)


def wait_for_session_completion(
    session_id: str,
    *,
    base_url: str,
    directory: Path,
    queue: Queue[dict[str, Any]],
    stop_stream: Event,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.1, min(2.0, deadline - time.time()))
        try:
            event = queue.get(timeout=remaining)
        except Empty:
            continue

        data = event.get("data")
        event_type = opencode_event_type(event)
        if event_type == "session.error" and session_matches(data, session_id):
            raise RuntimeError(f"OpenCode session error: {json.dumps(data, ensure_ascii=False)}")
        if event_type == "permission.asked" and session_matches(data, session_id):
            raise RuntimeError(f"OpenCode requested user input or approval: {json.dumps(data, ensure_ascii=False)}")
        if event_type == "session.idle" and session_matches(data, session_id):
            stop_stream.set()
            log_runner("session idle; fetching final message")
            messages = request_json("GET", f"{base_url}/session/{session_id}/message?limit=1", directory=directory, timeout=30)
            if not messages:
                raise RuntimeError("OpenCode session became idle but returned no messages")
            response = messages[0]
            error = response.get("info", {}).get("error")
            if error:
                raise RuntimeError(f"OpenCode task failed: {error}")
            log_runner("final OpenCode message fetched")
            return response

    stop_stream.set()
    raise RuntimeError("timed out")


def stream_session_events(base_url: str, directory: Path, queue: Queue[dict[str, Any]], stop_stream: Event) -> None:
    request = urllib.request.Request(f"{base_url}/event", headers=build_headers(directory))
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            for event in iter_sse_events(response):
                if stop_stream.is_set():
                    return
                event_name = event.get("event", "message")
                data = event.get("data")
                rendered = format_event_for_log(event_name, data)
                if rendered:
                    print(rendered, file=sys.stderr, flush=True)
                queue.put(event)
    except Exception as error:
        if not stop_stream.is_set():
            queue.put({"event": "session.error", "data": {"error": str(error)}})


def iter_sse_events(response) -> Iterable[dict[str, Any]]:
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        if not line:
            if data_lines:
                data_text = "\n".join(data_lines)
                yield {"event": event_name, "data": parse_event_data(data_text)}
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


def parse_event_data(data_text: str) -> Any:
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(data_text)
    return data_text


def opencode_event_type(event: dict[str, Any]) -> str:
    data = event.get("data")
    if isinstance(data, dict) and isinstance(data.get("type"), str):
        return data["type"]
    return str(event.get("event", "message"))


def session_matches(data: Any, session_id: str) -> bool:
    if isinstance(data, dict):
        for key in ("sessionID", "sessionId", "id"):
            if data.get(key) == session_id:
                return True
        for nested_key in ("session", "properties"):
            nested = data.get(nested_key)
            if isinstance(nested, dict) and session_matches(nested, session_id):
                return True
        info = data.get("info")
        if isinstance(info, dict):
            return session_matches(info, session_id)
    return False


def extract_assistant_text(data: Any) -> str | None:
    if not isinstance(data, dict) or data.get("type") != "message.part.updated":
        return None
    properties = data.get("properties")
    if not isinstance(properties, dict):
        return None
    part = properties.get("part")
    if not isinstance(part, dict) or part.get("type") != "text":
        return None
    text = part.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def truncate_for_log(value: str, *, limit: int = MAX_LOG_FIELD_LENGTH) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3]}..."


def redact_large_values(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("data:image/") or len(value) > MAX_LOG_FIELD_LENGTH:
            return truncate_for_log(value)
        return value
    if isinstance(value, list):
        return [redact_large_values(item) for item in value[:8]]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"url", "content"} and isinstance(item, str) and item.startswith("data:image/"):
                redacted[key] = "[image data suppressed]"
            else:
                redacted[key] = redact_large_values(item)
        return redacted
    return value


def compact_json_for_log(value: Any, *, limit: int = MAX_LOG_JSON_LENGTH) -> str:
    text = json.dumps(redact_large_values(value), ensure_ascii=False, sort_keys=True)
    return truncate_for_log(text, limit=limit)


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def summarize_tool_part(part: dict[str, Any]) -> str:
    tool_name = first_present(part, ("tool", "name", "callID", "id")) or "unknown"
    state = first_present(part, ("state", "status", "phase"))
    title = first_present(part, ("title", "description"))
    command = first_present(part, ("command", "cmd"))
    path = first_present(part, ("path", "file"))
    error = first_present(part, ("error", "stderr"))
    output = first_present(part, ("output", "stdout", "result"))
    input_payload = first_present(part, ("input", "args", "parameters"))

    pieces = [f"tool={tool_name}"]
    if isinstance(state, str):
        pieces.append(f"state={state}")
    if isinstance(title, str):
        pieces.append(f"title={truncate_for_log(title)}")
    if isinstance(path, str):
        pieces.append(f"path={path}")
    if isinstance(command, str):
        pieces.append(f"command={truncate_for_log(command)}")
    if error:
        pieces.append(f"error={compact_json_for_log(error) if not isinstance(error, str) else truncate_for_log(error)}")
    if output:
        pieces.append(f"output={compact_json_for_log(output) if not isinstance(output, str) else truncate_for_log(output)}")
    if input_payload and not command and not path:
        pieces.append(f"input={compact_json_for_log(input_payload)}")
    return "tool part " + " ".join(pieces)


def summarize_message_part(part: dict[str, Any]) -> str:
    part_type = part.get("type", "unknown")
    if part_type == "file":
        mime = part.get("mime", "unknown")
        return f"file part ({mime}) suppressed"
    if part_type == "tool":
        return summarize_tool_part(part)
    if part_type in {"reasoning", "step-start", "step-finish"}:
        return f"{part_type} part {compact_json_for_log(part)}"
    return f"{part_type} part {compact_json_for_log(part)}"


def summarize_event_payload(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)

    event_type = data.get("type")
    properties = data.get("properties")
    if event_type and not isinstance(properties, dict):
        return str(event_type)

    if event_type in {"server.heartbeat", "session.status", "session.diff", "session.updated"}:
        return None

    if event_type == "message.updated":
        info = properties.get("info")
        if isinstance(info, dict):
            role = info.get("role", "unknown")
            agent = info.get("agent")
            if isinstance(agent, str) and agent:
                return f"role={role} agent={agent}"
            return f"role={role}"

    if event_type == "message.part.updated":
        part = properties.get("part")
        if isinstance(part, dict):
            return summarize_message_part(part)

    if event_type == "message.part.delta":
        return None

    if event_type in {"tool.execute.before", "tool.execute.after"}:
        tool = properties.get("tool")
        title = properties.get("title")
        status = properties.get("status")
        pieces = [str(event_type)]
        if isinstance(tool, str) and tool:
            pieces.append(tool)
        if isinstance(title, str) and title:
            pieces.append(title)
        if isinstance(status, str) and status:
            pieces.append(f"status={status}")
        return " ".join(pieces)

    if event_type in {"session.idle", "session.error", "permission.asked", "server.connected"}:
        return str(event_type)

    return json.dumps(data, ensure_ascii=False)


def format_event_for_log(event_name: str, data: Any) -> str | None:
    assistant_text = extract_assistant_text(data)
    if assistant_text:
        return f"[assistant] {assistant_text}"

    body = summarize_event_payload(data)
    if not body:
        return None
    return f"[OpenCode event] {event_name}: {body}"


def safe_request(method: str, url: str, *, directory: Path, timeout: int) -> None:
    try:
        request_json(method, url, directory=directory, timeout=timeout)
    except Exception as error:
        print(f"warning: cleanup request failed: {error}", file=sys.stderr)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one task to OpenCode, then clean up local dev servers.")
    parser.add_argument("task", help="Task prompt to send to OpenCode")
    parser.add_argument("--image-url", required=True, help="Direct image URL to download and attach to the OpenCode task")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    base_url = os.environ.get("OPENCODE_SERVER_URL", DEFAULT_OPENCODE_URL).rstrip("/")
    timeout = int(os.environ.get("OPENCODE_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    directory = Path(os.environ.get("OPENCODE_APP_DIR", str(APP_DIR))).expanduser().resolve()

    try:
        screenshot_path = download_reference_image(args.image_url, DEFAULT_SCREENSHOT_PATH)

        response = run_task(args.task, screenshot_path=screenshot_path, base_url=base_url, directory=directory, timeout=timeout)

        log_runner("starting generated app dev server")
        dev_server = start_dev_server(directory)
        try:
            log_runner(f"waiting for generated app at {LOCAL_APP_URL}")
            wait_for_url(LOCAL_APP_URL)
            log_runner("capturing generated screenshot")
            generated_screenshot_path = capture_screenshot(LOCAL_APP_URL, AI_GENERATED_SCREENSHOT_PATH)
            log_runner(f"generated screenshot saved: {generated_screenshot_path}")
        finally:
            log_runner("stopping generated app dev server")
            stop_dev_server(dev_server)
        response["ai_generated_screenshot"] = str(generated_screenshot_path)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
