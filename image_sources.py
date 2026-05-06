from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import quote, urlparse


HUGGING_FACE_HOST = "huggingface.co"
HUGGING_FACE_DATASET_PREFIX = "/datasets/"
HUGGING_FACE_DEFAULT_REVISION = "main"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@dataclass(frozen=True)
class HuggingFaceDatasetRef:
    repo_id: str
    revision: str = HUGGING_FACE_DEFAULT_REVISION
    path: str = ""


def expand_image_source_urls(urls: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for url in urls:
        dataset_ref = parse_huggingface_dataset_url(url)
        if dataset_ref is None:
            expanded.append(url)
            continue
        expanded.extend(list_huggingface_dataset_image_urls(dataset_ref))
    return expanded


def resolve_single_image_url(url: str) -> str:
    dataset_ref = parse_huggingface_dataset_url(url)
    if dataset_ref is None:
        return url

    image_urls = list_huggingface_dataset_image_urls(dataset_ref)
    if not image_urls:
        raise RuntimeError(f"No image files were found in Hugging Face dataset source: {url}")
    if len(image_urls) > 1:
        raise RuntimeError(
            "Expected a single image from the Hugging Face dataset source, "
            f"but found {len(image_urls)} image files: {url}"
        )
    return image_urls[0]


def parse_huggingface_dataset_url(url: str) -> HuggingFaceDatasetRef | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != HUGGING_FACE_HOST:
        return None
    if not parsed.path.startswith(HUGGING_FACE_DATASET_PREFIX):
        return None

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 3 or segments[0] != "datasets":
        return None

    repo_id = f"{segments[1]}/{segments[2]}"
    if len(segments) == 3:
        return HuggingFaceDatasetRef(repo_id=repo_id)

    action = segments[3]
    if action == "resolve":
        return None

    revision = HUGGING_FACE_DEFAULT_REVISION
    dataset_path = ""
    if action == "tree":
        if len(segments) >= 5:
            revision = segments[4]
        if len(segments) >= 6:
            dataset_path = "/".join(segments[5:])
    else:
        dataset_path = "/".join(segments[3:])

    return HuggingFaceDatasetRef(repo_id=repo_id, revision=revision, path=dataset_path)


def list_huggingface_dataset_image_urls(dataset_ref: HuggingFaceDatasetRef) -> list[str]:
    api_url = build_huggingface_dataset_api_url(dataset_ref)
    request = urllib.request.Request(api_url, headers={"User-Agent": "ui-replicate/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to list Hugging Face dataset files with HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to list Hugging Face dataset files: {error.reason}") from error

    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError(f"Hugging Face dataset response did not include a file listing for {dataset_ref.repo_id}")

    prefix = dataset_ref.path.strip("/")
    image_urls: list[str] = []
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        filename = sibling.get("rfilename")
        if not isinstance(filename, str) or not _is_supported_image_file(filename):
            continue
        normalized_filename = filename.strip("/")
        if prefix and not _matches_dataset_prefix(normalized_filename, prefix):
            continue
        image_urls.append(build_huggingface_resolve_url(dataset_ref.repo_id, dataset_ref.revision, normalized_filename))

    if prefix:
        image_urls.sort(key=lambda item: PurePosixPath(urlparse(item).path).name.lower())
    else:
        image_urls.sort(key=str.lower)
    return image_urls


def build_huggingface_dataset_api_url(dataset_ref: HuggingFaceDatasetRef) -> str:
    repo_id = quote(dataset_ref.repo_id, safe="/")
    revision = quote(dataset_ref.revision, safe="")
    return f"https://{HUGGING_FACE_HOST}/api/datasets/{repo_id}/revision/{revision}?full=true"


def build_huggingface_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    repo_part = quote(repo_id, safe="/")
    revision_part = quote(revision, safe="")
    filename_part = quote(filename, safe="/")
    return f"https://{HUGGING_FACE_HOST}/datasets/{repo_part}/resolve/{revision_part}/{filename_part}"


def _is_supported_image_file(filename: str) -> bool:
    return PurePosixPath(filename).suffix.lower() in IMAGE_SUFFIXES


def _matches_dataset_prefix(filename: str, prefix: str) -> bool:
    normalized_prefix = prefix.strip("/")
    return filename == normalized_prefix or filename.startswith(f"{normalized_prefix}/")
