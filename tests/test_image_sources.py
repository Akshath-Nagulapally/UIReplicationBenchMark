from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import image_sources


class ImageSourcesTests(unittest.TestCase):
    def test_parse_huggingface_dataset_root_url(self):
        result = image_sources.parse_huggingface_dataset_url(
            "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark"
        )

        self.assertEqual(
            result,
            image_sources.HuggingFaceDatasetRef(
                repo_id="Akshath-Nag/UIReplicationBenchMark",
                revision="main",
                path="",
            ),
        )

    def test_parse_huggingface_dataset_tree_url_with_subfolder(self):
        result = image_sources.parse_huggingface_dataset_url(
            "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/tree/main/examples/mobile"
        )

        self.assertEqual(
            result,
            image_sources.HuggingFaceDatasetRef(
                repo_id="Akshath-Nag/UIReplicationBenchMark",
                revision="main",
                path="examples/mobile",
            ),
        )

    def test_parse_huggingface_direct_file_url_is_not_treated_as_dataset_folder(self):
        self.assertIsNone(
            image_sources.parse_huggingface_dataset_url(
                "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/resolve/main/Anthropic.png"
            )
        )

    def test_expand_image_source_urls_lists_supported_images_from_huggingface_dataset(self):
        dataset_payload = {
            "siblings": [
                {"rfilename": "Anthropic.png"},
                {"rfilename": "nested/OpenAI.png"},
                {"rfilename": "README.md"},
            ]
        }

        with patch.object(
            image_sources.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(dataset_payload).encode("utf-8")),
        ):
            result = image_sources.expand_image_source_urls(
                ["https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark"]
            )

        self.assertEqual(
            result,
            [
                "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/resolve/main/Anthropic.png",
                "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/resolve/main/nested/OpenAI.png",
            ],
        )

    def test_expand_image_source_urls_filters_to_requested_huggingface_subfolder(self):
        dataset_payload = {
            "siblings": [
                {"rfilename": "mobile/Anthropic.png"},
                {"rfilename": "desktop/OpenAI.png"},
            ]
        }

        with patch.object(
            image_sources.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(dataset_payload).encode("utf-8")),
        ):
            result = image_sources.expand_image_source_urls(
                ["https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/tree/main/mobile"]
            )

        self.assertEqual(
            result,
            [
                "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark/resolve/main/mobile/Anthropic.png"
            ],
        )

    def test_resolve_single_image_url_rejects_multi_image_dataset_sources(self):
        dataset_payload = {
            "siblings": [
                {"rfilename": "Anthropic.png"},
                {"rfilename": "OpenAI.png"},
            ]
        }

        with patch.object(
            image_sources.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(dataset_payload).encode("utf-8")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Expected a single image"):
                image_sources.resolve_single_image_url(
                    "https://huggingface.co/datasets/Akshath-Nag/UIReplicationBenchMark"
                )
