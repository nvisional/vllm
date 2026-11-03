# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    "developer_message",
    [
        {"role": "developer", "content": ""},
        {"role": "developer"},
        {"role": "developer", "content": []},
    ],
)
def test_empty_or_missing_developer_content_rejected(developer_message):
    """Empty/missing developer content must fail at request parse (HTTP 400).

    Regression for #65357: these payloads previously reached DeepSeek chat
    template encoding and returned HTTP 500 with leaked ``_image_indices``.
    """
    with pytest.raises(VLLMValidationError, match="Developer message"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [developer_message, {"role": "user", "content": "hi"}],
                "model": "facebook/opt-125m",
            }
        )


def test_null_developer_content_rejected():
    # ``content: null`` used to fail via pydantic string_type; the developer
    # content validator now rejects it as VLLMValidationError for consistency.
    with pytest.raises((VLLMValidationError, ValidationError)):
        ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "developer", "content": None},
                    {"role": "user", "content": "hi"},
                ],
                "model": "facebook/opt-125m",
            }
        )


def test_non_empty_developer_content_accepted():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "developer", "content": "Follow the policy."},
                {"role": "user", "content": "hi"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.messages[0]["content"] == "Follow the policy."


def test_empty_system_content_still_accepted():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hi"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.messages[0]["content"] == ""
