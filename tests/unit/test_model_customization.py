from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest

from app.agent.middleware import ModelCustomizationMiddleware


class FakeModel:
    def __init__(self, name: str = "fake_model", bound_kwargs: dict | None = None):
        self.name = name
        self.bound_kwargs = bound_kwargs or {}

    def bind(self, **kwargs):
        new_kwargs = {**self.bound_kwargs, **kwargs}
        return FakeModel(self.name, new_kwargs)


@pytest.mark.asyncio
async def test_model_customization_middleware_sync_and_async():
    middleware = ModelCustomizationMiddleware()
    fake_model = FakeModel()

    # Case 1: No customization in state
    req_empty = ModelRequest(
        model=fake_model,
        messages=[],
        system_message=None,
        tools=[],
        tool_choice=None,
        response_format=None,
        state={},
    )
    handler_mock = MagicMock(return_value="response_empty")
    res = middleware.wrap_model_call(req_empty, handler_mock)
    assert res == "response_empty"
    assert handler_mock.call_args[0][0].model.bound_kwargs == {}

    # Case 2: Customization provided
    state_custom = {
        "model_customization": {
            "temperature": 0.2,
            "top_p": 0.85,
            "top_k": 30,
            "max_output_tokens": 1024,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "enable_thinking": True,
        }
    }
    req_custom = ModelRequest(
        model=fake_model,
        messages=[],
        system_message=None,
        tools=[],
        tool_choice=None,
        response_format=None,
        state=state_custom,
    )

    handler_sync = MagicMock(return_value="sync_ok")
    res_sync = middleware.wrap_model_call(req_custom, handler_sync)
    assert res_sync == "sync_ok"
    called_model = handler_sync.call_args[0][0].model
    assert called_model.bound_kwargs["temperature"] == 0.2
    assert called_model.bound_kwargs["top_p"] == 0.85
    assert called_model.bound_kwargs["max_tokens"] == 1024
    assert called_model.bound_kwargs["presence_penalty"] == 0.1
    assert called_model.bound_kwargs["frequency_penalty"] == 0.05
    assert called_model.bound_kwargs["extra_body"] == {
        "top_k": 30,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    # Case 3: Async wrap
    async def async_handler(req):
        return ("async_ok", req.model.bound_kwargs)

    res_status, bound_data = await middleware.awrap_model_call(req_custom, async_handler)
    assert res_status == "async_ok"
    assert bound_data["temperature"] == 0.2
    assert bound_data["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
