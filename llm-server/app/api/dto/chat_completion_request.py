from typing import Literal, Union, Annotated, Any
from pydantic import BaseModel, Field

class _ChatCompletionContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class _ChatCompletionContentPartImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"

class _ChatCompletionContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: _ChatCompletionContentPartImageURL

class _ChatCompletionContentPartInputAudioAudio(BaseModel):
    data: str
    format: Literal["wav", "mp3"]

class _ChatCompletionContentPartInputAudio(BaseModel):
    type: Literal["input_audio"]
    input_audio: _ChatCompletionContentPartInputAudioAudio

class _ChatCompletionDeveloperMessage(BaseModel):
    role: Literal["developer"]
    content: str | list[_ChatCompletionContentPartText]
    name: str | None = None

class _ChatCompletionSystemMessage(BaseModel):
    role: Literal["system"]
    content: str | list[_ChatCompletionContentPartText]
    name: str | None = None

class _ChatCompletionUserMessage(BaseModel):
    role: Literal["user"]
    content: str | list[Union[_ChatCompletionContentPartText, _ChatCompletionContentPartImage, _ChatCompletionContentPartInputAudio]]
    name: str | None = None

class _ChatCompletionMessageFunctionCall(BaseModel):
    arguments: str
    name: str

class _ChatCompletionMessageToolCallFunction(BaseModel):
    arguments: str
    name: str

class _ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: _ChatCompletionMessageToolCallFunction

class _ChatCompletionContentPartRefusal(BaseModel):
    type: Literal["refusal"]
    refusal: str

class _ChatCompletionAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str | list[Union[_ChatCompletionContentPartText, _ChatCompletionContentPartRefusal]] | None = None
    name: str | None = None
    refusal: str | None = None
    tool_calls: list[_ChatCompletionMessageToolCall] | None = None
    function_call: _ChatCompletionMessageFunctionCall | None = None

class _ChatCompletionToolMessage(BaseModel):
    role: Literal["tool"]
    content: str | list[_ChatCompletionContentPartText]
    tool_call_id: str

class _ChatCompletionFunctionMessage(BaseModel):
    role: Literal["function"]
    content: str | None
    name: str

_ChatCompletionMessageParam = Annotated[
    Union[
        _ChatCompletionDeveloperMessage,
        _ChatCompletionSystemMessage,
        _ChatCompletionUserMessage,
        _ChatCompletionAssistantMessage,
        _ChatCompletionToolMessage,
        _ChatCompletionFunctionMessage,
    ],
    Field(discriminator="role"),
]

class _ResponseFormatText(BaseModel):
    type: Literal["text"]

class _ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"]

class _ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: dict[str, Any]

class _ChatCompletionToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

class _ChatCompletionTool(BaseModel):
    type: Literal["function"]
    function: _ChatCompletionToolFunction

class _ChatCompletionNamedToolChoiceFunction(BaseModel):
    name: str

class _ChatCompletionNamedToolChoice(BaseModel):
    type: Literal["function"]
    function: _ChatCompletionNamedToolChoiceFunction

class _ChatCompletionAudioParam(BaseModel):
    format: Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"]
    voice: str

class _ChatCompletionStreamOptions(BaseModel):
    include_obfuscation: bool | None = None
    include_usage: bool | None = None

class _ChatCompletionNamedToolChoiceCustomCustom(BaseModel):
    name: str

class _ChatCompletionNamedToolChoiceCustom(BaseModel):
    type: Literal["custom"]
    custom: _ChatCompletionNamedToolChoiceCustomCustom

class _ChatCompletionPredictionContent(BaseModel):
    type: Literal["content"]
    content: str | list[_ChatCompletionContentPartText]

class ChatCompletionRequest(BaseModel):
    """Request schema for OpenAI-compatible chat completions."""
    messages: list[_ChatCompletionMessageParam]
    model: str
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] | None = None
    response_format: Union[_ResponseFormatText, _ResponseFormatJSONObject, _ResponseFormatJSONSchema] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    tools: list[_ChatCompletionTool] | None = None
