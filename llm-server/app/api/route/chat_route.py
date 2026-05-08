import uuid
from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide

from app.container import Container
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.services.text_generation_service import TextGenerationService
from app.api.dto.chat_completion_request import ChatCompletionRequest
from app.api.dto.chat_completion_response import ChatCompletionResponse, ChatChoice
from app.api.mapper.chat_mapper import ChatMapper

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@inject
async def chat_completions(
    request: ChatCompletionRequest,
    service: TextGenerationService = Depends(Provide[Container.text_generation_service])
) -> ChatCompletionResponse:
    try:
        command = ChatMapper.to_generate_text_command(request)
        response_message = await service.generate_text(command)
        
        if request.response_format and request.response_format.type == "json_schema":
            schema = request.response_format.json_schema.get("schema")
            if schema and response_message.content:
                import json
                import jsonschema
                try:
                    parsed_content = json.loads(response_message.content)
                    jsonschema.validate(instance=parsed_content, schema=schema)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Model failed to generate valid JSON")
                except jsonschema.exceptions.ValidationError as ve:
                    raise HTTPException(status_code=400, detail=f"Model failed to generate desired format: {ve.message}")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
            ]
        )
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
