import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def openai_client():
    """Fixture to provide an OpenAI client pointing to the local running server."""
    port = os.getenv("PORT", "8001")
    return OpenAI(
        api_key="required-but-not-used",
        base_url=f"http://localhost:{port}/v1"
    )

def test_chat_completions(openai_client):
    """Test standard chat completions via official OpenAI SDK."""
    model_name = "models/chat/mlx-community/gemma-4-26b-a4b-it-4bit"
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_tokens=100,
        temperature=0.0
    )
    
    assert response.id.startswith("chatcmpl-")
    assert response.object == "chat.completion"
    assert response.model == model_name
    assert len(response.choices) > 0
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content is not None

def test_chat_completions_with_tools(openai_client):
    """Test chat completions with tool calls via official OpenAI SDK."""
    model_name = "models/chat/mlx-community/gemma-4-26b-a4b-it-4bit"
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is the weather like in Paris?"}
        ],
        max_tokens=500,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                }
            }
        ]
    )
    
    assert response.choices[0].message.role == "assistant"
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
    elif response.choices[0].message.content:
        assert isinstance(response.choices[0].message.content, str)

def test_chat_completions_with_response_format(openai_client):
    """Test chat completions with response_format to enforce json_schema output."""
    model_name = "models/chat/mlx-community/gemma-4-26b-a4b-it-4bit"
    schema = {
        "title": "ProfessionalProfile",
        "type": "object",
        "properties": {
            "identity": {
                "type": "object",
                "properties": {
                    "appellation": {"type": "string"},
                    "biometrics": {
                        "type": "object",
                        "properties": {
                            "years_since_birth": {"type": "integer"},
                            "gender_identity": {"type": "string", "enum": ["male", "female", "non-binary", "prefer not to say"]}
                        },
                        "required": ["years_since_birth", "gender_identity"]
                    }
                },
                "required": ["appellation", "biometrics"]
            },
            "geographic_footprint": {
                "type": "object",
                "properties": {
                    "urban_center": {"type": "string"},
                    "precise_location": {
                        "type": "object",
                        "properties": {
                            "thoroughfare": {"type": "string"},
                            "postal_identifier": {"type": "string"}
                        },
                        "required": ["thoroughfare"]
                    }
                },
                "required": ["urban_center", "precise_location"]
            },
            "career_assets": {
                "type": "object",
                "properties": {
                    "academic_status": {
                        "type": "object",
                        "properties": {
                            "is_active_student": {"type": "boolean"},
                            "highest_degree": {"type": "string"}
                        },
                        "required": ["is_active_student"]
                    },
                    "technical_stack": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {"type": "string"},
                                "proficiency_estimation": {"type": "string", "enum": ["novice", "competent", "master"]}
                            },
                            "required": ["domain", "proficiency_estimation"]
                        }
                    }
                },
                "required": ["academic_status", "technical_stack"]
            }
        },
        "required": ["identity", "geographic_footprint", "career_assets"]
    }
    
    ambiguous_prompt = (
        "The individual known as Alice, who saw the world for the first time exactly thirty years ago, "
        "identifies as a woman. She has established her primary residence in the City of Lights. "
        "Her home is situated on the famous thoroughfare Rue de Rivoli, at number 123, within the district 75001. "
        "When it comes to her professional repertoire, she is highly proficient in the language of snakes, "
        "possesses competent skills in the language that emphasizes memory safety, and is a novice in the "
        "statically typed language born at Google. Her days of formal schooling are a distant memory, "
        "and she holds a Master's degree in Computer Science."
    )

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a sophisticated data extraction engine. Parse the narrative into the requested high-fidelity JSON structure."},
            {"role": "user", "content": ambiguous_prompt}
        ],
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "professional_profile_extraction",
                "schema": schema,
                "strict": True
            }
        }
    )
    
    assert response.choices[0].message.role == "assistant"
    content = response.choices[0].message.content
    assert content is not None
    
    import json
    import jsonschema
    
    parsed_content = json.loads(content)
    print(f"\nParsed content: {json.dumps(parsed_content, indent=2)}")
    jsonschema.validate(instance=parsed_content, schema=schema)
