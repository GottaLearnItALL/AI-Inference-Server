from fastapi import APIRouter
from pydantic import BaseModel, Field
from server.engine import tokenizer, Request, request_queue
from uuid import uuid4
from time import time
from fastapi.responses import StreamingResponse
import json
api_router = APIRouter()

class Message(BaseModel):
    role: str = Field(description="The role of the author of this message")
    content: str = Field(description="The raw text")

class ChatCompletionsRequest(BaseModel):
    model: str = Field(description="The Model to use for this request")
    messages: list[Message] = Field(description='List of messages of the conversation')
    max_tokens: int | None = None
    stream: bool = Field(default=False,description="If set, message will be sent through SSE")
    temperature:float = 1.0

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str

class ChatCompletionsResponse(BaseModel):
    id: str = Field(description="Unique Request ID chatcmpl- ")
    object: str = "chat.completion"
    created: int = Field(description="Unix timestamp in seconds")
    model: str = Field(description="The model name that handled the request")
    usage: Usage
    choices: list[Choice]


@api_router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    message = [m.model_dump() for m in body.messages]
    max_tokens = body.max_tokens or 512
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True)
    
    request = Request(prompt,body.max_tokens)
    await request_queue.put(request)

    pieces = []
    while True:
        item  = await request.out_queue.get()
        if item['type'] == 'done':
            completion_token = item["generation_tokens"]
            finish_reason = item['finish_reason']
            break
        pieces.append(item["text"])
    text = "".join(pieces)

    if body.stream:

        async def event_generator():
            id='chatcmpl-' + uuid4().hex[:25]
            created=int(time())

            def chunk(delta, finish_reason):
                payload = {
                    "id": id,
                    "object":"chat.completion.chunk",
                    "created": created,
                    "model":body.model,
                    "choices": [{"index":0, "delta":delta, "finish_reason":finish_reason}]
                }
                return f"data: {json.dumps(payload)}\n\n"

            yield chunk(delta={"role": "assistant"}, finish_reason=None)

            messages = [m.model_dump() for m in body.messages]
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            request = Request(prompt_tokens=prompt_tokens, max_tokens=max_tokens)

        
            await request_queue.put(request)
            while True:
                item = await request.out_queue.get()
                if item['type'] == "done":
                    yield chunk(delta={}, finish_reason=item["finish_reason"])
                    yield f"data: [DONE]\n\n"
                    break
                yield chunk(delta={"content": item["text"]}, finish_reason=None)
        return StreamingResponse(event_generator(), media_type="text/event-stream")


    return ChatCompletionsResponse(
        id='chatcmpl-' + uuid4().hex[:25],
        created=int(time()),
        model=body.model,
        choices = [Choice(
            message = Message(role="assistant",content=text),
            finish_reason=finish_reason)],
        usage = Usage(
            prompt_tokens=len(prompt),
            completion_tokens = completion_token,
            total_tokens = len(prompt) + completion_token
        ),
    )

@api_router.get("/v1/models")
def get_model():
    created=int(time())
    data = [{"id":"qwen","object":"model","created":created,"owned_by":"local"}]
    return {"object":"list","data":data}

