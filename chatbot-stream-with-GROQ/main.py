from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot
import uvicorn

app = FastAPI()

# CORS middleware to fix "Failed to fetch" error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000"] if using strict frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_input = body.get("message", "")

    async def event_generator():
        try:
            async for chunk, _ in chatbot.astream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable': {'thread_id': 'thread-1'}},
                stream_mode='messages'
            ):
                yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
