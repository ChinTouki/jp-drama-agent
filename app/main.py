import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# 本地加? .env；在?上 Render 用?境?量
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# 使用官方 OpenAI SDK 客?端
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}


# ??：?看后端???到的 key / base / model（不泄露完整 key）
@app.get("/debug/key-info")
def debug_key_info():
    if not LLM_API_KEY:
        return {"has_key": False}
    return {
        "has_key": True,
        "length": len(LLM_API_KEY),
        "prefix": LLM_API_KEY[:7],   # 比如 sk-proj
        "suffix": LLM_API_KEY[-4:],  # 最后4位，?自己?一下
        "base": LLM_API_BASE,
        "model": LLM_MODEL,
    }


class ChatRequest(BaseModel):
    user_id: str
    mode: Literal["tutor"] = "tutor"
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt(mode: str) -> str:
    if mode == "tutor":
        return (
            "?是??日?老?，用??自然的日?回答中文用?的??，"
            "??地方用少量中文解?，?次?出?明有用的表?。"
        )
    return "?是一个友好的日?学?助手。"


async def call_llm(system_prompt: str, user_message: str) -> str:
    if not (LLM_API_KEY and LLM_MODEL):
        raise HTTPException(status_code=500, detail="LLM configuration missing")

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 直接把??信息抛出去，方便?在 /docs 里看到真?原因
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.mode)
    reply = await call_llm(system_prompt, req.message)
    return ChatResponse(reply=reply)
