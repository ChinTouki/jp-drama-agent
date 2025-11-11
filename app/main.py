import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# 本地加载 .env；Render 上使用环境变量
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# OpenAI 官方 SDK 客户端
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}


class ChatRequest(BaseModel):
    user_id: str
    mode: Literal["tutor", "otaku_waifu"] = "tutor"
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt(mode: str) -> str:
    if mode == "tutor":
        return (
            "你是专业日语老师，用户是中文母语者。"
            "用自然简洁的日语回答，必要时用少量中文解释。"
            "每次给1-3个实用表达，结合日本日常和日剧场景教学。"
        )
    if mode == "otaku_waifu":
        return (
            "你是可爱、懂ACG文化的宅系女友角色。"
            "主要用口语日语聊天，语气轻松黏人但健康边界明确，"
            "可以顺手教简单日语表达，不输出违规或露骨内容。"
        )
    return "你是一个友好的日语学习助手。"


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
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.mode)
    reply = await call_llm(system_prompt, req.message)
    return ChatResponse(reply=reply)
