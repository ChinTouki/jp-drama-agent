import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from dotenv import load_dotenv

# 本地调试时加载 .env（Render 上使用环境变量，不会受影响）
load_dotenv()

app = FastAPI()

# ---- 健康检查 ----
@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}


# ---- 请求 / 响应模型 ----

class ChatRequest(BaseModel):
    user_id: str
    mode: Literal["tutor"] = "tutor"  # 先只做 tutor 模式，后面再扩展
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ---- LLM 配置 ----

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")  # 例如 https://api.openai.com/v1
LLM_MODEL = os.getenv("LLM_MODEL")        # 例如 gpt-4.1-mini


async def call_llm(system_prompt: str, user_message: str) -> str:
    """
    最小通用 LLM 调用。
    按你实际使用的服务调整 base_url、路径和返回解析。
    """
    if not (LLM_API_KEY and LLM_API_BASE and LLM_MODEL):
        raise HTTPException(status_code=500, detail="LLM configuration missing")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    async with httpx.AsyncClient(base_url=LLM_API_BASE, timeout=30.0) as client:
        # 如用 /v1/chat/completions，则这里写 "/chat/completions"
        resp = await client.post("/chat/completions", json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected LLM response format")


def build_system_prompt(mode: str) -> str:
    """
    简易角色设定。现在只实现 tutor。
    """
    if mode == "tutor":
        return (
            "你是专业日语老师，用简洁友好的方式帮助中文用户学习日语。"
            "优先使用自然简洁的日语回答，关键点用少量中文解释。"
            "每次回复控制在几句话，不要太长，给用户具体可用的表达。"
        )
    return "你是一个友好的日语学习助手。"


# ---- 对话接口 ----

@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.mode)
    reply = await call_llm(system_prompt, req.message)
    return ChatResponse(reply=reply)
