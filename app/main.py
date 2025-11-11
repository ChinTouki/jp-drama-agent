import os
from typing import Literal, Optional
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# 本地加载 .env；Render 上使用环境变量
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
)

app = FastAPI()

# ===== 简单每日免费额度（MVP 用） =====
FREE_LIMIT_PER_DAY = 5  # 每个 user_id 每天免费 30 条
_usage: dict[str, dict] = {}  # {user_id: {"count": int, "reset": datetime}}


def check_quota(user_id: str):
    """按 user_id 做每日限额（内存版）。"""
    now = datetime.now(timezone.utc)
    info = _usage.get(user_id)

    # 第一次使用或新的一天，重置
    if not info or now >= info["reset"]:
        _usage[user_id] = {"count": 1, "reset": now + timedelta(days=1)}
        return

    # 超过限额
    if info["count"] >= FREE_LIMIT_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=(
                "今日免费体验次数已用完。\n"
                "如果你喜欢这个日语 / 宅系聊天 Agent，可以联系作者开通会员或订阅版。"
            ),
        )

    # 在限额内，+1
    info["count"] += 1


# ===== 基础路由 =====

@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}


# 简单试玩页面
@app.get("/playground", response_class=HTMLResponse)
def playground():
    return """
    <!DOCTYPE html>
    <html lang="zh-cn">
    <head>
      <meta charset="UTF-8" />
      <title>JP Drama / 宅系聊天 Agent Demo</title>
      <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
               max-width: 720px; margin: 40px auto; padding: 0 16px; }
        h1 { font-size: 22px; margin-bottom: 4px; }
        p.desc { font-size: 13px; color: #666; margin-top: 0; }
        label { display: block; margin-top: 16px; font-weight: 600; font-size: 14px; }
        select, textarea, button {
          width: 100%; margin-top: 6px; padding: 8px;
          font-size: 14px; border-radius: 6px; border: 1px solid #ddd;
          box-sizing: border-box;
        }
        button {
          margin-top: 14px; background: #111827; color: #fff; border: none;
          cursor: pointer; font-weight: 600;
        }
        button:disabled { opacity: .6; cursor: default; }
        .msg {
          margin-top: 10px; padding: 10px; min-height: 60px;
          border-radius: 6px; border: 1px solid #eee;
          white-space: pre-wrap; font-size: 14px; background: #fafafa;
        }
      </style>
    </head>
    <body>
      <h1>JP Drama / 宅系聊天 Agent</h1>
      <p class="desc">
        选择模式，输入内容，体验「日剧老师」或「宅系女友」风格的 AI。
        支持每天一定次数的免费体验，适合展示和拉新。
      </p>

      <label>模式 Mode</label>
      <select id="mode">
        <option value="tutor">日剧日语老师（解释表达、语法、口语）</option>
        <option value="otaku_waifu">宅系女友聊天（治愈陪聊＋顺手教日语）</option>
      </select>

      <label>你的输入</label>
      <textarea id="input" rows="3"
        placeholder="例如：教我一个在便利店常用的自然问句；或：今天有点累，用可爱的日语夸夸我。"></textarea>

      <button id="send">发送（Ctrl/⌘ + Enter 快速发送）</button>

      <label>AI 回复</label>
      <div id="reply" class="msg"></div>

      <script>
        const endpoint = "/agent/chat";
        const sendBtn = document.getElementById("send");
        const inputEl = document.getElementById("input");
        const modeEl = document.getElementById("mode");
        const replyEl = document.getElementById("reply");

        async function send() {
          const text = inputEl.value.trim();
          if (!text) return;
          const mode = modeEl.value;
          replyEl.textContent = "思考中 / 考え中...";
          sendBtn.disabled = true;
          try {
            const res = await fetch(endpoint, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                user_id: "web-demo",  // 正式版可换成真实用户ID
                mode,
                message: text
              })
            });
            const data = await res.json();
            replyEl.textContent = data.reply || JSON.stringify(data, null, 2);
          } catch (e) {
            replyEl.textContent = "出错了：" + e;
          } finally {
            sendBtn.disabled = false;
          }
        }

        sendBtn.addEventListener("click", send);
        inputEl.addEventListener("keydown", (e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            send();
          }
        });
      </script>
    </body>
    </html>
    """


# ===== 请求/响应模型 =====

class ChatRequest(BaseModel):
    user_id: str
    mode: Literal["tutor", "otaku_waifu"] = "tutor"
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ===== 角色设定 =====

def build_system_prompt(mode: str) -> str:
    if mode == "tutor":
        return (
            "你是专业日语老师，用户是中文母语者。"
            "使用自然简洁的日语回答，重要处用少量中文解释。"
            "每次给出1-3个实用表达，结合日本生活与日剧场景，避免废话。"
        )
    if mode == "otaku_waifu":
        return (
            "你是可爱、懂ACG文化的宅系女友角色。"
            "主要用口语日语聊天，语气黏人但健康，有边界感，"
            "可以顺手教简单日语表达，不输出露骨或违反平台政策的内容。"
        )
    return "你是一个友好的日语学习助手。"


# ===== 调用 LLM =====

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


# ===== 主对话接口 =====

@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    # 免费额度检查
    check_quota(req.user_id)

    system_prompt = build_system_prompt(req.mode)
    reply = await call_llm(system_prompt, req.message)
    return ChatResponse(reply=reply)
