import os
from typing import Literal, Optional

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


@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}


# 简单试玩页面：用户可选模式 + 输入内容，直连 /agent/chat
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
      <p class="desc">选择模式，输入内容，直接体验你的「日剧老师」或「宅系女友」Agent。这个页面前端很轻量，适合录屏、小红书展示。</p>

      <label>模式 Mode</label>
      <select id="mode">
        <option value="tutor">日剧日语老师（解释表达、语法、口语）</option>
        <option value="otaku_waifu">宅系女友聊天（治愈陪聊＋顺手教日语）</option>
      </select>

      <label>你的输入</label>
      <textarea id="input" rows="3"
        placeholder="例如：教我一个在便利店常用的自然问句；或：今天有点累，用可爱的日语夸夸我。"></textarea>

      <button id="send">发送</button>

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
                user_id: "web-demo",
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
            send();
          }
        });
      </script>
    </body>
    </html>
    """


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
            "使用自然简洁的日语回答，重要之处用少量中文解释。"
            "每次给出1-3个实用表达，结合日本生活和日剧场景讲解，避免废话。"
        )
    if mode == "otaku_waifu":
        return (
            "你是可爱、懂ACG文化的宅系女友角色。"
            "主要用口语日语聊天，语气黏人但健康、尊重边界，"
            "可以顺手教简单日语表达，但不输出露骨或违反平台政策的内容。"
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
