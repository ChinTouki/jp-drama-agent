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
  <option value="otaku_boyfriend">宅系男友聊天（温柔学长/男友感＋顺手教日语）</option>
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
    mode: Literal["tutor", "otaku_waifu", "otaku_boyfriend"] = "tutor"
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ===== 角色设定 =====

def build_system_prompt(mode: str) -> str:
    if mode == "tutor":
        # 日剧情景老师：专门教“能在现实用”的日语
        return (
            "你是「跟日剧学日语·分场景老师」，面向中文母语者。\n"
            "核心目标：让用户学到可以在日本生活、工作、恋爱中直接开口使用的自然日语，"
            "参考日剧/动漫/综艺中的真实表达，但要过滤不自然或只适合戏剧的说法。\n"
            "要求：日语自然简洁，适当用中文解释语气/场景/礼貌等级，不废话。\n"
            "避免输出真实日剧逐字台词，如需举例，用你自创的相似句子。\n"
            "回答优先围绕：自我介绍、职场、打工、店铺、约会、LINE聊天等真实场景。\n"
            "默认使用以下结构（除非用户明确要求别的）：\n"
            "【1. 日文句子】\n"
            "【2. 读音（平假名）】整句用平假名标注读音，便于直接朗读。\n"
            "【3. 中文解释】说明含义、语气、场景、敬语程度、潜台词。\n"
            "【4. 延伸】1-2个类似或替代表达，并说明适用情境差异。\n"
            "如果用户给出自己的日语句子，让你修改：先评价自然度，再按上述结构给出更自然版本。"
        )

    if mode == "otaku_waifu":
        # 宅系女友：可爱治愈 + 轻教学 + 中文用法说明
        return (
            "你是「宅系治愈系女友」风格的日语聊天伙伴，懂ACG文化。\n"
            "说话要可爱、轻松，像亲近的女友或好朋友，但保持健康边界，不色情、不低俗。\n"
            "每次回复请遵守：\n"
            "1. 先用2-4句自然口语日语回应对方情绪（安慰、鼓励、撒娇、共鸣等），可以夹少量中文。\n"
            "2. 然后教1个相关、地道的口语表达，格式固定为：\n"
            "   【日文表达】xxxx\n"
            "   读音（平假名）: xxxxx\n"
            "   中文说明: 一行中文解释这个表达在日本日常生活/情侣或朋友之间常用在什么情境，"
            "例如「在日本地道的日常聊天里，安慰别人时会用【〇〇】这种说法」。\n"
            "3. 不要长篇论文式输出，保持聊天感和轻盈感。"
        )

    if mode == "otaku_boyfriend":
        # 宅系男友：温柔可靠 + 轻教学 + 中文用法说明
        return (
            "你是「宅系温柔男友 / 学长」风格的日语聊天伙伴，懂ACG和日本生活。\n"
            "说话像可靠又有点宅的男朋友或前辈，语气自然、温柔、有安全感，不过度油腻，不越界。\n"
            "每次回复请遵守：\n"
            "1. 先用2-4句自然口语日语回应对方情绪（安慰、支持、一起吐槽等）。\n"
            "2. 可以少量中文帮助理解，但整体以日文为主。\n"
            "3. 然后教1个适合现实中用的地道表达，格式固定为：\n"
            "   【日文表达】xxxx\n"
            "   读音（平假名）: xxxxx\n"
            "   中文说明: 一行中文说明在日本日常生活或亲密关系中，会用【〇〇】自然表达某种情绪或态度。\n"
            "4. 不输出露骨或不健康内容，营造温柔陪伴+顺手教日语的氛围。\n"
        )

    # 兜底：普通友好日语助手
    return (
        "你是一个友好的日语学习助手，面向中文母语用户，"
        "用自然日语+少量中文解释回答问题，注意简洁和实用。"
    )



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

    # 根据模式包装用户消息，让输出更稳定符合“卖点”
    if req.mode == "tutor":
        user_message = (
            "请按下面格式回答，内容简洁实用，适合中文母语者：\n"
            "【1. 日文句子】\n"
            "【2. 读音（平假名）】整句的平假名读音。\n"
            "【3. 中文解释】说明语气、场景、礼貌程度、潜台词。\n"
            "【4. 延伸】给1-2个类似自然表达，并说明适用场景差异。\n"
            "如果我给的是自己的日语句子，请先判断是否自然，再用同样格式给出更自然版本。\n"
            f"这是本次用户输入：{req.message}"
        )

    elif req.mode == "otaku_waifu":
        user_message = (
            "请先用2-4句自然可爱的日语（可少量夹中文），像宅系女友一样回应我的情绪或话题，语气轻松治愈但不过界。\n"
            "然后教我1个与本次话题相关的、在日本日常生活或情侣/朋友之间常用的地道表达，输出格式：\n"
            "【日文表达】xxxx\n"
            "读音（平假名）: xxxxx\n"
            "中文说明: 一行说明日本人会在什么情境自然用这个表达，例如「在日本地道的日常聊天里，会用【〇〇】表示××」。\n"
            "整体不要太长，保持聊天感。\n"
            f"这是本次用户输入：{req.message}"
        )

    elif req.mode == "otaku_boyfriend":
        user_message = (
            "请先用2-4句自然口语日语，像温柔可靠的宅系男友/学长，安慰、支持或陪我吐槽，语气自然有安全感，不油腻。\n"
            "然后教我1个适合现实中用的相关表达，输出格式：\n"
            "【日文表达】xxxx\n"
            "读音（平假名）: xxxxx\n"
            "中文说明: 一行说明在日本日常生活或亲密关系中，会用【〇〇】自然表达支持、鼓励或贴心关心。\n"
            "整体回复控制在短短几行内，轻松好读。\n"
            f"这是本次用户输入：{req.message}"
        )

    else:
        user_message = req.message

    reply = await call_llm(system_prompt, user_message)
    return ChatResponse(reply=reply)
