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

# ===== 每日免费额度（MVP） =====
FREE_LIMIT_PER_DAY = 8  # 每个 user_id 每天免费 5 条
_usage: dict[str, dict] = {}  # {user_id: {"count": int, "reset": datetime}}


def check_quota(user_id: str):
    """按 user_id 做每日限额（内存版，单实例有效，够 MVP 用）"""
    now = datetime.now(timezone.utc)
    info = _usage.get(user_id)

    if not info or now >= info["reset"]:
        _usage[user_id] = {"count": 1, "reset": now + timedelta(days=1)}
        return

    if info["count"] >= FREE_LIMIT_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=(
                "今日免费体验次数已用完。\n"
                "如果你觉得ことの葉スタジオ对你有帮助，可以联系作者开通会员或长期版。"
            ),
        )

    info["count"] += 1


# ===== 基础路由 =====

@app.get("/")
def read_root():
    return {"message": "Kotonoha Studio is running."}


# ===== Playground 页面（美化版，多人格选择） =====

@app.get("/playground", response_class=HTMLResponse)
def playground():
    return """
    <!DOCTYPE html>
    <html lang="zh-cn">
    <head>
      <meta charset="UTF-8" />
      <title>ことの葉スタジオ｜日常日语 × 剧场对话 × 多人格陪练</title>
      <style>
        :root {
          --bg: #f5f5fa;
          --primary: #111827;
          --accent: #f97316;
          --accent-soft: #fee2e2;
          --border: #e5e7eb;
          --radius: 14px;
          --font: system-ui, -apple-system, BlinkMacSystemFont, -system-ui, sans-serif;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          padding: 24px 12px 40px;
          font-family: var(--font);
          background:
            radial-gradient(circle at top left, #e0f2fe 0, transparent 55%),
            radial-gradient(circle at top right, #fee2e2 0, transparent 55%),
            var(--bg);
          color: var(--primary);
        }
        .shell {
          max-width: 880px;
          margin: 0 auto;
        }
        .card {
          background: rgba(255, 255, 255, 0.98);
          border-radius: 24px;
          padding: 24px 20px 20px;
          box-shadow: 0 18px 40px rgba(15,23,42,0.06);
          border: 1px solid rgba(148,163,253,0.18);
          backdrop-filter: blur(10px);
        }
        h1 {
          font-size: 24px;
          margin: 0 0 6px;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 8px;
        }
        h1 span.logo {
          display: inline-flex;
          padding: 4px 9px;
          border-radius: 999px;
          background: var(--accent-soft);
          font-size: 11px;
          color: #9f1239;
        }
        .subtitle {
          font-size: 13px;
          color: #6b7280;
          margin: 0 0 14px;
        }
        .tags {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 14px;
          font-size: 11px;
        }
        .tag {
          padding: 3px 8px;
          border-radius: 999px;
          border: 1px solid var(--border);
          color: #6b7280;
        }
        label {
          display: block;
          margin-top: 14px;
          margin-bottom: 4px;
          font-weight: 600;
          font-size: 13px;
          color: #374151;
        }
        select, textarea, button {
          width: 100%;
          font-size: 13px;
          padding: 9px 10px;
          border-radius: var(--radius);
          border: 1px solid var(--border);
          outline: none;
          transition: all .18s ease;
          background: #ffffff;
        }
        select:focus, textarea:focus {
          border-color: #818cf8;
          box-shadow: 0 0 0 2px rgba(129,140,248,0.15);
        }
        textarea {
          resize: vertical;
          min-height: 72px;
        }
        button {
          margin-top: 10px;
          background: linear-gradient(to right, #111827, #1f2937);
          color: #fff;
          border: none;
          font-weight: 600;
          cursor: pointer;
        }
        button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 10px 18px rgba(15,23,42,0.18);
        }
        button:disabled {
          opacity: .6;
          cursor: default;
          box-shadow: none;
          transform: none;
        }
        .hint {
          font-size: 10px;
          color: #9ca3af;
          margin-top: 2px;
        }
        .reply-wrap {
          margin-top: 14px;
        }
        .reply-label {
          font-weight: 600;
          font-size: 13px;
          margin-bottom: 4px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          color: #374151;
        }
        .reply-box {
          border-radius: var(--radius);
          border: 1px solid var(--border);
          padding: 10px;
          min-height: 72px;
          background: #f9fafb;
          white-space: pre-wrap;
          font-size: 13px;
          line-height: 1.6;
        }
        .footer {
          margin-top: 10px;
          font-size: 9px;
          color: #9ca3af;
          display: flex;
          justify-content: space-between;
          gap: 8px;
          align-items: center;
        }
        .footer span.right {
          text-align: right;
        }
        @media (max-width: 600px) {
          .card { padding: 18px 14px 14px; border-radius: 18px; }
          h1 { font-size: 20px; }
        }
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="card">
          <h1>
            ことの葉スタジオ
            <span class="logo">为在日与向往日本生活的华人，提供多场景日语人格陪练</span>
          </h1>
          <p class="subtitle">
            从便利店、电车、会社，到留学面试、亲子沟通、旅行问路、情绪安慰，
            挑一个适合你的ことの葉人格，一起练出地道又得体的日语。
          </p>
          <div class="tags">
            <div class="tag">🏠 在日生活对话</div>
            <div class="tag">💼 日本职场敬语</div>
            <div class="tag">🎓 留学与面试表达</div>
            <div class="tag">👨‍👩‍👧 家长 & 单亲家庭支持</div>
            <div class="tag">🎮 动漫游戏 & ACG用语</div>
            <div class="tag">✈️ 旅行场景一句话</div>
          </div>

          <label for="mode">选择人格 / モード</label>
          <select id="mode">
            <option value="daily">🏠 日常日语场景教练｜ことの葉デイリー</option>
            <option value="campus">🎓 留学与校园场景｜ことの葉キャンパスナビ</option>
            <option value="office">💼 职场敬语与汇报｜ことの葉オフィス先輩</option>
            <option value="family">👨‍👩‍👧 家长 & 学校沟通｜ことの葉ファミリーサポート</option>
            <option value="comfort_soft">🌸 暖心陪练・柔｜温柔安慰＋教你温暖说法</option>
            <option value="comfort_calm">🕶 沉稳陪练・穏｜冷静支持＋教你得体表达</option>
            <option value="culture">🎮 ACG文化会话｜ことの葉カルチャートーク</option>
            <option value="travel">✈️ 旅行日语向导｜ことの葉トラベル</option>
          </select>
          <div class="hint">
            用中文描述你的场景就好，例如：
            「第一次去日本公司上班要怎么自我介绍？」或「单亲妈妈给老师写联系本」。
          </div>

          <label for="input">输入你的场景 / 心情 / 句子</label>
          <textarea id="input"
            placeholder="例如：今天上司帮了我，想用自然又有礼貌的日语发消息感谢他。\n或：我要去看病，想学怎么跟医生描述肚子疼。"></textarea>

          <button id="send">发送给 ことの葉 ▶</button>
          <div class="hint">快捷键：Ctrl / ⌘ + Enter 发送</div>

          <div class="reply-wrap">
            <div class="reply-label">
              <span>ことの葉回复</span>
              <span style="font-size:10px;color:#9ca3af;">包含日文句子＋平假名读音＋中文解释＋必要场景说明</span>
            </div>
            <div id="reply" class="reply-box">这里会出现针对你场景的人格化日语建议。</div>
          </div>

          <div class="footer">
            <span>体验版每日调用有限制，仅供内测和演示。</span>
            <span class="right">Powered by Kotonoha Studio</span>
          </div>
        </div>
      </div>

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
          replyEl.textContent = "考え中… / 正在为你组织最自然的表达…";
          sendBtn.disabled = true;
          try {
            const res = await fetch(endpoint, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                user_id: "web-playground",
                mode,
                message: text
              })
            });
            const data = await res.json();
            replyEl.textContent = data.reply || JSON.stringify(data, null, 2);
          } catch (e) {
            replyEl.textContent = "出错了，请稍后重试：" + e;
          } finally {
            sendBtn.disabled = false;
          }
        }

        sendBtn.addEventListener("click", send);
        inputEl.addEventListener("keydown", (e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            send();
          }
        });
      </script>
    </body>
    </html>
    """


# ===== 请求 / 响应模型 =====

class ChatRequest(BaseModel):
    user_id: str
    mode: Literal[
        "daily",
        "campus",
        "office",
        "family",
        "comfort_soft",
        "comfort_calm",
        "culture",
        "travel",
        # 兼容旧参数
        "tutor",
        "otaku_waifu",
        "otaku_boyfriend",
    ] = "daily"
    message: str
    episode: Optional[int] = None
    line_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ===== 人格设定 =====

def build_system_prompt(mode: str) -> str:
    # 兼容旧 mode 映射
    if mode == "tutor":
        mode = "daily"
    elif mode == "otaku_waifu":
        mode = "comfort_soft"
    elif mode == "otaku_boyfriend":
        mode = "comfort_calm"

    if mode == "daily":
        return (
            "你是「ことの葉デイリー」，面向在日或准备来日本生活的华人。\n"
            "核心：教用户在超市、便利店、医院、银行、电车、公司等真实场景中敢开口的自然日语。\n"
            "输出要求：\n"
            "【1. 日文句子】自然、不生硬。\n"
            "【2. 读音（平假名）】整句平假名。\n"
            "【3. 中文解释】说明语气、适用场景、礼貌程度，点出地道点。\n"
            "【4. 延伸】1-2 个类似表达或替代表达，简要说明差异。\n"
            "避免逐字翻译腔，避免教奇怪或只存在于动漫的说法。"
        )

    if mode == "campus":
        return (
            "你是「ことの葉キャンパスナビ」，留学与校园场景教练。\n"
            "面向想来日本读书或在日学生，帮助应对面试、自我介绍、课堂发言、研究室交流、打工等。\n"
            "输出要求同样使用：日文句子 + 平假名读音 + 中文解释 + 延伸表达，"
            "特别强调礼貌得体、给老师/面试官留下好印象的说法。"
        )

    if mode == "office":
        return (
            "你是「ことの葉オフィス先輩」，日本职场日语与敬语教练。\n"
            "面向在日会社员或准备求职的人，专注：邮件、汇报、开会发言、电话、请假、道歉、感谢等。\n"
            "输出：\n"
            "【日文句子】正式或半正式、自然。\n"
            "【读音（平假名）】帮助用户确认发音。\n"
            "【中文解释】点出敬语等级、上下关系与潜台词。\n"
            "【延伸】给出更柔/更硬的替代表达，说明什么时候用。"
        )

    if mode == "family":
        return (
            "你是「ことの葉ファミリーサポート」，专注家庭与学校沟通。\n"
            "面向在日家长（包括单亲家庭）。\n"
            "帮助写联络本、和保育园/学校老师沟通、说明家庭情况、表达感谢和担忧。\n"
            "风格：温和、不评判、替家长找到既诚实又保护孩子的说法。\n"
            "同样使用：日文句子 + 平假名 + 中文解释 + 场景说明。"
        )

    if mode == "comfort_soft":
        return (
            "你是「ことの葉コンフォート・柔」，暖心日语陪练（柔和版）。\n"
            "面向压力大、孤独或想被温柔对待的用户。\n"
            "先用 2-4 句自然日文（可少量夹中文）温柔回应，再教 1 个温暖表达：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】在日本日常或亲密关系中如何自然使用这个说法。\n"
            "不擦边、不色情，保持治愈、真诚。"
        )

    if mode == "comfort_calm":
        return (
            "你是「ことの葉コンフォート・穏」，沉稳日语陪练（冷静版）。\n"
            "像可靠的日本前辈/朋友，帮用户整理思路、给温柔建议。\n"
            "先用 2-4 句自然日文表达理解与支持，再教 1 个适合对上司/同事/家人使用的得体表达：\n"
            "含平假名读音与简短中文说明。\n"
            "不夸张不油腻，重点是安全感和可实用表达。"
        )

    if mode == "culture":
        return (
            "你是「ことの葉カルチャートーク」，ACG & 次文化会话教练。\n"
            "懂动画、漫画、游戏、偶像文化，教用户如何用自然日语聊作品、推、角色，而不让日本人尴尬。\n"
            "每次给：\n"
            "【日文说法】\n"
            "【读音（平假名）】\n"
            "【中文说明】解释语气和是否宅圈限定，用于哪些场合合适。"
        )

    if mode == "travel":
        return (
            "你是「ことの葉トラベルコンシェルジュ」，日本旅行日语向导。\n"
            "教用户在机场、车站、餐厅、商店、景点、药妆店中，一两句话解决问题。\n"
            "输出简短：日文句子 + 平假名 + 中文解释；优先简单好记、礼貌但不啰嗦的说法。"
        )

    # 兜底
    return (
        "你是一个友好的日语学习助手，面向中文母语用户，"
        "用自然日语+少量中文解释，给出简洁、实用、可以直接在日本使用的表达。"
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
    check_quota(req.user_id)

    # 映射旧 mode
    mode = req.mode
    if mode == "tutor":
        mode = "daily"
    elif mode == "otaku_waifu":
        mode = "comfort_soft"
    elif mode == "otaku_boyfriend":
        mode = "comfort_calm"

    system_prompt = build_system_prompt(mode)

    # 根据模式给额外格式指令
    if mode in ["daily", "campus", "office", "family", "travel"]:
        user_message = (
            "请严格按以下结构输出，内容简洁实用：\n"
            "【1. 日文句子】\n"
            "【2. 读音（平假名）】\n"
            "【3. 中文解释】说明语气、场景、礼貌程度。\n"
            "【4. 延伸】1-2个类似表达或常见替代表达（如有必要）。\n"
            f"我的具体需求是：{req.message}"
        )
    elif mode == "comfort_soft":
        user_message = (
            "请先用2-4句自然、温柔的日语（可少量夹中文）回应我的情绪，"
            "然后教我1个温暖、真实常用的表达，按以下结构输出：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】一句话说明在日本日常或亲密关系中如何自然使用。\n"
            f"我的具体情况是：{req.message}"
        )
    elif mode == "comfort_calm":
        user_message = (
            "请先用2-4句自然口语日语，像可靠前辈一样，冷静支持或给建议，"
            "然后给出1个适合现实中对上司/同事/家人使用的表达，按以下结构输出：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】一句话说明适用场景和语气。\n"
            f"我的具体情况是：{req.message}"
        )
    elif mode == "culture":
        user_message = (
            "请用自然日语帮我聊与ACG/次文化相关的话题，并教1个相关表达：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】说明是否是宅圈用语、适合和谁说、在什么场合用合适。\n"
            f"我的具体话题是：{req.message}"
        )
    else:
        user_message = req.message

    reply = await call_llm(system_prompt, user_message)
    return ChatResponse(reply=reply)
