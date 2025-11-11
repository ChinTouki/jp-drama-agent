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
FREE_LIMIT_PER_DAY = 5  # 每个 user_id 每天免费 5 条
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
                "如果你觉得「ことの葉スタジオ（言叶日语场景工坊）」对你有帮助，"
                "可以联系作者开通会员或长期版。"
            ),
        )

    info["count"] += 1


# ===== 基础路由 =====

@app.get("/")
def read_root():
    return {"message": "Kotonoha Studio（言叶日语场景工坊） is running."}


# ===== Playground 页面（美化版，多人格选择，按热度排序） =====

@app.get("/playground", response_class=HTMLResponse)
def playground():
    return """
    <!DOCTYPE html>
    <html lang="zh-cn">
    <head>
      <meta charset="UTF-8" />
      <title>ことの葉スタジオ（言叶日语场景工坊）｜多场景日语人格陪练</title>
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
            ことの葉スタジオ（言叶日语场景工坊）
            <span class="logo">为在日与向往日本生活的华语用户提供多场景日语人格陪练</span>
          </h1>
          <p class="subtitle">
            从电车、便利店、餐厅、会社、学校，到医院、育儿、住房、八卦和动漫日剧，
            按场景选人格，比普通聊天更贴近真实日本生活。
          </p>
          <div class="tags">
            <div class="tag">🏠 日常生活</div>
            <div class="tag">🍣 餐厅·咖啡·服装·理发店</div>
            <div class="tag">✈️ 旅行常用句</div>
            <div class="tag">💼 日本职场敬语</div>
            <div class="tag">🎓 留学·校园·打工</div>
            <div class="tag">🏥 医院就诊 & 孩子看病</div>
            <div class="tag">👨‍👩‍👧 家长·亲子·老师沟通</div>
            <div class="tag">🏡 租房·邻里·手续</div>
            <div class="tag">📺 动漫·日剧·综艺·游戏</div>
            <div class="tag">🗣 安全八卦 & 闲聊</div>
          </div>

          <label for="mode">选择人格 / モード（按常用程度排序）</label>
          <select id="mode">
            <option value="daily">🏠 日常日语场景｜ことの葉デイリー</option>
            <option value="service">🍣 店铺服务场景｜ことの葉サービストーク</option>
            <option value="travel">✈️ 旅行日语向导｜ことの葉トラベル</option>
            <option value="office">💼 职场敬语与汇报｜ことの葉オフィス先輩</option>
            <option value="campus">🎓 留学与校园场景｜ことの葉キャンパスナビ</option>
            <option value="medical">🏥 医院就诊 & 儿科沟通｜ことの葉メディカル会話</option>
            <option value="family">👨‍👩‍👧 家长 & 学校沟通｜ことの葉ファミリーサポート</option>
            <option value="parenting">👨‍👧 亲子沟通 & 教育｜ことの葉ペアレンティング</option>
            <option value="housing">🏡 租房·邻里·手续咨询｜ことの葉ライフサポート</option>
            <option value="culture">📺 动漫·日剧·综艺·游戏｜ことの葉カルチャートーク</option>
            <option value="gossip">🗣 妈妈友·邻居·同事闲聊｜ことの葉ご近所トーク</option>
            <option value="comfort_soft">🌸 暖心陪练・柔｜ことの葉コンフォート・柔</option>
            <option value="comfort_calm">🕶 沉稳陪练・穏｜ことの葉コンフォート・穏</option>
          </select>
          <div class="hint">
            直接用中文写，比如：「理发时想说不要剪太短」「当店员提醒客人不能在店内拍照」「孩子发烧去小儿科怎么说」。
          </div>

          <label for="input">输入你的场景 / 心情 / 句子</label>
          <textarea id="input"
            placeholder="例如：\n- 在连锁居酒屋打工，想用礼貌自然的日语招呼客人。\n- 想跟服装店店员问有没有小一号。\n- 给房东发消息，说马桶坏了。\n- 和妈妈友聊孩子上幼儿园的适应情况。"></textarea>

          <button id="send">发送给 ことの葉 ▶</button>
          <div class="hint">快捷键：Ctrl / ⌘ + Enter 发送</div>

          <div class="reply-wrap">
            <div class="reply-label">
              <span>ことの葉回复</span>
              <span style="font-size:10px;color:#9ca3af;">包含日文句子＋平假名读音＋中文解释＋必要场景提示</span>
            </div>
            <div id="reply" class="reply-box">这里会出现针对你场景的日语表达建议。</div>
          </div>

          <div class="footer">
            <span>体验版每日调用有限制；医疗等内容仅作语言参考，不替代专业诊疗。</span>
            <span class="right">Powered by Kotonoha Studio（言叶日语场景工坊）</span>
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
        "service",
        "office",
        "campus",
        "family",
        "parenting",
        "medical",
        "housing",
        "travel",
        "culture",
        "gossip",
        "comfort_soft",
        "comfort_calm",
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

    # 1. 日常生活
    if mode == "daily":
        return (
            "你是「ことの葉デイリー」，面向在日或准备来日本生活的华语用户。\n"
            "核心：教用户在超市、便利店、电车、简单社交等场景中敢开口的自然日语。\n"
            "输出：\n"
            "【1. 日文句子】自然不生硬。\n"
            "【2. 读音（平假名）】整句平假名。\n"
            "【3. 中文解释】说明语气、场景、礼貌程度、微妙差别。\n"
            "【4. 延伸】1-2 个类似表达或常见替换说法。\n"
            "避免奇怪直译和只存在于动漫里的夸张台词。"
        )

    # 2. 餐饮·店铺·理发等服务业
    if mode == "service":
        return (
            "你是「ことの葉サービストーク」，专门处理日本各类服务业场景的语言教练。\n"
            "涵盖：餐厅、咖啡店、居酒屋、便利店、百货、服装店、药妆店、理发店、美甲、美发沙龙等。\n"
            "需同时支持：店员/店长视角 & 顾客视角，让对话自然、有礼貌、符合日本服务业习惯。\n"
            "输出结构：\n"
            "【1. 日文句子】1-3句，对应用户身份（店员/老板/顾客）。\n"
            "【2. 读音（平假名）】\n"
            "【3. 中文解释】说明语气、敬语等级、适用场景（连锁店/小店/熟客等）。\n"
            "【4. 延伸】补充1-3个高频表达，如欢迎用语、推荐、拒绝、道歉、提醒规则等。\n"
            "不教无礼或过度油腻说法，强调专业友善。"
        )

    # 3. 职场
    if mode == "office":
        return (
            "你是「ことの葉オフィス先輩」，日本职场沟通教练。\n"
            "专注：邮件、聊天工具、会议、电话、请假、致谢、道歉、催进度等。\n"
            "输出包含句子＋平假名＋中文解释，重点讲敬语等级、上下关系与潜台词。"
        )

    # 4. 校园 / 留学
    if mode == "campus":
        return (
            "你是「ことの葉キャンパスナビ」，留学与校园场景教练。\n"
            "面向来日本读语言学校、专门学校、大学、大学院的学生。\n"
            "包括面试、自我介绍、课堂发言、研究室交流、和老师同学相处、打工沟通。\n"
            "使用：日文句子＋平假名＋中文解释＋延伸表达，突出礼貌、自然、自信。"
        )

    # 5. 家长 & 学校沟通
    if mode == "family":
        return (
            "你是「ことの葉ファミリーサポート」，专注家庭与学校沟通。\n"
            "面向在日有孩子的家长，涵盖多种家庭背景。\n"
            "帮助写联络本、和保育园/学校老师沟通、说明生活与家庭情况、表达感谢和担忧。\n"
            "风格温和、不评判，给家长既自然礼貌又保护孩子的表达方式。\n"
            "输出：日文句子＋平假名＋中文解释＋简短场景说明。"
        )

    # 6. 亲子沟通 & 教育
    if mode == "parenting":
        return (
            "你是「ことの葉ペアレンティング」，亲子沟通与教育表达教练。\n"
            "帮助家长用自然、尊重的日语和孩子沟通：提醒、表扬、设规则、安抚情绪、鼓励坚持等。\n"
            "避免辱骂、威胁和恐吓语言，倡导正向养育。\n"
            "输出：\n"
            "【日文句子】1-2句，可直接对孩子说；如有年龄信息，自动调整难度。\n"
            "【读音（平假名）】\n"
            "【中文解释】标注语气和适合年龄段。\n"
            "【延伸】给更温柔/更坚定等替代表达。"
        )

    # 7. 医院·医生·药局
    if mode == "medical":
        return (
            "你是「ことの葉メディカル会話」，日本医院/诊所/药局就诊时的语言教练。\n"
            "对象：成人患者和带孩子看病的家长。\n"
            "只提供如何【说明症状】【询问信息】【听懂常见用语】的语言示例，不做诊断，不给医疗结论。\n"
            "输出建议：\n"
            "【1. 日文句子】1-3句，包含症状、时间、部位、程度等关键信息。\n"
            "【2. 读音（平假名）】\n"
            "【3. 中文解释】说明这几句传达了什么、是否礼貌。\n"
            "【4. 常见词汇补充】1-5个：日文＋平假名＋中文，例如发烧、咳嗽、小儿科、挂号等。\n"
            "如症状严重，请提醒用户务必听从医生与专业机构判断。"
        )

    # 8. 租房·邻里·手续
    if mode == "housing":
        return (
            "你是「ことの葉ライフサポート」，负责租房、邻里关系及基础手续相关表达。\n"
            "涵盖：找房、中介沟通、签约、续约、退租、报修、邻居噪音、垃圾规则、区役所/市役所基础手续等。\n"
            "输出结构：日文句子＋平假名＋中文解释＋如有需要的小提示，强调礼貌、清晰、避免冲突。"
        )

    # 9. 旅行
    if mode == "travel":
        return (
            "你是「ことの葉トラベルコンシェルジュ」，日本旅行日语向导。\n"
            "教用户在机场、车站、餐厅、商店、药妆店、景点，用1-2句解决问题。\n"
            "输出：简单礼貌的日文句子＋平假名＋中文解释，优先好记好用。"
        )

    # 10. 动漫·日剧·综艺·游戏
    if mode == "culture":
        return (
            "你是「ことの葉カルチャートーク」，负责动漫、日剧、综艺、游戏等流行文化相关的日语表达。\n"
            "帮用户用自然日语聊作品、角色、演员、梗、推，不显得尴尬或用错场合。\n"
            "每次给：日文表达＋平假名＋中文说明（语气、是否粉丝向/圈内用、适用场景）。"
        )

    # 11. 安全八卦 & 闲聊
    if mode == "gossip":
        return (
            "你是「ことの葉ご近所トーク」，练习日本式轻松八卦和闲聊表达的教练。\n"
            "场景：妈妈友、邻居、同事之间的聊天。\n"
            "原则：只用匿名化/泛化例子，不点名现实人物，不造谣，不鼓励攻击或歧视。\n"
            "教委婉表达、含蓄评论和有分寸的吐槽，帮助用户掌握日本式社交分寸。\n"
            "输出：日文句子＋平假名＋中文解释＋1-3个相关安全用语。"
        )

    # 12. 暖心陪练（柔）
    if mode == "comfort_soft":
        return (
            "你是「ことの葉コンフォート・柔」，暖心日语陪练（柔和版）。\n"
            "先用2-4句自然、温柔的日语（可少量夹中文）回应和安慰用户，"
            "再教1个温暖且真实常用的表达：日文＋平假名＋简短中文说明。\n"
            "不擦边、不色情，像亲近朋友一样。"
        )

    # 13. 沉稳陪练（穏）
    if mode == "comfort_calm":
        return (
            "你是「ことの葉コンフォート・穏」，沉稳日语陪练（冷静版）。\n"
            "像可靠的日本前辈/同事/朋友，先用2-4句自然日语表达理解和建议，"
            "再给出1个适合对上司/同事/家人使用的得体表达：日文＋平假名＋简短中文说明。"
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

    # 兼容旧 mode
    mode = req.mode
    if mode == "tutor":
        mode = "daily"
    elif mode == "otaku_waifu":
        mode = "comfort_soft"
    elif mode == "otaku_boyfriend":
        mode = "comfort_calm"

    system_prompt = build_system_prompt(mode)

    # 结构化输出指令，根据人格微调
    if mode in ["daily", "service", "office", "campus", "family", "parenting", "housing", "travel"]:
        user_message = (
            "请严格按以下结构输出，内容简洁实用，适合中文母语者：\n"
            "【1. 日文句子】\n"
            "【2. 读音（平假名）】\n"
            "【3. 中文解释】说明语气、场景、礼貌程度及注意点。\n"
            "【4. 延伸】如有必要，给1-2个类似或更自然的替代表达。\n"
            f"我的具体需求是：{req.message}"
        )

    elif mode == "medical":
        user_message = (
            "这是医疗就诊相关的语言需求，请只从【如何向医生/护士/药剂师清楚说明情况】、"
            "【如何听懂常见用语】的角度回答，不做诊断，不替代医生。\n"
            "请按以下结构输出：\n"
            "【1. 日文句子】1-3句，帮助说明当前症状或需求。\n"
            "【2. 读音（平假名）】\n"
            "【3. 中文解释】\n"
            "【4. 常见词汇补充】1-5个相关单词/短语：日文＋平假名＋中文。\n"
            "如症状严重，请提醒务必遵从日本医生与专业机构判断。\n"
            f"我的具体情况是：{req.message}"
        )

    elif mode == "culture":
        user_message = (
            "请围绕动漫、日剧、综艺、游戏等话题，给出自然说法，避免中式尴尬。\n"
            "并教1个相关表达：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】语气、是否粉丝向/圈内用、适用场景。\n"
            f"我的具体话题是：{req.message}"
        )

    elif mode == "gossip":
        user_message = (
            "请用轻松但不恶意的日语，模拟日本人之间的日常闲聊/小八卦场景，"
            "例如妈妈友、邻居、同事之间的聊天。只使用虚构或泛化人物，不点名真实人物，"
            "不造谣、不鼓励攻击或歧视。\n"
            "结构：\n"
            "【日文句子】1-2句，自然口语。\n"
            "【读音（平假名）】\n"
            "【中文解释】说明语气和适用关系。\n"
            "【关键词补充】1-3个相关委婉表达。\n"
            f"我的具体话题是：{req.message}"
        )

    elif mode == "comfort_soft":
        user_message = (
            "请先用2-4句自然、温柔的日语（可少量夹中文）回应和安慰我，"
            "然后教我1个温暖常用的表达：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】一句话说明在日本日常或亲密关系中如何自然使用。\n"
            f"我的具体情况是：{req.message}"
        )

    elif mode == "comfort_calm":
        user_message = (
            "请先用2-4句自然日语，像可靠前辈一样冷静支持或给建议，"
            "然后给出1个适合对上司/同事/家人使用的得体表达：\n"
            "【日文表达】\n"
            "【读音（平假名）】\n"
            "【中文说明】一句话说明适用场景和语气。\n"
            f"我的具体情况是：{req.message}"
        )

    else:
        user_message = req.message

    reply = await call_llm(system_prompt, user_message)
    return ChatResponse(reply=reply)
