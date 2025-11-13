import os
from typing import Literal, Optional
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# 本地加载 .env；Render 上使用环境变量
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TTS_MODEL = os.getenv("LLM_TTS_MODEL", "gpt-4o-mini-tts")


client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
)

app = FastAPI()


# ===== 每日免费额度（MVP） =====
FREE_LIMIT_PER_DAY = 10  # 每个 user_id 每天免费 5 条
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
def render_playground_html() -> str:
    return """
    <!DOCTYPE html>
    <html lang="zh-cn">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>ことの葉スタジオ（言叶日语场景工坊）｜多场景日语陪练</title>
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
          padding: 16px 10px 24px;
          font-family: var(--font);
          background:
            radial-gradient(circle at top left, #e0f2fe 0, transparent 55%),
            radial-gradient(circle at top right, #fee2e2 0, transparent 55%),
            var(--bg);
          color: --primary;
        }
        .shell {
          max-width: 900px;
          margin: 0 auto;
        }
        .card {
          background: rgba(255, 255, 255, 0.98);
          border-radius: 20px;
          padding: 16px 12px 14px;
          box-shadow: 0 16px 40px rgba(15,23,42,0.06);
          border: 1px solid rgba(148,163,253,0.16);
          backdrop-filter: blur(8px);
        }
        h1 {
          font-size: 20px;
          margin: 0 0 4px;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 6px;
        }
        h1 span.logo {
          display: inline-flex;
          padding: 3px 7px;
          border-radius: 999px;
          background: var(--accent-soft);
          font-size: 10px;
          color: #9f1239;
        }
        .subtitle {
          font-size: 11px;
          color: #6b7280;
          margin: 0 0 10px;
          line-height: 1.5;
        }
        .tags {
          display: flex;
          flex-wrap: wrap;
          gap: 5px;
          margin-bottom: 10px;
          font-size: 9px;
        }
        .tag {
          padding: 3px 7px;
          border-radius: 999px;
          border: 1px solid var(--border);
          color: #6b7280;
          white-space: nowrap;
        }
        label {
          display: block;
          margin-top: 10px;
          margin-bottom: 3px;
          font-weight: 600;
          font-size: 11px;
          color: #374151;
        }
        select, textarea, button {
          width: 100%;
          font-size: 12px;
          padding: 9px 9px;
          border-radius: var(--radius);
          border: 1px solid var(--border);
          outline: none;
          transition: all .16s ease;
          background: #ffffff;
        }
        select:focus, textarea:focus {
          border-color: #818cf8;
          box-shadow: 0 0 0 2px rgba(129,140,248,0.16);
        }
        textarea {
          resize: vertical;
          min-height: 80px;
          line-height: 1.5;
        }
        button {
          margin-top: 8px;
          background: linear-gradient(to right, #111827, #1f2937);
          color: #fff;
          border: none;
          font-weight: 600;
          cursor: pointer;
          border-radius: 999px;
        }
        button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 8px 16px rgba(15,23,42,0.18);
        }
        button:disabled {
          opacity: .6;
          cursor: default;
          box-shadow: none;
          transform: none;
        }
        .btn-secondary {
          background: #4b5563;
        }
        .hint {
          font-size: 9px;
          color: #9ca3af;
          margin-top: 2px;
        }
        .reply-wrap {
          margin-top: 10px;
        }
        .reply-label {
          font-weight: 600;
          font-size: 11px;
          margin-bottom: 3px;
          display: flex;
          justify-content: space-between;
          gap: 6px;
          align-items: center;
          color: #374151;
        }
        .reply-label span.sub {
          font-size: 8px;
          color: #9ca3af;
        }
        .reply-box {
          border-radius: 14px;
          border: 1px solid var(--border);
          padding: 8px;
          min-height: 80px;
          background: #f9fafb;
          white-space: pre-wrap;
          font-size: 11px;
          line-height: 1.6;
        }

        /* === 紧凑按钮 + 放大输入/回答（覆盖前面的默认样式）=== */

/* 1) 按钮更瘦、留白更小 */
button{
  padding: 6px 10px;        /* 原先 padding 比较大 → 改小 */
  font-size: 12px;          /* 字体更小更紧凑 */
  height: 32px;             /* 统一高度，行高由浏览器算 */
  margin-top: 4px;          /* 顶部间距更小 */
  border-radius: 10px;      /* 圆角略小，视觉更利落 */
}

/* 如果你有“主按钮”（比如 语音输入 / 发送），可稍微大一点 */
#btnMic, #btnSend{
  padding: 8px 14px;
  font-size: 13px;
  height: 36px;
}

/* 二级按钮（朗读当前、使用本机朗读、上一条、下一条、清空）弱一点的样式 */
#btnTTS, #btnReadReply, #btnPrev, #btnNext, #btnClear{
  opacity: .85;
  border: 1px solid #e5e7eb;
  background: #fff;
}

/* 2) 输入框更大（如果你的输入是 textarea/#inputText） */
textarea, #inputText{
  min-height: 120px;        /* 原来太矮 → 拉高 */
  max-height: 40vh;         /* 最多占 40% 视口高度，可拖拽 */
  resize: vertical;         /* 允许竖向拖拽 */
  font-size: 14px;
  line-height: 1.6;
  padding: 10px 12px;
}

/* 3) 回答框更大、字更清晰 */
.reply-box{
  min-height: 160px;        /* 原来是 80px → 提升为 160px */
  font-size: 13px;          /* 略放大可读性 */
  line-height: 1.6;
  overflow: auto;           /* 内容多时可滚动 */
}

/* 4) 按钮容器（如果有）允许换行，挤占更少垂直空间 */
.toolbar, .actions, .button-row{
  display: flex;
  flex-wrap: wrap;          /* 自动换行 */
  gap: 6px;
  align-items: center;
  margin-top: 6px;
}

/* 5) 小屏时只显示图标（如果按钮里有 .label 文本容器） */
@media (max-width: 900px){
  .btn .label{ display:none; }
  .btn{ width: 36px; padding: 0; justify-content: center; }
}


        .footer {
          margin-top: 8px;
          font-size: 8px;
          color: #9ca3af;
          display: flex;
          justify-content: space-between;
          gap: 8px;
          align-items: center;
          flex-wrap: wrap;
        }
        .footer span.right {
          text-align: right;
        }
        audio {
          width: 100%;
          margin-top: 4px;
        }
        @media (min-width: 640px) {
          body { padding: 24px 16px 32px; }
          .card { padding: 22px 18px 18px; border-radius: 24px; }
          h1 { font-size: 24px; }
          .subtitle { font-size: 12px; }
          .tags { font-size: 10px; }
          label { font-size: 12px; }
          select, textarea, button { font-size: 13px; }
          .reply-box { font-size: 12px; }
          .footer { font-size: 9px; }
        }

        /* === 超紧凑：按钮更小、输入/回答更大 === */

/* 让聊天区吃满，输入区贴底 */
html,body{height:100%;}
body{display:flex;flex-direction:column;min-height:100vh;}
.app,.container,.chat,main{display:flex;flex-direction:column;flex:1 1 auto;min-height:0;}
.messages,.chat-messages,#messages{flex:1 1 auto;min-height:0;overflow:auto;padding:8px 12px;}

/* 输入框更高 */
textarea, #inputText{
  min-height:140px;      /* 再抬高 */
  max-height:45vh;       /* 允许更高 */
  font-size:15px; line-height:1.6; padding:10px 12px;
  resize:vertical;
}

/* 按钮区超紧凑 */
.toolbar, .actions, .button-row{
  display:flex; flex-wrap:wrap; gap:4px; align-items:center;
  margin-top:4px; row-gap:4px;
}

/* 统一把所有按钮“瘦身” */
button, .btn,
#btnMic,#btnSend,#btnReadReply,#btnTTS,#btnClear,#btnPrev,#btnNext{
  padding:4px 8px;
  height:28px;
  font-size:11px;
  line-height:1;
  border-radius:8px;
  margin-top:2px;
  min-width:auto;        /* 允许更窄 */
}

/* 主按钮（语音输入/发送）稍大一丁点以突出 */
#btnMic,#btnSend{
  padding:6px 10px;
  height:30px;
  font-size:12px;
  font-weight:600;
}

/* 次要动作弱化（更不抢占视觉） */
#btnReadReply,#btnTTS,#btnPrev,#btnNext,#btnClear{
  border:1px solid #e5e7eb; background:#fff; opacity:.85;
}

/* 小屏或你想“图标化”：隐藏文字标签（若按钮内有 .label） */
@media (max-width: 1100px){
  .btn .label{display:none;}
  .btn{width:32px; padding:0; justify-content:center;}
}

/* 顶部栏也更薄一点（若存在） */
header{padding:4px 8px !important;}

/* 打开任意一行即可 */
#btnPrev, #btnNext { display:none !important; }   /* 上一条/下一条 */
#btnClear { display:none !important; }            /* 清空输入 */
#btnReadReply { display:none !important; }        /* 朗读当前回复 */
#btnTTS { display:none !important; }              /* 使用本机朗读 */

      </style>
    </head>
    <body>
      <div class="shell">
        <div class="card">
          <h1>
            ことの葉スタジオ（言叶日语场景工坊）
            <span class="logo">在日 & 来日前｜手机优先的多场景日语陪练</span>
          </h1>
          <p class="subtitle">
            选择场景人格，用中文描述需求，
            即时获得「日文句子＋平假名＋中文解释」，适合在手机浏览器中边看边学。
          </p>

          <div class="tags">
            <div class="tag">🏠 日常生活</div>
            <div class="tag">🍣 餐厅・店铺・理发店</div>
            <div class="tag">✈️ 旅行</div>
            <div class="tag">💼 职场</div>
            <div class="tag">🎓 留学</div>
            <div class="tag">🏥 医院 & 孩子看病</div>
            <div class="tag">👨‍👩‍👧 家长·亲子·老师</div>
            <div class="tag">🏡 租房·邻里·手续</div>
            <div class="tag">📺 动漫·日剧·综艺·游戏</div>
            <div class="tag">🗣 安全八卦·关西ことば</div>
          </div>

          <label for="mode">选择人格 / モード</label>
          <select id="mode">
            <option value="daily">🏠 日常日语场景｜ことの葉デイリー</option>
            <option value="service">🍣 店铺服务场景｜ことの葉サービストーク</option>
            <option value="travel">✈️ 旅行日语向导｜ことの葉トラベル</option>
            <option value="office">💼 职场敬语与汇报｜ことの葉オフィス先輩</option>
            <option value="campus">🎓 留学与校园场景｜ことの葉キャンパスナビ</option>
            <option value="medical">🏥 医院就诊 & 儿科沟通｜ことの葉メディカル会話</option>
            <option value="family">👨‍👩‍👧 家长 & 学校沟通｜ことの葉ファミリーサポート</option>
            <option value="parenting">👨‍👧 亲子沟通 & 教育｜ことの葉ペアレンティング</option>
            <option value="housing">🏡 租房·邻里·手续｜ことの葉ライフサポート</option>
            <option value="culture">📺 动漫·日剧·综艺·游戏｜ことの葉カルチャートーク</option>
            <option value="kansai">🌀 关西ことば入门｜ことの葉関西ことば</option>
            <option value="gossip">🗣 妈妈友·邻居·同事闲聊｜ことの葉ご近所トーク</option>
            <option value="comfort_soft">🌸 暖心陪练・柔｜ことの葉コンフォート・柔</option>
            <option value="comfort_calm">🕶 沉稳陪练・穏｜ことの葉コンフォート・穏</option>
          </select>
          <div class="hint">
            示例：理发时说「不要剪太短」；做店员欢迎客人；孩子生病说明症状；和关西同事轻松寒暄等等。
          </div>

          <label for="input">用中文描述你的场景</label>
          <textarea id="input"
            placeholder="例如：\n- 明天第一次去日本公司上班想自我介绍。\n- 孩子咳嗽一周了，想在医院说清楚。\n- 在大阪打工想学自然的关西ことば问候客人。"></textarea>
          <label for="srLang" style="margin-right:6px;">音声言語</label>
         <select id="srLang">
         <option value="zh-CN" selected>中文（普通话, 中国）</option>
         <option value="ja-JP">日本語（日本）</option>
         </select>

<button id="btnMic" type="button" aria-pressed="false" title="音声入力 (Ctrl+M)">🎤 语音输入</button>
<small id="micStatus" style="margin-left:8px;color:#666">待机中</small>

            
<button id="send">发送给 ことの葉 ▶ 生成日语表达</button>
<button id="speak" class="btn-secondary">🔊 朗读当前回复（需要已开通语音额度）</button>
<button id="speak-local" class="btn-secondary">
  📱 使用本机朗读（平假名示范发音）
</button>

<div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:4px;">
  <button id="clear-input" class="btn-secondary" style="flex:1;">🧹 清空输入</button>
  <button id="prev-history" class="btn-secondary" style="flex:1;">⬅ 上一条</button>
  <button id="next-history" class="btn-secondary" style="flex:1;">下一条 ➜</button>
</div>

<div id="reply" class="reply-box">这里会出现针对你场景的日语表达建议。</div>
<audio id="audio" controls></audio>


          <div class="footer">
            <span>体验版每日调用有限制；医疗相关内容仅作语言示例，不替代专业诊疗。</span>
            <span class="right">手机浏览器 / INS / TikTok 内置打开均适用。</span>
          </div>
        </div>
      </div>

 <script>
document.addEventListener('DOMContentLoaded', function () {
  var chatEndpoint = "/agent/chat";
  var ttsEndpoint = "/tts";

  var sendBtn = document.getElementById("send");
  var speakBtn = document.getElementById("speak");
  var speakLocalBtn = document.getElementById("speak-local");
  var clearInputBtn = document.getElementById("clear-input");
  var prevBtn = document.getElementById("prev-history");
  var nextBtn = document.getElementById("next-history");

  var inputEl = document.getElementById("input");
  var modeEl = document.getElementById("mode");
  var replyEl = document.getElementById("reply");
  var audioEl = document.getElementById("audio");

  var history = [];     // { mode, input, reply }
  var historyIndex = -1;
  var localReadState = "idle"; // idle | playing | paused

  // 只抽取【读音（平假名）】里的平假名来朗读
  function extractHiraganaOnly(text) {
    var parts = text.split("\\n");  // 注意这里是 \\n
    var result = "";
    var inReading = false;

    for (var i = 0; i < parts.length; i++) {
      var line = (parts[i] || "").trim();

      // 命中“读音”/“平假名”标题，进入读音区（从下一行开始）
      if (line.indexOf("读音") !== -1 || line.indexOf("平假名") !== -1) {
        inReading = true;
        continue;
      }
      // 在读音区内，遇到新【…】标题（且不含读音字样）则结束
      if (inReading && line.indexOf("【") === 0 &&
          line.indexOf("读音") === -1 && line.indexOf("平假名") === -1) {
        break;
      }
      if (!inReading) continue;

      // 只保留平假名和长音符号
      var cleaned = line.replace(/[^ぁ-んー]+/g, "");
      if (cleaned) result += cleaned + "。";
    }
    return result.trim();
  }

  function resetLocalSpeakButton() {
    if (speakLocalBtn) {
      speakLocalBtn.textContent = "📱 使用本机朗读（平假名示范发音）";
    }
  }

  function stopLocalRead() {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    localReadState = "idle";
    resetLocalSpeakButton();
  }

  function updateHistoryButtons() {
    if (!prevBtn || !nextBtn) return;
    prevBtn.disabled = historyIndex <= 0;
    nextBtn.disabled = historyIndex < 0 || historyIndex >= history.length - 1;
  }

  function loadHistory(index) {
    if (index < 0 || index >= history.length) return;
    historyIndex = index;
    var item = history[historyIndex];
    if (modeEl) modeEl.value = item.mode;
    if (inputEl) inputEl.value = item.input;
    replyEl.textContent = item.reply;
    if (audioEl) audioEl.removeAttribute("src");
    stopLocalRead();
    updateHistoryButtons();
  }

  // 发送
  function send() {
    var text = (inputEl && inputEl.value || "").trim();
    if (!text) return;
    var mode = modeEl ? modeEl.value : "daily";

    replyEl.textContent = "考え中… / 正在为你组织最自然的表达…";
    if (audioEl) audioEl.removeAttribute("src");
    stopLocalRead();
    if (sendBtn) sendBtn.disabled = true;

    fetch(chatEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: "web-playground", mode: mode, message: text })
    })
    .then(function (res) { return res.json(); })
    .then(function (data) {
      var reply = data.reply || JSON.stringify(data, null, 2);
      replyEl.textContent = reply;
      history.push({ mode: mode, input: text, reply: reply });
      historyIndex = history.length - 1;
      updateHistoryButtons();
    })
    .catch(function (e) {
      replyEl.textContent = "出错了，请稍后重试：" + e;
    })
    .finally(function () {
      if (sendBtn) sendBtn.disabled = false;
    });
  }

  // （可选）云端 TTS
  function speak() {
    if (!speakBtn) return;
    var text = replyEl.textContent.trim();
    if (!text) {
      replyEl.textContent = "请先生成一条回复，再点击朗读。";
      return;
    }
    speakBtn.disabled = true;
    speakBtn.textContent = "语音生成中…";
    fetch(ttsEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, voice: "alloy" })
    })
    .then(function (res) {
      if (!res.ok) return res.json().then(function (e) { throw e; });
      return res.blob();
    })
    .then(function (blob) {
      if (blob instanceof Blob && audioEl) {
        var url = URL.createObjectURL(blob);
        audioEl.src = url;
        audioEl.play();
      } else {
        replyEl.textContent = "语音生成失败：权限或额度问题。";
      }
    })
    .catch(function (e) {
      var msg = (e && e.detail) ? e.detail : (e.status || "") + " " + (e.message || "");
      replyEl.textContent = "语音请求出错：" + msg;
    })
    .finally(function () {
      speakBtn.disabled = false;
      speakBtn.textContent = "🔊 朗读当前回复（需要已开通语音额度）";
    });
  }

  // 本机朗读：只读平假名；播放 → 暂停 → 继续
  function speakLocal() {
    if (!window.speechSynthesis) {
      replyEl.textContent = "当前浏览器不支持本机语音朗读功能，请尝试用系统浏览器或电脑打开。";
      return;
    }

    if (localReadState === "playing") {
      window.speechSynthesis.pause();
      localReadState = "paused";
      if (speakLocalBtn) speakLocalBtn.textContent = "▶ 继续本机朗读";
      return;
    }

    if (localReadState === "paused") {
      window.speechSynthesis.resume();
      localReadState = "playing";
      if (speakLocalBtn) speakLocalBtn.textContent = "⏸ 暂停本机朗读";
      return;
    }

    var raw = replyEl.textContent.trim();
    if (!raw) {
      replyEl.textContent = "请先生成一条日语回复，再点击本机朗读。";
      return;
    }

    var hira = extractHiraganaOnly(raw);
    if (!hira) {
      replyEl.textContent = "当前回复中没有可朗读的平假名内容，请确认已生成带读音的回复。";
      return;
    }

    stopLocalRead(); // 先清理之前的
    var utter = new SpeechSynthesisUtterance(hira);
    utter.lang = "ja-JP";

    var voices = window.speechSynthesis.getVoices();
    for (var i = 0; i < voices.length; i++) {
      var v = voices[i];
      if (v.lang && v.lang.toLowerCase().indexOf("ja") === 0) {
        utter.voice = v; break;
      }
    }

    localReadState = "playing";
    if (speakLocalBtn) speakLocalBtn.textContent = "⏸ 暂停本机朗读";

    utter.onend = function () { localReadState = "idle"; resetLocalSpeakButton(); };
    utter.onerror = function () { localReadState = "idle"; resetLocalSpeakButton(); };

    window.speechSynthesis.speak(utter);
  }

  // 清空输入
  function clearInput() {
    if (inputEl) inputEl.value = "";
    stopLocalRead();
  }

  // 历史导航
  function showPrev() { if (historyIndex > 0) loadHistory(historyIndex - 1); }
  function showNext() { if (historyIndex >= 0 && historyIndex < history.length - 1) loadHistory(historyIndex + 1); }

  // 事件绑定
  if (sendBtn) sendBtn.addEventListener("click", send);
  if (speakBtn) speakBtn.addEventListener("click", speak);
  if (speakLocalBtn) speakLocalBtn.addEventListener("click", speakLocal);
  if (clearInputBtn) clearInputBtn.addEventListener("click", clearInput);
  if (prevBtn) prevBtn.addEventListener("click", showPrev);
  if (nextBtn) nextBtn.addEventListener("click", showNext);
  if (inputEl) {
    inputEl.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); send(); }
    });
  }

  // 初始化
  if (window.speechSynthesis) window.speechSynthesis.getVoices();
  updateHistoryButtons();
});
</script>

<script>
// ========== Speech Input Add-on (safe, non-destructive) ==========
(() => {
  // ---- 必填：改成你页面里真实的选择器 ----
  const INPUT_SELECTOR = "#input";   // 你的输入框（如 textarea 或 input）的选择器
  const SEND_BTN_SELECTOR = "#btnSend";  // 你的发送按钮选择器（可留空）
  const LANG_DEFAULT = "zh-CN";          // 默认中文（可切换 ja-JP）

  // ---- 可选：行为开关 ----
  const CONTINUOUS = false;              // true=持续听写
  const INTERIM = true;                  // true=状态栏显示临时结果
  const APPEND_MODE = false;             // true=在输入框末尾追加；false=覆盖
  const AUTO_CLICK_SEND_ON_END = false;  // true=识别结束后自动点发送按钮
  const CALL_CUSTOM_SUBMIT_FN = "";      // 如果你有全局函数如 runAnalysis，则填 "runAnalysis"

  // ---- 麦克风 UI（若你已放置了按钮/下拉，保持相同 id 即可；没有就自动创建）----
  let $btn = document.getElementById("btnMic");
  let $status = document.getElementById("micStatus");
  let $langSel = document.getElementById("srLang");

  function ensureBasicUI() {
    // 若页面没有这三个控件，就简易创建并插到输入框后面
    const $input = document.querySelector(INPUT_SELECTOR);
    if (!$input) return; // 没有就什么都不干（不报错）
    const host = $input.parentElement || document.body;

    if (!$langSel) {
      $langSel = document.createElement("select");
      $langSel.id = "srLang";
      $langSel.innerHTML = `
        <option value="zh-CN" selected>中文</option>
        <option value="ja-JP">日本語</option>
      `;
      $langSel.style.marginRight = "6px";
      host.appendChild($langSel);
    }
    if (!$btn) {
      $btn = document.createElement("button");
      $btn.id = "btnMic";
      $btn.type = "button";
      $btn.textContent = "🎤 语音输入";
      $btn.style.marginLeft = "6px";
      host.appendChild($btn);
    }
    if (!$status) {
      $status = document.createElement("small");
      $status.id = "micStatus";
      $status.textContent = "待机中";
      $status.style.marginLeft = "8px";
      $status.style.color = "#666";
      host.appendChild($status);
    }
  }

  // ---- 安全取元素，不存在就返回 null，不抛错 ----
  function getEl(sel) {
    try { return sel ? document.querySelector(sel) : null; } catch { return null; }
  }

  function setStatus(msg) { if ($status) $status.textContent = msg; }
  function setPressed(on) {
    if ($btn) {
      $btn.setAttribute("aria-pressed", String(on));
      $btn.style.background = on ? "rgba(0,128,255,.08)" : "";
      $btn.style.borderColor = on ? "dodgerblue" : "";
    }
  }

  function sanitize(s){ return String(s||"").replace(/\s+/g," ").trim(); }
  function autoPunct(text, lang) {
    const end = text.slice(-1);
    const enders = lang.startsWith("zh") ? "。！？!?" : "。！？!?";
    if (!enders.includes(end)) return text + (lang.startsWith("zh") ? "。" : "。");
    return text;
  }

  function writeToInput(text) {
    const $input = getEl(INPUT_SELECTOR);
    if (!$input) return;
    const val = $input.value ?? "";
    if (APPEND_MODE) {
      const sep = val && !/\s$/.test(val) ? " " : "";
      $input.value = val + sep + text;
    } else {
      $input.value = text;
    }
    // 触发你已有监听
    $input.focus?.();
    const len = $input.value.length;
    $input.setSelectionRange?.(len, len);
    $input.dispatchEvent(new Event("input",{bubbles:true}));
    $input.dispatchEvent(new Event("change",{bubbles:true}));
  }

  function triggerSubmit() {
    // 方案A：调用你的全局函数
    if (CALL_CUSTOM_SUBMIT_FN && typeof window[CALL_CUSTOM_SUBMIT_FN] === "function") {
      try { window[CALL_CUSTOM_SUBMIT_FN](); return; } catch(e){ console.warn(e); }
    }
    // 方案B：模拟点击发送按钮
    const $send = getEl(SEND_BTN_SELECTOR);
    if ($send) { $send.click(); }
  }

  function main() {
    ensureBasicUI();

    const $input = getEl(INPUT_SELECTOR);
    if (!$input) {
      console.warn("[speech-addon] 找不到输入框：", INPUT_SELECTOR);
      return; // 不破坏其他逻辑
    }

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      if ($btn){ $btn.disabled = true; $btn.title = "此浏览器不支持语音输入"; }
      setStatus("音声非対応");
      return;
    }

    let rec = null, listening = false, finalBuffer = "";

    function newRec() {
      const r = new SR();
      const lang = $langSel?.value || LANG_DEFAULT;
      r.lang = lang;
      r.continuous = CONTINUOUS;
      r.interimResults = INTERIM;
      r.maxAlternatives = 1;

      r.onstart = () => {
        listening = true;
        setPressed(true);
        setStatus(lang.startsWith("zh") ? "聆听中…（再次点击停止）" : "傾聴中…（もう一度クリックで停止）");
      };

      r.onresult = (ev) => {
        let interim = "";
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const res = ev.results[i];
          const text = res[0]?.transcript || "";
          if (res.isFinal) finalBuffer += text;
          else if (INTERIM) interim += text;
        }
        if (INTERIM && interim) setStatus((r.lang.startsWith("zh")?"临时: ":"暫定: ")+sanitize(interim));
      };

      r.onerror = (e) => {
        setStatus("错误/エラー: " + e.error);
        setPressed(false);
        listening = false;
      };

      r.onend = () => {
        setPressed(false);
        listening = false;
        if (finalBuffer.trim()) {
          const text = autoPunct(finalBuffer.trim(), r.lang);
          finalBuffer = "";
          writeToInput(text);
          setStatus(r.lang.startsWith("zh") ? "识别结束" : "認識終了");
          if (AUTO_CLICK_SEND_ON_END) triggerSubmit();
        } else {
          setStatus(r.lang.startsWith("zh") ? "待机中" : "待機中");
        }
      };
      return r;
    }

    function ensureRec(){ if (!rec) rec = newRec(); }

    $langSel?.addEventListener("change", () => {
      if (listening) { try { rec?.stop(); } catch{} }
      rec = newRec();
      setStatus(($langSel.value||"").startsWith("zh") ? "已切换到中文" : "日本語に切替えました");
    });

    function toggle() {
      ensureRec();
      try {
        if (!listening) { finalBuffer = ""; rec.start(); }
        else { rec.stop(); }
      } catch (err) {
        setStatus("无法开始/開始できません: " + (err.message || err.name || "unknown"));
        setPressed(false);
        listening = false;
      }
    }

    $btn?.addEventListener("click", toggle);
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey||e.metaKey) && (e.key==="m"||e.key==="M")) { e.preventDefault(); toggle(); }
    });

    setStatus("待机中"); // 初始状态
  }

  // 等 DOM ready，避免元素未渲染导致报错
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main, {once:true});
  } else {
    main();
  }
})();
</script>

<script>
// ====== Smart TTS (zh/ja) - free, local voices, safe attach ======
(() => {
  // ——可选：你的朗读按钮与文本来源——
  const BTN_TTS_SELECTOR = "#btnTTS";      // 若有“本机朗读”按钮，改成你的选择器；没有可留空
  const INPUT_SELECTOR  = '[data-tts-target="true"], [data-sr-target="true"], #inputText, #userInput, textarea, input[type="text"]';
  // ——默认参数——
  const DEFAULT_RATE   = 1.0;   // 语速 0.1~10
  const DEFAULT_PITCH  = 1.0;   // 音高 0~2
  const DEFAULT_VOLUME = 1.0;   // 音量 0~1
  const FALLBACK_LANG  = "zh-CN"; // 无法判断时默认中文

  // 各语言优先候选（按名称包含关键字匹配，Edge 的 Microsoft * Natural 很好用）
  const PREFS = {
    "zh-CN": [
      "Microsoft Xiaoxiao", "Microsoft Yunxi", "Microsoft Yunyang", // Edge
      "Google 普通话", "Google Mandarin", "Google 中国"
    ],
    "ja-JP": [
      "Microsoft Haruka", "Microsoft Ayumi", // Edge
      "Google 日本語", "Google 日本语"
    ]
  };

  // 语言检测：发现日文假名 → ja-JP；否则包含大量汉字→zh-CN；否则默认
  function detectLang(text) {
    const t = (text || "").trim();
    if (!t) return FALLBACK_LANG;
    const hasHiragana = /[\u3040-\u309F]/.test(t);
    const hasKatakana = /[\u30A0-\u30FF\u31F0-\u31FF]/.test(t);
    if (hasHiragana || hasKatakana) return "ja-JP";
    const hanCount = (t.match(/[\u4E00-\u9FFF]/g) || []).length;
    if (hanCount >= Math.max(2, t.length * 0.1)) return "zh-CN";
    return FALLBACK_LANG;
  }

  // 等待系统 voices 就绪（Chrome/Edge 需要异步）
  function waitVoicesReady(timeoutMs = 2000) {
    return new Promise(resolve => {
      const has = speechSynthesis.getVoices();
      if (has && has.length) return resolve(has);
      let done = false;
      const timer = setTimeout(() => { if (!done) { done = true; resolve(speechSynthesis.getVoices()); } }, timeoutMs);
      window.speechSynthesis.onvoiceschanged = () => {
        if (!done) { done = true; clearTimeout(timer); resolve(speechSynthesis.getVoices()); }
      };
      // 触发一次加载
      speechSynthesis.getVoices();
    });
  }

  function pickBestVoice(lang, voices) {
    const list = voices || speechSynthesis.getVoices() || [];
    if (!list.length) return null;
    const prefs = PREFS[lang] || [];
    // 1) 完整 lang 精确匹配 + 首选名称
    const p1 = list.find(v => v.lang === lang && prefs.some(p => (v.name||"").includes(p)));
    if (p1) return p1;
    // 2) 完整 lang 精确匹配（不看名称）
    const p2 = list.find(v => v.lang === lang);
    if (p2) return p2;
    // 3) 语言前缀匹配（如 zh-XX / ja-XX）
    const base = lang.split("-")[0];
    const p3 = list.find(v => (v.lang||"").toLowerCase().startsWith(base));
    if (p3) return p3;
    // 4) 兜底
    return list[0];
  }

  // 核心朗读函数
  async function speakSmart(text, opts = {}) {
    if (!("speechSynthesis" in window)) {
      console.warn("[smart-tts] 当前浏览器不支持 SpeechSynthesis");
      return;
    }
    const msg = String(text ?? "").trim();
    if (!msg) return;

    // 避免叠音
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel();

    // 语言：优先显式传入，其次自动检测
    const lang = opts.lang || detectLang(msg);

    // 等待声音可用并挑选
    const voices = await waitVoicesReady();
    const voice = pickBestVoice(lang, voices);

    const u = new SpeechSynthesisUtterance(msg);
    if (voice) u.voice = voice;
    u.lang   = voice?.lang || lang;
    u.rate   = opts.rate   ?? DEFAULT_RATE;
    u.pitch  = opts.pitch  ?? DEFAULT_PITCH;
    u.volume = opts.volume ?? DEFAULT_VOLUME;

    // 可选：更自然的小停顿（简单句读替换）
    u.text = msg.replace(/([，、,])\s*/g, "$1 ").replace(/([。！？!?])\s*/g, "$1 ");

    u.onerror = e => console.warn("[smart-tts] error:", e.error || e.name || e);
    speechSynthesis.speak(u);
  }

  // ——对外暴露：window.speakSmart(text, {lang, rate, pitch, volume})——
  window.speakSmart = speakSmart;

  // ——如果页面有“朗读”按钮：点击朗读输入框里的内容——
  function getInputEl() {
    // 优先 data-tts/sr 标记；否则退化到常见输入框
    const el = document.querySelector(INPUT_SELECTOR);
    return el && isVisible(el) ? el : null;
  }
  function isVisible(el){
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.display!=="none" && s.visibility!=="hidden" && r.width>0 && r.height>0;
  }

  function attachButton() {
    const btn = document.querySelector(BTN_TTS_SELECTOR);
    if (!btn) return;
    btn.addEventListener("click", () => {
      const el = getInputEl();
      // contenteditable 也支持
      const text = el ? (("value" in el) ? el.value : el.innerText || el.textContent || "") : "";
      speakSmart(text);
    });
  }

  // DOM ready 后挂载
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachButton, { once: true });
  } else {
    attachButton();
  }

})();
</script>

<script>
(() => {
  // ——日语读音专用：强制 ja-JP、假名安全、分句停顿——

  // 可按需调整
  const JA_RATE   = 1.02;  // 稍慢更清晰
  const JA_PITCH  = 1.0;
  const JA_VOLUME = 1.0;

  // 优先选择更自然的日语 voice 名称关键字（Edge/Chrome 免费可用）
  const JAPANESE_PREF_NAMES = [
    "Microsoft Haruka", "Microsoft Ayumi", "Microsoft Nanami",
    "Google 日本語", "Google 日本语"
  ];

  function toJaFriendlyText(raw) {
    if (!raw) return "";
    // 1) 统一到全角（含半角片假名→全角）与规范形
    let t = raw.normalize("NFKC");

    // 2) 英文标点→日文标点，加入合理停顿
    t = t
      .replace(/[,，]/g, "、")
      .replace(/[.．。]{1,}/g, "。")
      .replace(/[!！]{1,}/g, "！")
      .replace(/[?？]{1,}/g, "？")
      // 括号前后加轻微停顿
      .replace(/（/g, "（")
      .replace(/）/g, "）")
      // 冒号/分号→停顿
      .replace(/[:：]/g, "、")
      .replace(/[;；]/g, "、");

    // 3) 连续空白→单空格；空格转为小停顿（读音里常见空格）
    t = t.replace(/\s+/g, " ");
    t = t.replace(/ /g, "、");

    // 4) 规范中点与长音
    t = t.replace(/・/g, "・").replace(/ｰ/g, "ー");

    // 5) 去掉多余停顿
    t = t.replace(/、{2,}/g, "、").replace(/。{2,}/g, "。");

    // 6) 句末补句点
    if (!/[。！？]$/.test(t)) t += "。";
    return t;
  }

  function splitJaChunks(t) {
    // 依据「。！？」「；、」等切块，避免一次性很长导致发音奇怪
    const parts = t.split(/(?<=[。！？])/);
    // 再把极长段落用顿号再切一层
    const chunks = [];
    for (const p of parts) {
      const s = p.trim();
      if (!s) continue;
      if (s.length > 40 && s.includes("、")) {
        s.split("、").forEach((q, i, arr) => {
          const q2 = q.trim();
          if (q2) chunks.push(i < arr.length - 1 ? (q2 + "、") : q2);
        });
      } else {
        chunks.push(s);
      }
    }
    return chunks;
  }

  function waitVoicesReady(timeoutMs = 2000) {
    return new Promise(resolve => {
      const now = speechSynthesis.getVoices();
      if (now && now.length) return resolve(now);
      let done = false;
      const timer = setTimeout(() => { if (!done) { done = true; resolve(speechSynthesis.getVoices()); } }, timeoutMs);
      speechSynthesis.onvoiceschanged = () => {
        if (!done) { done = true; clearTimeout(timer); resolve(speechSynthesis.getVoices()); }
      };
      speechSynthesis.getVoices(); // 触发加载
    });
  }

  function pickJapaneseVoice(voices) {
    const list = voices || speechSynthesis.getVoices() || [];
    // 1) 精确 ja-JP 且名称匹配偏好
    const p1 = list.find(v => v.lang === "ja-JP" && JAPANESE_PREF_NAMES.some(k => (v.name||"").includes(k)));
    if (p1) return p1;
    // 2) 仅按 ja-JP
    const p2 = list.find(v => v.lang === "ja-JP");
    if (p2) return p2;
    // 3) 语言前缀 ja-*
    const p3 = list.find(v => (v.lang||"").toLowerCase().startsWith("ja"));
    if (p3) return p3;
    // 4) 兜底
    return list[0] || null;
  }

  // ——对外函数：专读【读音】（片假名/平假名混合 OK）——
  async function speakPronunciationJa(text) {
    if (!("speechSynthesis" in window)) return;
    const msg = String(text ?? "").trim();
    if (!msg) return;

    // 清理叠音
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel();

    // 预处理 & 分句
    const prepared = toJaFriendlyText(msg);
    const chunks = splitJaChunks(prepared);

    const voices = await waitVoicesReady();
    const voice = pickJapaneseVoice(voices);

    // 逐句播报，保证停顿自然
    let idx = 0;
    const speakNext = () => {
      if (idx >= chunks.length) return;
      const u = new SpeechSynthesisUtterance(chunks[idx++]);
      if (voice) u.voice = voice;
      u.lang   = voice?.lang || "ja-JP";
      u.rate   = JA_RATE;
      u.pitch  = JA_PITCH;
      u.volume = JA_VOLUME;
      u.onerror = e => console.warn("[pron-ja] error:", e.error || e.name || e);
      u.onend = () => speakNext();
      speechSynthesis.speak(u);
    };
    speakNext();
  }

  // 暴露到全局，供你在“朗读【读音】”按钮或流程里调用
  window.speakPronunciationJa = speakPronunciationJa;

  // ——（可选）如果你有【读音】的 DOM，可直接挂按钮——
  // 给读音容器加 data-pron-text="true"，给按钮加 id="btnPronTTS"
  const btn = document.querySelector("#btnPronTTS");
  const pronEl = document.querySelector('[data-pron-text="true"]');
  if (btn && pronEl) {
    btn.addEventListener("click", () => {
      const text = ("value" in pronEl) ? pronEl.value : (pronEl.innerText || pronEl.textContent || "");
      speakPronunciationJa(text);
    });
  }
})();
</script>
<script>
(() => {
  // ——可按需改的参数——
  const IOS_JA_PREF = ["Kyoko","Otoya","Siri"]; // iOS 常见更自然日语
  const IOS_ZH_PREF = ["Ting","Siri"];          // iOS 常见中文（Ting-Ting 等）
  const AND_JA_PREF = ["Google 日本語"];        // Android Google TTS
  const AND_ZH_PREF = ["Google 普通话","Google Mandarin","Google 中国"];
  const JA_RATE = 1.02, ZH_RATE = 1.0; // 语速建议
  const PITCH = 1.0, VOL = 1.0;

  const isIOS = /iP(hone|ad|od)/i.test(navigator.userAgent);

  // ——文本预处理（专为日语读音）——
  function toJaFriendlyText(raw) {
    let t = (raw || "").normalize("NFKC");
    t = t.replace(/[,，]/g,"、")
         .replace(/[.．。]{1,}/g,"。")
         .replace(/[!！]{1,}/g,"！")
         .replace(/[?？]{1,}/g,"？")
         .replace(/[:：]/g,"、").replace(/[;；]/g,"、")
         .replace(/\s+/g," ").replace(/ /g,"、")
         .replace(/・/g,"・").replace(/ｰ/g,"ー")
         .replace(/、{2,}/g,"、").replace(/。{2,}/g,"。");
    if (!/[。！？]$/.test(t)) t += "。";
    return t;
  }
  function splitChunks(t, hardLen=80) {
    const first = t.split(/(?<=[。！？])/);
    const out = [];
    for (const seg of first) {
      const s = seg.trim();
      if (!s) continue;
      if (s.length > hardLen && s.includes("、")) {
        const parts = s.split("、");
        parts.forEach((p,i)=>{ const q=p.trim(); if(q) out.push(i<parts.length-1?q+"、":q); });
      } else out.push(s);
    }
    return out;
  }

  function waitVoicesReady(timeoutMs=2000) {
    return new Promise(resolve=>{
      const now = speechSynthesis.getVoices();
      if (now && now.length) return resolve(now);
      let done=false;
      const timer=setTimeout(()=>{ if(!done){done=true;resolve(speechSynthesis.getVoices());}}, timeoutMs);
      speechSynthesis.onvoiceschanged = ()=>{ if(!done){done=true;clearTimeout(timer);resolve(speechSynthesis.getVoices());}};
      speechSynthesis.getVoices();
    });
  }

  function pickVoice(lang, voices, prefs=[]) {
    const list = voices || speechSynthesis.getVoices() || [];
    // 精确 lang + 名称偏好
    const p1 = list.find(v => v.lang === lang && prefs.some(k => (v.name||"").includes(k)));
    if (p1) return p1;
    // 精确 lang
    const p2 = list.find(v => v.lang === lang);
    if (p2) return p2;
    // 前缀 lang
    const pre = lang.split("-")[0];
    const p3 = list.find(v => (v.lang||"").toLowerCase().startsWith(pre));
    return p3 || list[0] || null;
  }

  function detectLang(text) {
    const t = (text||"").trim();
    if (!t) return "zh-CN";
    const hasJa = /[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]/.test(t);
    if (hasJa) return "ja-JP";
    const han = (t.match(/[\u4E00-\u9FFF]/g)||[]).length;
    return han >= Math.max(2, t.length*0.1) ? "zh-CN" : "zh-CN";
  }

  async function speakChunks(chunks, lang, rate) {
    if (!chunks.length) return;
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel();

    const voices = await waitVoicesReady();
    // 平台定制化的“偏好名称”
    const prefs = (lang==="ja-JP")
      ? (isIOS ? IOS_JA_PREF : AND_JA_PREF)
      : (isIOS ? IOS_ZH_PREF : AND_ZH_PREF);
    const v = pickVoice(lang, voices, prefs);

    let i=0;
    const play = () => {
      if (i >= chunks.length) return;
      const u = new SpeechSynthesisUtterance(chunks[i++]);
      if (v) u.voice = v;
      u.lang = v?.lang || lang;
      u.rate = rate;
      u.pitch = PITCH;
      u.volume = VOL;
      // iOS 某些版本需要在 onend 里串行，否则丢句
      u.onend = () => play();
      u.onerror = e => console.warn("[mobile-tts]", e.error || e.name || e);
      speechSynthesis.speak(u);
    };
    play();
  }

  // ——导出两个函数——
  // 1) 专读【读音】：固定日语、预处理、分句
  async function speakPronunciationJaMobile(text){
    const pre = toJaFriendlyText(text||"");
    const chunks = splitChunks(pre, 60);  // 移动端更短更稳
    await speakChunks(chunks, "ja-JP", JA_RATE);
  }
  // 2) 一般文本：自动判中/日
  async function speakSmartMobile(text){
    const t = (text||"").trim(); if(!t) return;
    const lang = detectLang(t);
    if (lang==="ja-JP"){
      const pre = toJaFriendlyText(t);
      return speakChunks(splitChunks(pre, 80), "ja-JP", JA_RATE);
    } else {
      // 中文这边不做重写，直接分句
      const chunks = t.replace(/[,，]/g,"，").replace(/[.．。]{1,}/g,"。")
                      .replace(/[!！]{1,}/g,"！").replace(/[?？]{1,}/g,"？")
                      .split(/(?<=[。！？])/).map(s=>s.trim()).filter(Boolean);
      return speakChunks(chunks, "zh-CN", ZH_RATE);
    }
  }

  // 暴露
  window.speakPronunciationJaMobile = speakPronunciationJaMobile;
  window.speakSmartMobile = speakSmartMobile;
})();
</script>
<script>
/** ===== iOS-friendly 日语读音朗读（片假名OK） ===== */
(() => {
  const isIOS = /iP(hone|ad|od)/i.test(navigator.userAgent);

  // 推荐先在 iOS: 设置→辅助功能→朗读内容→声音→日本語 下载 "Kyoko/Otoya" 或 Siri 日语
  const IOS_JA_PREF = /Kyoko|Otoya|Siri/i; // 按名称优先

  // ——把“读音”文本规范化为更利于日语 TTS 的形式——
  function normalizeJaPron(text) {
    if (!text) return "";
    let t = String(text).normalize("NFKC");      // 全角化（含半角片假名→全角）
    t = t.replace(/[,，]/g,"、")
         .replace(/[.．。]{1,}/g,"。")
         .replace(/[!！]{1,}/g,"！")
         .replace(/[?？]{1,}/g,"？")
         .replace(/[:：]/g,"、").replace(/[;；]/g,"、")
         .replace(/･/g,"・").replace(/ｰ/g,"ー")
         .replace(/\s+/g," ")                   // 连续空白
         .replace(/ /g,"、")                    // 空格→小停顿
         .replace(/、{2,}/g,"、")
         .replace(/。{2,}/g,"。");
    if (!/[。！？]$/.test(t)) t += "。";
    return t;
  }

  // ——按句号/感叹号/问号切块；过长句再按顿号细分——
  function splitJaChunks(t, hardLen = isIOS ? 60 : 80) {
    const first = t.split(/(?<=[。！？])/);
    const out = [];
    for (const seg of first) {
      const s = seg.trim();
      if (!s) continue;
      if (s.length > hardLen && s.includes("、")) {
        const parts = s.split("、");
        parts.forEach((p,i)=>{ const q=p.trim(); if(q) out.push(i<parts.length-1?q+"、":q); });
      } else out.push(s);
    }
    return out;
  }

  // ——等待系统 voices 可用（iOS 首次常为空）——
  function waitVoicesReady(timeoutMs = 2500) {
    return new Promise(resolve => {
      const now = speechSynthesis.getVoices();
      if (now && now.length) return resolve(now);
      let done = false;
      const timer = setTimeout(() => { if (!done){ done = true; resolve(speechSynthesis.getVoices()); }}, timeoutMs);
      window.speechSynthesis.onvoiceschanged = () => {
        if (!done){ done = true; clearTimeout(timer); resolve(speechSynthesis.getVoices()); }
      };
      speechSynthesis.getVoices(); // 触发加载
    });
  }

  // ——挑选最合适的日语 voice（优先 Kyoko/Otoya/Siri）——
  function pickJaVoice(voices) {
    const list = voices || speechSynthesis.getVoices() || [];
    // 1) ja-JP 且名字匹配偏好
    let v = list.find(v => v.lang === "ja-JP" && IOS_JA_PREF.test(v.name || ""));
    if (v) return v;
    // 2) 纯 ja-JP
    v = list.find(v => v.lang === "ja-JP");
    if (v) return v;
    // 3) 语言前缀 ja-
    v = list.find(v => (v.lang || "").toLowerCase().startsWith("ja"));
    return v || list[0] || null;
  }

  // ——核心：专读【读音】——
  async function speakPronunciationJaIOS(pronText, opts = {}) {
    if (!("speechSynthesis" in window)) return;
    const raw = (pronText ?? "").trim();
    if (!raw) return;

    // iOS 需要用户手势后调用；多次点击避免叠音
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel();

    const prepared = normalizeJaPron(raw);
    const chunks = splitJaChunks(prepared);

    const voices = await waitVoicesReady();
    const voice = pickJaVoice(voices);

    const rate = opts.rate ?? 1.02; // 稍慢清晰
    const pitch = opts.pitch ?? 1.0;
    const vol  = opts.volume ?? 1.0;

    let i = 0;
    const play = () => {
      if (i >= chunks.length) return;
      const u = new SpeechSynthesisUtterance(chunks[i++]);
      if (voice) u.voice = voice;
      u.lang   = "ja-JP";
      u.rate   = rate;
      u.pitch  = pitch;
      u.volume = vol;
      u.onerror = e => console.warn("[ios-ja-pron]", e.error || e.name || e);
      u.onend   = () => play();   // 串行播放，iOS 更稳
      speechSynthesis.speak(u);
    };
    play();
  }

  // 暴露到全局，供你在【读音】按钮或流程里直接调用
  window.speakPronunciationJaIOS = speakPronunciationJaIOS;
})();
</script>
<script>
// ===== Disable Local TTS (hard kill switch) =====
(() => {
  // 1) 标记：全局开关（有需要可在未来再打开）
  window.USE_LOCAL_TTS = false;

  // 2) 把所有可能的朗读函数置为 no-op（不报错、不发声）
  const noop = (..._) => { /* TTS disabled */ };
  const fns = [
    "speakSmart", "speakSmartMobile",
    "speakPronunciationJa", "speakPronunciationJaMobile", "speakPronunciationJaIOS",
    "speakPronunciationJaAuto", "speakSmartAuto"
  ];
  fns.forEach(name => { try { window[name] = noop; } catch {} });

  // 3) 禁用/隐藏相关按钮（按你之前的ID/数据属性来）
  const sels = [
    "#btnTTS", "#btnPronTTS",    // 你页面里加过的朗读按钮
    "[data-tts-btn]", "[data-pron-tts-btn]"
  ];
  sels.forEach(sel => {
    document.querySelectorAll(sel).forEach(el => {
      el.disabled = true;
      el.style.pointerEvents = "none";
      el.style.opacity = "0.4";
      el.title = "朗读功能已关闭";
      // 如需直接不显示，改为：el.style.display = "none";
    });
  });

  // 4) 如果你在代码里有“自动朗读”的调用，做保护性短路
  // 只要未来误调用，也不会有任何副作用：
  window.maybeSpeakSmart = noop;
  window.maybeSpeakPronJa = noop;
})();
</script>
<script>
// ===== Hard Disable Local TTS (no PWA) =====
(() => {
  // 关总开关
  window.USE_LOCAL_TTS = false;

  // 1) 把你页面里可能存在的朗读函数全部置空（不报错，不发声）
  const noop = () => {};
  [
    "speakSmart","speakSmartMobile",
    "speakPronunciationJa","speakPronunciationJaMobile","speakPronunciationJaIOS",
    "speakPronunciationJaAuto","speakSmartAuto",
    "maybeSpeakSmart","maybeSpeakPronJa"
  ].forEach(n => { try { window[n] = noop; } catch {} });

  // 2) 从根上拦：将 speechSynthesis 也“静音”
  try {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel?.();
      window.speechSynthesis.speak = noop;
      window.speechSynthesis.cancel = noop;
    }
  } catch {}

  // 3) 移除朗读按钮的绑定并禁用/隐藏
  const killBtn = (el) => {
    if (!el) return;
    const clone = el.cloneNode(true);     // 克隆替换，清掉所有事件监听
    el.replaceWith(clone);
    clone.disabled = true;
    clone.style.pointerEvents = "none";
    clone.style.opacity = "0.4";
    clone.style.display = "none";         // 想保留位置就注释掉这行
    clone.title = "朗读功能已关闭";
  };
  const selectors = [
    "#btnTTS", "#btnPronTTS",
    "[data-tts-btn]", "[data-pron-tts-btn]",
    "button[title*='朗读']", "button[title*='読み上げ']"
  ];
  selectors.forEach(sel => document.querySelectorAll(sel).forEach(killBtn));

  // 4) 兜底：捕获点击，若仍有“朗读/発音/TTS”文案的按钮，阻止默认行为
  window.addEventListener("click", (e) => {
    const txt = (e.target?.textContent || "").trim();
    if (/朗读|読み上げ|発音|TTS/i.test(txt)) {
      e.stopImmediatePropagation();
      e.preventDefault();
      console.warn("[TTS disabled] blocked click on:", txt);
    }
  }, true);

  console.log("[TTS disabled] local TTS is fully disabled.");
})();
</script>

<script>
// ===== Hard Disable Local TTS (no PWA) =====
(() => {
  // 关总开关
  window.USE_LOCAL_TTS = false;

  // 1) 把你页面里可能存在的朗读函数全部置空（不报错，不发声）
  const noop = () => {};
  [
    "speakSmart","speakSmartMobile",
    "speakPronunciationJa","speakPronunciationJaMobile","speakPronunciationJaIOS",
    "speakPronunciationJaAuto","speakSmartAuto",
    "maybeSpeakSmart","maybeSpeakPronJa"
  ].forEach(n => { try { window[n] = noop; } catch {} });

  // 2) 从根上拦：将 speechSynthesis 也“静音”
  try {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel?.();
      window.speechSynthesis.speak = noop;
      window.speechSynthesis.cancel = noop;
    }
  } catch {}

  // 3) 移除朗读按钮的绑定并禁用/隐藏
  const killBtn = (el) => {
    if (!el) return;
    const clone = el.cloneNode(true);     // 克隆替换，清掉所有事件监听
    el.replaceWith(clone);
    clone.disabled = true;
    clone.style.pointerEvents = "none";
    clone.style.opacity = "0.4";
    clone.style.display = "none";         // 想保留位置就注释掉这行
    clone.title = "朗读功能已关闭";
  };
  const selectors = [
    "#btnTTS", "#btnPronTTS",
    "[data-tts-btn]", "[data-pron-tts-btn]",
    "button[title*='朗读']", "button[title*='読み上げ']"
  ];
  selectors.forEach(sel => document.querySelectorAll(sel).forEach(killBtn));

  // 4) 兜底：捕获点击，若仍有“朗读/発音/TTS”文案的按钮，阻止默认行为
  window.addEventListener("click", (e) => {
    const txt = (e.target?.textContent || "").trim();
    if (/朗读|読み上げ|発音|TTS/i.test(txt)) {
      e.stopImmediatePropagation();
      e.preventDefault();
      console.warn("[TTS disabled] blocked click on:", txt);
    }
  }, true);

  console.log("[TTS disabled] local TTS is fully disabled.");
})();
</script>





    </body>
    </html>
    """


@app.get("/playground", response_class=HTMLResponse)
def playground():
    # 默认网页入口（PC & 手机通用，手机优先设计）
    return render_playground_html()


@app.get("/m", response_class=HTMLResponse)
@app.get("/mobile", response_class=HTMLResponse)
def playground_mobile():
    # 短链接 /m /mobile，便于在 IG / TikTok 简介中使用
    return render_playground_html()


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
        "kansai",
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
class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"  # 可选：用同一个默认声音


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
    if mode == "kansai":
        return (
            "你是「ことの葉関西ことば」，教用户在理解标准日语的基础上，"
            "安全、有趣地接触关西地区（日语）口音和表达（以大阪周边为主）。\n"
            "原则：\n"
            "1. 先给【标准日语版本】，再给【关西说法】，不只给方言，避免听不懂。\n"
            "2. 说明哪些适合朋友之间・关西本地日常，哪些不适合对上司、客户或正式场合。\n"
            "3. 不强化刻板印象，不教攻击性或过度粗鲁表达。\n"
            "输出结构：\n"
            "【标准日语】一句自然说法。\n"
            "【关西版本】对应的关西ことば说法。\n"
            "【读音（平假名）】以关西版本为主，标注读音。\n"
            "【中文解释】说明语气差异、适用场景，提醒使用边界。\n"
            "适合作为“通过关西腔增加听感与趣味”的进阶学习入口。"
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
@app.post("/tts")
async def tts(req: TTSRequest):
    if not (LLM_API_KEY and LLM_TTS_MODEL):
        raise HTTPException(status_code=500, detail="TTS not configured")

    try:
        audio_response = client.audio.speech.create(
            model=LLM_TTS_MODEL,
            voice=req.voice,
            input=req.text,
        )
        audio_bytes = (
            audio_response.read()
            if hasattr(audio_response, "read")
            else audio_response
        )
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            # 专门给额度不足的友好提示
            raise HTTPException(
                status_code=402,
                detail="当前语音额度不足，暂时只能使用文字学习功能。如需开启朗读功能，请为 API 充值或更换有额度的密钥。"
            )
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


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

    elif mode == "kansai":
        user_message = (
            "请用下面结构教我一个例子，通过对比标准日语和关西ことば来学习：\n"
            "【标准日语】\n"
            "【关西版本】\n"
            "【读音（平假名）】（以关西版本为主）\n"
            "【中文解释】说明差异、语气、适合对谁说，提醒不要在正式场合乱用。\n"
            "请使用自然、真实但不过分粗鲁的关西表达，只用虚构/一般场景，不针对真实个人。\n"
            f"我的具体内容是：{req.message}"
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
