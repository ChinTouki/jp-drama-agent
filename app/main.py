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
(() => {
  // === 配置 ===
  const TARGET_TEXTAREA_ID = "inputText"; // <- 改成你的输入框 id
  const CONTINUOUS = false;               // 每次说完自动停止；想持续听写可改 true
  const INTERIM = true;                   // 显示临时结果
  const APPEND_MODE = false;              // true=追加；false=覆盖

  const $btn = document.getElementById("btnMic");
  const $status = document.getElementById("micStatus");
  const $ta = document.getElementById(TARGET_TEXTAREA_ID);
  const $sel = document.getElementById("srLang");

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    if ($btn) { $btn.disabled = true; $btn.title = "此浏览器不支持语音输入"; }
    if ($status) $status.textContent = "音声非対応";
    return;
  }

  let recognition = null;
  let listening = false;
  let finalBuffer = "";

  function newRecognizer() {
    const rec = new SR();
    rec.lang = ($sel && $sel.value) ? $sel.value : "zh-CN"; // 默认中文
    rec.continuous = CONTINUOUS;
    rec.interimResults = INTERIM;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      listening = true;
      setPressed(true);
      setStatus(rec.lang.startsWith("zh") ? "聆听中…（再次点击停止）" : "傾聴中…（もう一度クリックで停止）");
    };

    rec.onend = () => {
      setPressed(false);
      listening = false;
      if (finalBuffer.trim()) {
        // 可选：自动补句号（中文/日文简单处理）
        const text = autoPunct(finalBuffer.trim(), rec.lang);
        writeToTextarea(text);
        finalBuffer = "";
        setStatus(rec.lang.startsWith("zh") ? "识别结束" : "認識終了");
      } else {
        setStatus(rec.lang.startsWith("zh") ? "待机中" : "待機中");
      }
    };

    rec.onresult = (ev) => {
      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        const text = res[0] && res[0].transcript ? res[0].transcript : "";
        if (res.isFinal) finalBuffer += text;
        else if (INTERIM) interim += text;
      }
      if (INTERIM && interim) setStatus((rec.lang.startsWith("zh") ? "临时: " : "暫定: ") + sanitize(interim));
    };

    rec.onerror = (e) => {
      setStatus("错误/エラー: " + e.error);
      setPressed(false);
      listening = false;
    };

    return rec;
  }

  function ensureInstance() {
    if (!recognition) recognition = newRecognizer();
  }

  // 语言切换时，重建识别器（若正在听，先停）
  if ($sel) {
    $sel.addEventListener("change", () => {
      if (listening && recognition) try { recognition.stop(); } catch {}
      recognition = newRecognizer();
      setStatus($sel.value.startsWith("zh") ? "已切换到中文" : "日本語に切替えました");
    });
  }

  function setPressed(on) {
    if ($btn) {
      $btn.setAttribute("aria-pressed", String(on));
      $btn.style.background = on ? "rgba(0,128,255,.08)" : "";
      $btn.style.borderColor = on ? "dodgerblue" : "";
    }
  }
  function setStatus(msg) { if ($status) $status.textContent = msg; }
  function sanitize(s) { return s.replace(/\s+/g, " ").trim(); }

  function autoPunct(text, lang) {
    // 简单规则：末尾无终止符则补一个（中文「。！？」，日文同理）
    const end = text.slice(-1);
    const enders = lang.startsWith("zh") ? "。！？!?" : "。！？!?";
    if (!enders.includes(end)) return text + (lang.startsWith("zh") ? "。" : "。");
    return text;
  }

  function writeToTextarea(text) {
    if (!$ta) return;
    if (APPEND_MODE) {
      const sep = $ta.value && !/\s$/.test($ta.value) ? " " : "";
      $ta.value = $ta.value + sep + text;
    } else {
      $ta.value = text;
    }
    // 触发你现有的监听逻辑
    $ta.focus();
    $ta.setSelectionRange($ta.value.length, $ta.value.length);
    $ta.dispatchEvent(new Event("input", { bubbles: true }));
    $ta.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async function toggle() {
    ensureInstance();
    if (!recognition) return;
    try {
      if (!listening) {
        finalBuffer = "";
        recognition.start();
      } else {
        recognition.stop();
      }
    } catch (err) {
      setStatus("无法开始/開始できません: " + (err.message || err.name || "unknown"));
      setPressed(false);
      listening = false;
    }
  }

  if ($btn) $btn.addEventListener("click", toggle);
  // 快捷键：Ctrl/Cmd + M
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === "m" || e.key === "M")) {
      e.preventDefault();
      toggle();
    }
  });
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
