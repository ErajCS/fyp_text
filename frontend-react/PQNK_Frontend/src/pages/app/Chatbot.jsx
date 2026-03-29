import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";

// ── Icons ─────────────────────────────────────────────────────────────────────
const IconSend = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);
const IconBot = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
    strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
    <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" />
    <line x1="12" y1="3" x2="12" y2="7" /><circle cx="8.5" cy="16" r="1" /><circle cx="15.5" cy="16" r="1" />
    <path d="M9 20h6" />
  </svg>
);
const IconUser = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
    strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
  </svg>
);
const IconCopy = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);
const IconCheck = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);
const IconPlus = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);
const IconVolume = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
    <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
  </svg>
);
const IconMic = ({ active }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <rect x="9" y="2" width="6" height="12" rx="3" fill={active ? "currentColor" : "none"} />
    <path d="M5 10a7 7 0 0 0 14 0" />
    <line x1="12" y1="17" x2="12" y2="22" />
    <line x1="8" y1="22" x2="16" y2="22" />
  </svg>
);
const IconThumbsUp = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3z" />
    <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
  </svg>
);
const IconThumbsDown = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3z" />
    <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
  </svg>
);

// ── Quick Topic Icons (from wireframe) ────────────────────────────────────────
const QUICK_TOPICS = [
  { icon: "🌿", label: "Crop Problem" },
  { icon: "💧", label: "When to Water?" },
  { icon: "🧪", label: "Fertilizer Info" },
  { icon: "🌦️", label: "Weather Forecast" },
  { icon: "🐛", label: "Pest Control" },
];

const SUGGESTED_PROMPTS = [
  { label: "Wheat Irrigation Schedule", text: "What is the best irrigation schedule for wheat crops in arid climates?" },
  { label: "Soil Fertility Tips", text: "How can I improve soil fertility naturally without chemical fertilizers?" },
  { label: "Pest Control Methods", text: "What are organic pest control methods for cotton crops?" },
  { label: "Crop Yield Optimization", text: "How can I optimize crop yield for rice in monsoon season?" },
  { label: "Seedling Care Guide", text: "What are the best practices for seedling care in nursery settings?" },
];

const RECENT_CHATS = [
  { id: 1, title: "Wheat Irrigation Advice", preview: "Discussed optimal water scheduling…", time: "2h ago" },
  { id: 2, title: "Soil Nitrogen Deficiency", preview: "NPK levels and organic amendments…", time: "Yesterday" },
  { id: 3, title: "Cotton Pest Outbreak", preview: "Identified bollworm infestation…", time: "3d ago" },
  { id: 4, title: "Maize Fertilizer Plan", preview: "Seasonal nutrition strategy…", time: "1w ago" },
];

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function useCopyToClipboard() {
  const [copiedId, setCopiedId] = useState(null);
  const copy = (text, id) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };
  return { copiedId, copy };
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-green-500"
          style={{ animation: `typingDot 1.2s ease-in-out ${i * 0.2}s infinite`, display: "inline-block" }}
        />
      ))}
    </div>
  );
}

// ── Audio Button ──────────────────────────────────────────────────────────────
function AudioButton({ text }) {
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);

  const handlePlay = async () => {
    if (loading || playing) return;
    setLoading(true);
    try {
      const res = await fetch("/generate_audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (data.audio_url) {
        const audio = new Audio(data.audio_url);
        setPlaying(true);
        audio.play().catch(console.error);
        audio.onended = () => setPlaying(false);
      }
    } catch (err) {
      console.error("Audio error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handlePlay}
      title="Listen"
      className={`w-7 h-7 rounded-full flex items-center justify-center transition-all shadow-sm ${loading ? "bg-amber-500/30 text-amber-300 animate-pulse"
        : playing ? "bg-green-500/30 text-green-300"
          : "bg-emerald-700/60 hover:bg-emerald-600/80 text-white/70 hover:text-white"
        }`}
    >
      <IconVolume />
    </button>
  );
}

// ── Markdown Renderer ─────────────────────────────────────────────────────────
function MarkdownMessage({ content }) {
  // Split into lines and render each with formatting
  const lines = content.split("\n");

  const renderInline = (text) => {
    // Bold: **text** or __text__
    const parts = text.split(/(\*\*[^*]+\*\*|__[^_]+__)/g);
    return parts.map((part, i) => {
      if (/^\*\*[^*]+\*\*$/.test(part) || /^__[^_]+__$/.test(part)) {
        const inner = part.replace(/^\*\*|\*\*$|^__|__$/g, "");
        return <strong key={i} className="font-semibold text-green-200">{inner}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  const rendered = [];
  let i = 0;
  let inList = false;
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      rendered.push(
        <ul key={`ul-${i}`} className="space-y-1 my-1.5 pl-1">
          {listItems.map((item, j) => (
            <li key={j} className="flex gap-2 items-start">
              <span className="text-green-400 mt-0.5 flex-shrink-0">•</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Empty line
    if (!trimmed) {
      flushList();
      rendered.push(<div key={`br-${i}`} className="h-1.5" />);
      i++;
      continue;
    }

    // H1 / H2
    if (/^#{1,2}\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^#{1,2}\s/, "");
      rendered.push(
        <p key={`h2-${i}`} className="font-bold text-green-300 text-sm mt-2 mb-0.5">
          {renderInline(text)}
        </p>
      );
      i++;
      continue;
    }

    // H3 — Sources / section heading
    if (/^###\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^###\s/, "");
      rendered.push(
        <div key={`h3-${i}`} className="mt-3">
          <div className="border-t border-white/10 mb-2" />
          <p className="font-semibold text-emerald-400 text-xs uppercase tracking-wide">
            {renderInline(text)}
          </p>
        </div>
      );
      i++;
      continue;
    }

    // Numbered list  e.g. "1. item"
    if (/^\d+\.\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^\d+\.\s/, "");
      // collect consecutive numbered items
      const numItems = [text];
      while (i + 1 < lines.length && /^\d+\.\s/.test(lines[i + 1].trim())) {
        i++;
        numItems.push(lines[i].trim().replace(/^\d+\.\s/, ""));
      }
      rendered.push(
        <ol key={`ol-${i}`} className="space-y-1 my-1.5 pl-1">
          {numItems.map((item, j) => (
            <li key={j} className="flex gap-2 items-start">
              <span className="text-emerald-400 font-semibold text-xs mt-0.5 flex-shrink-0">{j + 1}.</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      i++;
      continue;
    }

    // Bullet: "- " or "* "
    if (/^[-*]\s/.test(trimmed)) {
      inList = true;
      listItems.push(trimmed.replace(/^[-*]\s/, ""));
      i++;
      continue;
    }

    // Normal paragraph
    flushList();
    rendered.push(
      <p key={`p-${i}`} className="leading-relaxed">{renderInline(trimmed)}</p>
    );
    i++;
  }
  flushList();

  return <div className="space-y-0.5 text-sm" dir="auto">{rendered}</div>;
}

// ── Message Bubble ────────────────────────────────────────────────────────────
function MessageBubble({ msg, index, copiedId, onCopy }) {
  const isUser = msg.role === "user";
  return (
    <div
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} items-end`}
      style={{ animation: "chatFadeIn 0.35s ease-out both" }}
    >
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-md ${isUser
          ? "bg-gradient-to-br from-amber-400 to-yellow-500 text-white"
          : "bg-gradient-to-br from-green-500 to-emerald-600 text-white"
          }`}
      >
        {isUser ? <IconUser /> : <IconBot />}
      </div>

      <div className={`flex flex-col max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`relative px-5 py-3 rounded-2xl shadow-lg leading-relaxed text-sm ${isUser
            ? "bg-gradient-to-br from-amber-400 to-yellow-500 text-white rounded-br-none"
            : "bg-white/15 backdrop-blur-md text-white border border-white/15 rounded-bl-none"
            }`}
        >
          {isUser ? (
            <p className="text-sm leading-relaxed" dir="auto">{msg.content}</p>
          ) : (
            <MarkdownMessage content={msg.content} />
          )}
        </div>

        <div className={`flex items-center gap-2 mt-1 ${isUser ? "flex-row-reverse" : ""}`}>
          <span className="text-white/40 text-xs">{formatTime(msg.timestamp)}</span>
          {!isUser && (
            <div className="flex gap-1">
              <AudioButton text={msg.content} />
              <button onClick={() => onCopy(msg.content, index)} className="text-white/30 hover:text-white transition-colors" title="Copy">
                {copiedId === index ? <IconCheck /> : <IconCopy />}
              </button>
              <button className="text-white/30 hover:text-green-400 transition-colors"><IconThumbsUp /></button>
              <button className="text-white/30 hover:text-red-400 transition-colors"><IconThumbsDown /></button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
// ── Speech Recognition Hook ──────────────────────────────────────────────────
const SPEECH_LANGS = [
  { code: "en-US", label: "EN", name: "English" },
  { code: "ur-PK", label: "اردو", name: "Urdu" },
];

function useSpeechInput({ onResult, onError }) {
  const recognitionRef = useRef(null);
  const [listening, setListening] = useState(false);
  const [langIndex, setLangIndex] = useState(0);
  const supported = typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  const startListening = () => {
    if (!supported) { onError("Voice input is not supported in this browser. Please use Google Chrome."); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = SPEECH_LANGS[langIndex].code;
    recognition.maxAlternatives = 1;
    recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      onResult(transcript);
    };
    recognition.onerror = (e) => {
      console.error("Speech error:", e.error);
      if (e.error !== "no-speech") onError("Voice input error: " + e.error);
      setListening(false);
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
    setListening(false);
  };

  const toggleLang = () => setLangIndex((i) => (i + 1) % SPEECH_LANGS.length);

  return { listening, supported, startListening, stopListening, langIndex, toggleLang, langInfo: SPEECH_LANGS[langIndex] };
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Chatbot() {
  const navigate = useNavigate();
  const location = useLocation();
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! 🌾 I'm your AgriChat AI Assistant, powered by PQNK agricultural intelligence. I can help you with crop management, irrigation planning, soil health, pest control, fertilizer recommendations, and more. What would you like to explore today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [activeChat, setActiveChat] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const { copiedId, copy } = useCopyToClipboard();

  const sendMessage = async (text) => {
    const content = (text ?? input).trim();
    if (!content) return;

    setMessages((prev) => [...prev, { role: "user", content, timestamp: new Date() }]);
    setInput("");
    setTyping(true);

    try {
      const res = await fetch("/get_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ msg: content }),
      });
      if (!res.ok) throw new Error("Server responded with status " + res.status);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response || "No response received.", timestamp: new Date() },
      ]);
    } catch (error) {
      console.error("Chat API error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Oops! I encountered an error. Please ensure the backend is running.", timestamp: new Date() },
      ]);
    } finally {
      setTyping(false);
    }
  };

  const handleNewChat = () => {
    setMessages([{
      role: "assistant",
      content: "Hello again! 🌱 Starting a fresh session. What agricultural topic can I help you with today?",
      timestamp: new Date(),
    }]);
    setActiveChat(null);
    setInput("");
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  const charLimit = 500;
  const charCount = input.length;
  const hasUserMessages = messages.filter((m) => m.role === "user").length > 0;
  const [voiceError, setVoiceError] = useState("");

  const { listening, supported: speechSupported, startListening, stopListening, langIndex, toggleLang, langInfo } = useSpeechInput({
    onResult: (transcript) => {
      setInput((prev) => (prev ? prev + " " + transcript : transcript));
      setVoiceError("");
      // Focus the textarea
      setTimeout(() => inputRef.current?.focus(), 50);
    },
    onError: (msg) => {
      setVoiceError(msg);
      setTimeout(() => setVoiceError(""), 5000);
    },
  });

  return (
    <>
      <style>{`
        @keyframes chatFadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes typingDot {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30%            { transform: translateY(-6px); opacity: 1; }
        }
        @keyframes pulseGlow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
          50%       { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
        }
        .status-dot { animation: pulseGlow 2s infinite; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 2px; }
      `}</style>

      <div
        className="flex h-screen font-sans overflow-hidden relative"
        style={{
          backgroundImage: "url('/agri-bg.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-green-950/60 via-emerald-900/50 to-green-950/70 z-0" />

        {/* ── SIDEBAR — dark emerald ── */}
        <aside className="w-[260px] bg-gradient-to-b from-emerald-900 to-green-950 backdrop-blur-xl shadow-2xl border-r border-emerald-700/30 flex flex-col hidden md:flex z-10 relative">

          {/* Logo — AgriChat */}
          <div className="p-5 border-b border-white/10">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-lg shadow-lg shadow-green-500/20">
                🌿
              </div>
              <div>
                <h1 className="text-white font-bold text-base leading-none">AgriChat</h1>
                <p className="text-emerald-400/70 text-[10px] mt-0.5 tracking-wider uppercase">AI Assistant</p>
              </div>
            </div>
          </div>

          {/* New Chat */}
          <div className="p-4">
            <button
              onClick={handleNewChat}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 text-white py-2.5 px-4 rounded-xl transition-all duration-200 text-sm font-medium shadow-lg hover:shadow-green-500/25"
            >
              <IconPlus /> New Conversation
            </button>
          </div>

          {/* Quick Topics — wireframe inspired */}
          <div className="px-4 pb-3">
            <p className="text-white/40 text-xs uppercase tracking-widest mb-2 font-semibold">Quick Topics</p>
            <div className="space-y-1">
              {QUICK_TOPICS.map((t, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(`Tell me about ${t.label.toLowerCase()}`)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-white/10 transition-colors text-left text-white/60 hover:text-white text-xs"
                >
                  <span className="text-base">{t.icon}</span>
                  <span className="truncate">{t.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="mx-4 border-t border-white/10 my-2" />

          {/* Recent Chats */}
          <div className="px-4 flex-1 overflow-y-auto">
            <p className="text-white/40 text-xs uppercase tracking-widest mb-2 font-semibold">Recent Conversations</p>
            <div className="space-y-1">
              {RECENT_CHATS.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => setActiveChat(chat.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl transition-all duration-200 ${activeChat === chat.id
                    ? "bg-white/15 border border-white/10"
                    : "hover:bg-white/10"
                    }`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-white/80 text-xs font-medium truncate pr-2">{chat.title}</span>
                    <span className="text-white/30 text-[10px] flex-shrink-0">{chat.time}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* User Info */}
          <div className="p-4 border-t border-white/10">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-emerald-950 text-xs font-bold shadow">
                MW
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white text-xs font-medium truncate">Muhammad Wajeeh</p>
                <p className="text-emerald-400/60 text-[11px]">Agriculture Researcher</p>
              </div>
              <div className="w-2 h-2 rounded-full bg-green-400 status-dot flex-shrink-0" />
            </div>
          </div>
        </aside>

        {/* ── MAIN CHAT AREA ── */}
        <div className="flex-1 flex flex-col min-w-0 relative z-10">

          {/* Header */}
          <header className="flex items-center justify-between px-6 py-3 bg-emerald-950/70 backdrop-blur-xl border-b border-white/10 flex-shrink-0">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center shadow-lg shadow-green-500/20 text-white">
                <IconBot />
              </div>
              <div>
                <h2 className="text-white font-semibold text-base leading-none">AgriChat AI Assistant</h2>
                <div className="flex items-center gap-2 mt-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
                  <span className="text-emerald-400 text-xs">Online · Powered by PQNK IntelliAgri</span>
                </div>
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-1">
              {[
                { label: "Home", path: "/dashboard" },
                { label: "About Us", path: "/about" },
                { label: "Resources", path: "/browse-repository" },
                { label: "Contact Us", path: "/contact" },
              ].map((link) => (
                <button
                  key={link.path}
                  onClick={() => navigate(link.path)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${location.pathname === link.path
                    ? "bg-white/15 text-white"
                    : "text-white/50 hover:text-white hover:bg-white/10"
                    }`}
                >
                  {link.label}
                </button>
              ))}
              <button className="ml-2 px-2.5 py-1 rounded-lg bg-white/10 border border-white/15 text-white/70 text-xs font-medium hover:bg-white/20 transition">
                اردو
              </button>
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">

            {/* Welcome banner — "Welcome to AgriChat" (wireframe) */}
            {!hasUserMessages && (
              <div className="bg-white/10 backdrop-blur-lg border border-white/15 rounded-2xl p-6 mb-2 shadow-lg" style={{ animation: "chatFadeIn 0.5s ease-out" }}>
                <h3 className="text-white font-bold text-xl mb-2">Welcome to AgriChat</h3>
                <p className="text-white/60 text-sm leading-relaxed">
                  I'm trained on extensive agricultural datasets covering crops, soil science, irrigation, climate adaptation, pest management, and more. Ask me anything — from basic farming to advanced agronomic planning.
                </p>
                <div className="flex flex-wrap gap-2 mt-4">
                  {["Pakistan Crops", "Smart Irrigation", "Organic Farming", "Climate Advisory"].map(tag => (
                    <span key={tag} className="bg-emerald-800/50 border border-emerald-500/20 text-emerald-300 text-xs px-3 py-1 rounded-full">{tag}</span>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, index) => (
              <MessageBubble key={index} msg={msg} index={index} copiedId={copiedId} onCopy={copy} />
            ))}

            {typing && (
              <div className="flex items-end gap-3" style={{ animation: "chatFadeIn 0.3s ease-out" }}>
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 text-white">
                  <IconBot />
                </div>
                <div className="bg-white/10 border border-white/15 rounded-2xl rounded-bl-none shadow-lg">
                  <TypingDots />
                </div>
                <span className="text-white/40 text-xs mb-1">AgriChat is thinking…</span>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Quick Topic Icons — wireframe-inspired bottom strip */}
          {!hasUserMessages && (
            <div className="px-6 pb-3">
              <p className="text-white/70 text-xs mb-2 uppercase tracking-widest drop-shadow">Quick Topics</p>
              <div className="flex gap-3">
                {QUICK_TOPICS.map((t, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(`Tell me about ${t.label.toLowerCase()}`)}
                    className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl bg-white/10 backdrop-blur-lg border border-white/15 shadow-lg hover:bg-white/20 hover:scale-[1.03] transition-all"
                  >
                    <span className="text-xl">{t.icon}</span>
                    <span className="text-[11px] text-white/70 font-medium whitespace-nowrap">{t.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Suggested Prompts */}
          {!hasUserMessages && (
            <div className="px-6 pb-3">
              <p className="text-white/70 text-xs mb-2 uppercase tracking-widest drop-shadow">Suggested Questions</p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_PROMPTS.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(p.text)}
                    className="bg-white/10 hover:bg-white/20 border border-white/15 hover:border-emerald-400/30 text-white/70 hover:text-white text-xs px-3 py-2 rounded-full transition-all duration-200"
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input bar */}
          <div className="px-6 pb-5 pt-3 bg-emerald-950/50 backdrop-blur-xl border-t border-white/10 flex-shrink-0 relative z-10">
            <div className="relative flex items-end gap-3 bg-white/10 border border-white/15 hover:border-emerald-400/40 focus-within:border-emerald-400/60 focus-within:ring-2 focus-within:ring-green-400/20 rounded-2xl px-4 py-3 transition-all duration-200 shadow-lg">
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={(e) => {
                  if (e.target.value.length <= charLimit) setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
                placeholder="Type your message…"
                className="flex-1 bg-transparent text-white placeholder-white/40 resize-none focus:outline-none text-sm leading-relaxed"
                style={{ minHeight: "24px", maxHeight: "120px" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                    e.target.style.height = "auto";
                  }
                }}
              />
              <div className="flex items-center gap-2 flex-shrink-0 self-end mb-0.5">
                {/* Voice Language Toggle */}
                <button
                  type="button"
                  onClick={toggleLang}
                  title={`Switch to ${SPEECH_LANGS[(langIndex + 1) % SPEECH_LANGS.length].name}`}
                  className="px-2 py-1 rounded-lg bg-white/10 border border-white/15 text-white/60 hover:text-white hover:bg-white/20 text-[11px] font-semibold transition-all"
                >
                  {langInfo.label}
                </button>
                {/* Mic Button */}
                <button
                  type="button"
                  onClick={listening ? stopListening : startListening}
                  title={!speechSupported ? "Voice input requires Google Chrome" : listening ? "Stop recording" : `Record in ${langInfo.name}`}
                  className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200 shadow-lg relative ${!speechSupported
                      ? "bg-white/5 text-white/20 cursor-not-allowed"
                      : listening
                        ? "bg-red-500/80 text-white border-2 border-red-400 scale-105"
                        : "bg-white/10 border border-white/15 text-white/60 hover:bg-emerald-700/60 hover:text-white hover:border-emerald-400/40"
                    }`}
                >
                  {listening && (
                    <span className="absolute inset-0 rounded-xl bg-red-500/30 animate-ping" />
                  )}
                  <IconMic active={listening} />
                </button>
                {charCount > 0 && (
                  <span className={`text-xs ${charCount > charLimit * 0.9 ? "text-amber-400" : "text-white/40"}`}>
                    {charCount}/{charLimit}
                  </span>
                )}
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || typing}
                  className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200 shadow-lg ${input.trim() && !typing
                    ? "bg-gradient-to-br from-green-400 to-emerald-500 hover:from-green-300 hover:to-emerald-400 text-white hover:scale-105"
                    : "bg-white/10 text-white/30 cursor-not-allowed"
                    }`}
                >
                  <IconSend />
                </button>
              </div>
            </div>
            {/* Voice status bar */}
            {(listening || voiceError) && (
              <div className={`flex items-center justify-center gap-2 mt-2 text-xs rounded-lg px-3 py-1.5 ${voiceError
                  ? "bg-red-500/15 border border-red-400/20 text-red-300"
                  : "bg-emerald-800/40 border border-emerald-500/20 text-emerald-300"
                }`}>
                {listening && (
                  <span className="flex gap-0.5 items-end h-3">
                    {[1, 2, 3, 4, 3].map((h, i) => (
                      <span key={i} className="w-0.5 bg-emerald-400 rounded-full"
                        style={{ height: `${h * 3}px`, animation: `typingDot 0.8s ease-in-out ${i * 0.12}s infinite` }} />
                    ))}
                  </span>
                )}
                {voiceError ? `⚠️ ${voiceError}` : `🎙️ Listening in ${langInfo.name}… speak now`}
                {listening && <button onClick={stopListening} className="ml-2 text-red-300 hover:text-red-200 font-semibold">Stop</button>}
              </div>
            )}
            {!listening && !voiceError && (
              <p className="text-white/30 text-xs text-center mt-2">
                Press <kbd className="bg-white/10 px-1.5 py-0.5 rounded text-white/40 font-mono text-[10px]">Enter</kbd> to send ·
                <kbd className="bg-white/10 px-1.5 py-0.5 rounded text-white/40 font-mono text-[10px]"> Shift+Enter</kbd> new line ·
                <span className="text-white/20 ml-1">🎙️ mic for voice</span>
              </p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
