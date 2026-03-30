import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useLanguage, TRANSLATIONS } from "../../context/LanguageContext";

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
const IconMic = ({ active, recording }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
    <rect x="9" y="2" width="6" height="12" rx="3" fill={recording ? "currentColor" : "none"} />
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
const IconTrash = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" className="w-3.5 h-3.5">
    <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" />
    <path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" />
  </svg>
);


const QUICK_TOPICS = [
  { id: "crop", icon: "🌿", label: "Crop Problem" },
  { id: "water", icon: "💧", label: "When to Water?" },
  { id: "fert", icon: "🧪", label: "Fertilizer Info" },
  { id: "pest", icon: "🐛", label: "Pest Control" },
];

const SUGGESTED_PROMPTS = [
  { id: "wheat", label: "Wheat Irrigation" },
  { id: "soil", label: "Soil Health" },
  { id: "pest_cot", label: "Pest Control" },
  { id: "rice", label: "Rice Yield" },
  { id: "nursery", label: "Seedling Care" },
];

function formatTime(date) {
  return new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function formatRelative(dateStr) {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return d.toLocaleDateString();
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

// ── Audio Button ───────────────────────────────────────────────────────────────
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
    } catch (err) { console.error("Audio error:", err); }
    finally { setLoading(false); }
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

// ── Markdown Renderer ──────────────────────────────────────────────────────────
function MarkdownMessage({ content }) {
  const lines = content.split("\n");
  const renderInline = (text) => {
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
    if (!trimmed) { flushList(); rendered.push(<div key={`br-${i}`} className="h-1.5" />); i++; continue; }
    if (/^#{1,2}\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^#{1,2}\s/, "");
      rendered.push(<p key={`h2-${i}`} className="font-bold text-green-300 text-sm mt-2 mb-0.5">{renderInline(text)}</p>);
      i++; continue;
    }
    if (/^###\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^###\s/, "");
      rendered.push(
        <div key={`h3-${i}`} className="mt-3">
          <div className="border-t border-white/10 mb-2" />
          <p className="font-semibold text-emerald-400 text-xs uppercase tracking-wide">{renderInline(text)}</p>
        </div>
      );
      i++; continue;
    }
    if (/^\d+\.\s/.test(trimmed)) {
      flushList();
      const text = trimmed.replace(/^\d+\.\s/, "");
      const numItems = [text];
      while (i + 1 < lines.length && /^\d+\.\s/.test(lines[i + 1].trim())) {
        i++; numItems.push(lines[i].trim().replace(/^\d+\.\s/, ""));
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
      i++; continue;
    }
    if (/^[-*]\s/.test(trimmed)) {
      inList = true;
      listItems.push(trimmed.replace(/^[-*]\s/, ""));
      i++; continue;
    }
    flushList();
    rendered.push(<p key={`p-${i}`} className="leading-relaxed">{renderInline(trimmed)}</p>);
    i++;
  }
  flushList();
  return <div className="space-y-0.5 text-sm" dir="auto">{rendered}</div>;
}

// ── Message Bubble ─────────────────────────────────────────────────────────────
function MessageBubble({ msg, index, copiedId, onCopy, isStreaming }) {
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
            <>
              <MarkdownMessage content={msg.content} />
              {isStreaming && <span className="inline-block w-0.5 h-4 bg-green-400 animate-pulse ml-0.5 align-middle" />}
            </>
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

// ── Whisper Voice Hook (replaces Web Speech API) ─────────────────────────────
function useWhisperInput({ onResult, onError, onTranscribing, lang = "en" }) {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setTranscribing(true);
        onTranscribing?.();
        try {
          const formData = new FormData();
          formData.append("audio", blob, "audio.webm");
          // Send language hint so Whisper transcribes in the correct language
          // 'ur' for Urdu to prevent accidental Hindi output
          formData.append("language", lang === "ur" ? "ur" : "en");
          const res = await fetch("/api/transcribe", {
            method: "POST",
            credentials: "include",
            body: formData,
          });
          const data = await res.json();
          if (data.transcript) onResult(data.transcript);
          else onError(data.error || "No transcript returned");
        } catch (e) {
          onError("Transcription failed: " + e.message);
        } finally {
          setTranscribing(false);
        }
      };
      mediaRecorderRef.current = mr;
      mr.start();
      setRecording(true);
    } catch (e) {
      onError("Microphone access denied. Please allow microphone permissions.");
    }
  }, [onResult, onError, onTranscribing]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }, []);

  const toggle = useCallback(() => {
    if (recording) stopRecording();
    else startRecording();
  }, [recording, startRecording, stopRecording]);

  return { recording, transcribing, toggle };
}

// ── Language Toggle Button ─────────────────────────────────────────────────────
// ── Main Component ─────────────────────────────────────────────────────────────
export default function Chatbot() {
  const navigate = useNavigate();
  const location = useLocation();

  // ── Language (from global context — shared across all pages) ─────────────
  const { lang, toggleLang } = useLanguage();
  const isRtl = lang === "ur";
  const t = (key) => TRANSLATIONS[lang][key] || TRANSLATIONS.en[key];

  // ── Messages ──────────────────────────────────────────────────────────────
  const WELCOME_MSG = {
    role: "assistant",
    content: "Hello! 🌾 I'm your AgriChat AI Assistant, powered by PQNK agricultural intelligence. I can help you with crop management, irrigation planning, soil health, pest control, fertilizer recommendations, and more. What would you like to explore today?",
    timestamp: new Date(),
  };
  const [messages, setMessages] = useState([WELCOME_MSG]);
  const [streamingIndex, setStreamingIndex] = useState(null);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);

  // ── Conversations ─────────────────────────────────────────────────────────
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const { copiedId, copy } = useCopyToClipboard();

  const charLimit = 500;
  const hasUserMessages = messages.some((m) => m.role === "user");

  // ── Fetch conversation list ───────────────────────────────────────────────
  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/chat/history", { credentials: "include" });
      if (res.ok) setConversations(await res.json());
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  // ── Load a past conversation ──────────────────────────────────────────────
  const loadConversation = async (convId) => {
    setLoadingHistory(true);
    setActiveConvId(convId);
    try {
      const res = await fetch(`/api/chat/${convId}/messages`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to load");
      const msgs = await res.json();
      setMessages(msgs.map((m) => ({ ...m, timestamp: new Date(m.timestamp) })));
    } catch {
      setMessages([{ role: "assistant", content: "⚠️ Could not load this conversation.", timestamp: new Date() }]);
    } finally {
      setLoadingHistory(false);
    }
  };

  // ── New Chat ──────────────────────────────────────────────────────────────
  const handleNewChat = useCallback(async () => {
    try {
      const res = await fetch("/api/chat/new", { method: "POST", credentials: "include" });
      const data = await res.json();
      setActiveConvId(data.id);
      setMessages([WELCOME_MSG]);
      setInput("");
      fetchHistory();
    } catch {
      setActiveConvId(null);
      setMessages([WELCOME_MSG]);
    }
  }, [fetchHistory]);

  // ── Delete conversation ───────────────────────────────────────────────────
  const deleteConversation = async (e, convId) => {
    e.stopPropagation();
    await fetch(`/api/chat/${convId}`, { method: "DELETE", credentials: "include" });
    if (activeConvId === convId) { setActiveConvId(null); setMessages([WELCOME_MSG]); }
    fetchHistory();
  };

  // ── Send message with streaming ────────────────────────────────────────────
  const sendMessage = useCallback(async (text) => {
    const content = (text ?? input).trim();
    if (!content || typing) return;

    const userMsg = { role: "user", content, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTyping(true);
    if (inputRef.current) inputRef.current.style.height = "auto";

    // Create a placeholder for the AI answer
    const aiPlaceholder = { role: "assistant", content: "", timestamp: new Date() };
    setMessages((prev) => [...prev, aiPlaceholder]);
    const aiIndex = messages.length + 1; // index of the streaming bubble
    setStreamingIndex(aiIndex);

    try {
      const res = await fetch("/get_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ msg: content, conversation_id: activeConvId }),
      });

      if (!res.ok) throw new Error("Status " + res.status);
      const data = await res.json();
      const fullText = data.response || "No response received.";

      // Simulate streaming by revealing text word-by-word
      const words = fullText.split(" ");
      let currentText = "";
      for (let w = 0; w < words.length; w++) {
        currentText += (w === 0 ? "" : " ") + words[w];
        const captured = currentText;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: captured, timestamp: new Date() };
          return updated;
        });
        // Faster for longer texts to keep speed reasonable
        if (w < words.length - 1) await new Promise((r) => setTimeout(r, Math.max(12, 35 - words.length * 0.2)));
      }

      // Update conversation tracking
      if (data.conversation_id) {
        setActiveConvId(data.conversation_id);
        fetchHistory();
      }
    } catch (error) {
      console.error("Chat API error:", error);
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "⚠️ I encountered an error. Please ensure the backend is running.",
          timestamp: new Date(),
        };
        return updated;
      });
    } finally {
      setTyping(false);
      setStreamingIndex(null);
    }
  }, [input, typing, activeConvId, messages.length, fetchHistory]);

  // ── Whisper voice input ───────────────────────────────────────────────────
  const [voiceStatus, setVoiceStatus] = useState(""); // "recording" | "transcribing" | ""
  const { recording, transcribing, toggle: toggleMic } = useWhisperInput({
    lang,   // 'ur' or 'en' — sent to Whisper backend to force correct transcription language
    onResult: (text) => {
      setInput((prev) => (prev ? prev + " " + text : text));
      setVoiceStatus("");
      setTimeout(() => inputRef.current?.focus(), 50);
    },
    onError: (msg) => {
      setVoiceStatus("error:" + msg);
      setTimeout(() => setVoiceStatus(""), 5000);
    },
    onTranscribing: () => setVoiceStatus("transcribing"),
  });
  useEffect(() => { if (recording) setVoiceStatus("recording"); }, [recording]);

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
        dir={lang === "ur" ? "rtl" : "ltr"}
        style={{
          backgroundImage: "url('/agri-bg.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-green-950/60 via-emerald-900/50 to-green-950/70 z-0" />

        {/* ── SIDEBAR ── */}
        <aside className="w-[260px] bg-gradient-to-b from-emerald-900 to-green-950 backdrop-blur-xl shadow-2xl border-r border-emerald-700/30 flex flex-col hidden md:flex z-10 relative">

          {/* Logo */}
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
              <IconPlus /> {t("newConversation")}
            </button>
          </div>

          {/* Quick Topics */}
          <div className="px-4 pb-3">
            <p className="text-white/40 text-xs uppercase tracking-widest mb-2 font-semibold">{t("quickTopics")}</p>
            <div className="space-y-1">
              {QUICK_TOPICS.map((topic, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(`Tell me about ${topic.label.toLowerCase()}`)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-white/10 transition-colors text-left text-white/60 hover:text-white text-xs"
                >
                  <span className="text-base">{topic.icon}</span>
                  <span className="truncate">{lang === "ur" ? topic.urdu : topic.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="mx-4 border-t border-white/10 my-2" />

          {/* Conversations list */}
          <div className="px-4 flex-1 overflow-y-auto">
            <p className="text-white/40 text-xs uppercase tracking-widest mb-2 font-semibold">{t("recentConversations")}</p>
            {conversations.length === 0 ? (
              <p className="text-white/20 text-xs text-center py-4">{t("noConversations")}</p>
            ) : (
              <div className="space-y-1">
                {conversations.map((conv) => (
                  <div key={conv.id} className="group relative">
                    <button
                      onClick={() => loadConversation(conv.id)}
                      className={`w-full text-left px-3 py-2.5 rounded-xl transition-all duration-200 pr-8 ${activeConvId === conv.id
                        ? "bg-white/15 border border-white/10"
                        : "hover:bg-white/10"
                        }`}
                    >
                      <div className="flex justify-between items-start">
                        <span className="text-white/80 text-xs font-medium truncate pr-2">{conv.title}</span>
                        <span className="text-white/30 text-[10px] flex-shrink-0">{formatRelative(conv.updated_at)}</span>
                      </div>
                    </button>
                    {/* Delete button */}
                    <button
                      onClick={(e) => deleteConversation(e, conv.id)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 w-6 h-6 rounded-lg bg-red-500/0 group-hover:bg-red-500/20 hover:!bg-red-500/40 text-white/0 group-hover:text-red-300 flex items-center justify-center transition-all duration-150"
                      title="Delete conversation"
                    >
                      <IconTrash />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* User Info */}
          <div className="p-4 border-t border-white/10">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-emerald-950 text-xs font-bold shadow">
                🌾
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white text-xs font-medium truncate">PQNK User</p>
                <p className="text-emerald-400/60 text-[11px]">Agriculture Intelligence</p>
              </div>
              <div className="w-2 h-2 rounded-full bg-green-400 status-dot flex-shrink-0" />
            </div>
          </div>
        </aside>

        {/* ── MAIN CHAT AREA ── */}
        <div className={`flex-1 flex flex-col min-w-0 relative z-10 ${isRtl ? "font-urdu" : ""}`} dir={isRtl ? "rtl" : "ltr"}>
          {/* Header */}
          <header className={`flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5 backdrop-blur-md sticky top-0 z-20 ${isRtl ? "flex-row-reverse" : ""}`}>
            <div className={`flex items-center gap-3 ${isRtl ? "flex-row-reverse" : ""}`}>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-white shadow-lg shadow-green-500/20">
                <IconBot />
              </div>
              <div className={isRtl ? "text-right" : ""}>
                <h2 className="text-white font-semibold text-base leading-none">AgriChat AI Assistant</h2>
                <div className={`flex items-center gap-2 mt-1 ${isRtl ? "flex-row-reverse" : ""}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
                  <span className="text-emerald-400 text-xs">{t("online")}</span>
                </div>
              </div>
            </div>

            <div className={`hidden sm:flex items-center gap-1 ${isRtl ? "flex-row-reverse" : ""}`}>
              {[
                { labelKey: "home", path: "/dashboard" },
                { labelKey: "about", path: "/about" },
                { labelKey: "resources", path: "/browse-repository" },
                { labelKey: "contact", path: "/contact" },
              ].map((link) => (
                <button
                  key={link.path}
                  onClick={() => navigate(link.path)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${location.pathname === link.path
                    ? "bg-white/15 text-white"
                    : "text-white/50 hover:text-white hover:bg-white/10"
                    }`}
                >
                  {t(link.labelKey)}
                </button>
              ))}
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
            {/* Welcome banner */}
            {!hasUserMessages && !loadingHistory && (
              <div className="bg-white/10 backdrop-blur-lg border border-white/15 rounded-2xl p-6 mb-2 shadow-lg" style={{ animation: "chatFadeIn 0.5s ease-out" }}>
                <h3 className="text-white font-bold text-xl mb-2">{t("welcomeTitle")}</h3>
                <p className="text-white/60 text-sm leading-relaxed">{t("welcomeBody")}</p>
                <div className="flex flex-wrap gap-2 mt-4">
                  {["Pakistan Crops", "Smart Irrigation", "Organic Farming", "Climate Advisory"].map((tag) => (
                    <span key={tag} className="bg-emerald-800/50 border border-emerald-500/20 text-emerald-300 text-xs px-3 py-1 rounded-full">{tag}</span>
                  ))}
                </div>
              </div>
            )}

            {loadingHistory && (
              <div className="flex justify-center py-8">
                <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              </div>
            )}

            {messages.map((msg, index) => (
              <MessageBubble
                key={index}
                msg={msg}
                index={index}
                copiedId={copiedId}
                onCopy={copy}
                isStreaming={typing && index === messages.length - 1 && msg.role === "assistant"}
              />
            ))}

            {typing && messages[messages.length - 1]?.content === "" && (
              <div className="flex items-end gap-3" style={{ animation: "chatFadeIn 0.3s ease-out" }}>
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 text-white">
                  <IconBot />
                </div>
                <div className="bg-white/10 border border-white/15 rounded-2xl rounded-bl-none shadow-lg">
                  <TypingDots />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Quick Topics & Suggested Prompts */}
          {!hasUserMessages && !loadingHistory && (
            <>
              <div className="px-6 pb-3">
                <div className="flex gap-3 flex-wrap">
                  {QUICK_TOPICS.map((topic, i) => (
                    <button
                      key={topic.id}
                      onClick={() => sendMessage(t(`query${topic.id.charAt(0).toUpperCase() + topic.id.slice(1)}`))}
                      className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl bg-white/10 backdrop-blur-lg border border-white/15 shadow-lg hover:bg-white/20 hover:scale-[1.03] transition-all"
                    >
                      <span className="text-xl">{topic.icon}</span>
                      <span className="text-[11px] text-white/70 font-medium whitespace-nowrap">
                        {t(`topic${topic.id.charAt(0).toUpperCase() + topic.id.slice(1)}`)}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="px-6 pb-3">
                <p className="text-white/70 text-xs mb-2 uppercase tracking-widest drop-shadow">{t("suggestedQ")}</p>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTED_PROMPTS.map((p, i) => {
                    const promptKey = `prompt${p.id.charAt(0).toUpperCase() + p.id.slice(1)}`;
                    const queryKey = `query${p.id.charAt(0).toUpperCase() + p.id.slice(1).replace("_cot", "Cot")}`;
                    return (
                      <button
                        key={p.id}
                        onClick={() => sendMessage(t(queryKey))}
                        className={`bg-white/10 hover:bg-white/20 border border-white/15 hover:border-emerald-400/30 text-white/70 hover:text-white text-xs px-3 py-2 rounded-full transition-all duration-200 ${isRtl ? "font-urdu" : ""}`}
                      >
                        {t(promptKey)}
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* ── Input bar ── */}
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
                placeholder={t("placeholder")}
                className="flex-1 bg-transparent text-white placeholder-white/40 resize-none focus:outline-none text-sm leading-relaxed"
                style={{ minHeight: "24px", maxHeight: "120px" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />

              <div className="flex items-center gap-2 flex-shrink-0 self-end mb-0.5">
                {/* Whisper mic button */}
                <button
                  type="button"
                  onClick={toggleMic}
                  disabled={transcribing}
                  title={recording ? "Tap to stop recording" : transcribing ? "Transcribing…" : "Voice input (English & Urdu)"}
                  className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200 shadow-lg relative ${transcribing
                    ? "bg-amber-500/30 text-amber-300 animate-pulse cursor-wait"
                    : recording
                      ? "bg-red-500/80 text-white border-2 border-red-400 scale-105"
                      : "bg-white/10 border border-white/15 text-white/60 hover:bg-emerald-700/60 hover:text-white hover:border-emerald-400/40"
                    }`}
                >
                  {recording && <span className="absolute inset-0 rounded-xl bg-red-500/30 animate-ping" />}
                  <IconMic active recording={recording} />
                </button>

                {input.length > 0 && (
                  <span className={`text-xs ${input.length > charLimit * 0.9 ? "text-amber-400" : "text-white/40"}`}>
                    {input.length}/{charLimit}
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
            {(recording || transcribing || voiceStatus.startsWith("error:")) && (
              <div className={`flex items-center justify-center gap-2 mt-2 text-xs rounded-lg px-3 py-1.5 ${voiceStatus.startsWith("error:")
                ? "bg-red-500/15 border border-red-400/20 text-red-300"
                : transcribing
                  ? "bg-amber-500/15 border border-amber-400/20 text-amber-300"
                  : "bg-emerald-800/40 border border-emerald-500/20 text-emerald-300"
                }`}>
                {recording && (
                  <span className="flex gap-0.5 items-end h-3">
                    {[1, 2, 3, 4, 3].map((h, i) => (
                      <span key={i} className="w-0.5 bg-emerald-400 rounded-full"
                        style={{ height: `${h * 3}px`, animation: `typingDot 0.8s ease-in-out ${i * 0.12}s infinite` }} />
                    ))}
                  </span>
                )}
                {voiceStatus.startsWith("error:")
                  ? `⚠️ ${voiceStatus.slice(6)}`
                  : transcribing
                    ? `⏳ ${t("transcribing")}`
                    : `🎙️ ${t("recording")}`
                }
              </div>
            )}
            {!recording && !transcribing && !voiceStatus && (
              <p className="text-white/30 text-xs text-center mt-2">
                {t("enterHint")} ·{" "}
                <span className="text-white/20">🎙️ mic for English & Urdu voice</span>
              </p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
