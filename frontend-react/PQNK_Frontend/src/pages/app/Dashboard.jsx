import DashboardLayout from "../../layouts/DashboardLayout";
import { useNavigate } from "react-router-dom";

// ── PQNK Knowledge Cards ──────────────────────────────────────────────────────
const PQNK_PILLARS = [
  {
    icon: "🌱",
    title: "Natural Soil Rejuvenation",
    desc: "PQNK eliminates synthetic fertilizers by rebuilding soil microbiomes through compost, vermicompost, and natural mulching — restoring long-term fertility without chemical dependency.",
    color: "from-green-500/20 to-emerald-600/10 border-green-400/30",
    tag: "Soil Health",
  },
  {
    icon: "💧",
    title: "Water Conservation",
    desc: "Precision irrigation guided by soil moisture indicators and crop cycles reduces water usage by up to 40% compared to conventional flood irrigation practices.",
    color: "from-blue-500/20 to-cyan-600/10 border-blue-400/30",
    tag: "Irrigation",
  },
  {
    icon: "🐛",
    title: "Biological Pest Control",
    desc: "Neem-based sprays, companion planting, and beneficial insect habitats replace chemical pesticides — protecting crops while maintaining ecological balance.",
    color: "from-amber-500/20 to-yellow-600/10 border-amber-400/30",
    tag: "Pest Control",
  },
  {
    icon: "🌾",
    title: "Seed Sovereignty",
    desc: "PQNK promotes traditional, open-pollinated Pakistani seed varieties — preserving genetic diversity and freeing farmers from corporate seed dependency.",
    color: "from-emerald-500/20 to-green-600/10 border-emerald-400/30",
    tag: "Seeds",
  },
  {
    icon: "🔬",
    title: "Knowledge Integration",
    desc: "Ancient farming wisdom is validated through modern agronomic science, creating a methodology that is both culturally rooted and evidence-based.",
    color: "from-purple-500/20 to-violet-600/10 border-purple-400/30",
    tag: "Research",
  },
  {
    icon: "🤝",
    title: "Community Farming",
    desc: "Collective knowledge-sharing networks among farmers amplify PQNK adoption, creating village-level agricultural communities that support each other.",
    color: "from-rose-500/20 to-pink-600/10 border-rose-400/30",
    tag: "Community",
  },
];

const BENEFITS = [
  { stat: "40%", label: "Reduction in water usage", icon: "💧" },
  { stat: "60%", label: "Lower input costs vs conventional", icon: "💰" },
  { stat: "35%", label: "Improvement in soil organic matter", icon: "🌱" },
  { stat: "0", label: "Synthetic chemicals used", icon: "🧪" },
];

const CROP_CATEGORIES = [
  { name: "Wheat", urdu: "گندم", icon: "🌾", status: "Rabi (Winter)" },
  { name: "Rice", urdu: "چاول", icon: "🍚", status: "Kharif (Summer)" },
  { name: "Cotton", urdu: "کپاس", icon: "🌿", status: "Kharif (Summer)" },
  { name: "Maize", urdu: "مکئی", icon: "🌽", status: "Kharif (Summer)" },
  { name: "Sugarcane", urdu: "گنا", icon: "🎋", status: "Year-round" },
  { name: "Vegetables", urdu: "سبزیاں", icon: "🥬", status: "Seasonal" },
];

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <DashboardLayout>
      <>
        <style>{`
          @keyframes fadeUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
          @keyframes shimmer { 0%,100% { opacity:.7; } 50% { opacity:1; } }
          .fade-up   { animation: fadeUp .6s ease-out both; }
          .fade-up-1 { animation: fadeUp .6s ease-out .1s both; }
          .fade-up-2 { animation: fadeUp .6s ease-out .2s both; }
          .fade-up-3 { animation: fadeUp .6s ease-out .3s both; }
          .fade-up-4 { animation: fadeUp .6s ease-out .4s both; }
          .founder-glow { box-shadow: 0 0 60px rgba(52,211,153,0.25); }
        `}</style>

        {/* ── HERO — PQNK Introduction ── */}
        <div className="fade-up mb-10">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-900/80 to-green-950/90 backdrop-blur-xl border border-emerald-500/20 shadow-2xl p-10">
            {/* Background decoration */}
            <div className="absolute top-0 right-0 w-96 h-96 bg-green-400/5 rounded-full -translate-y-1/2 translate-x-1/4 blur-3xl" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-emerald-500/5 rounded-full translate-y-1/2 -translate-x-1/4 blur-2xl" />

            <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center gap-8">
              <div className="flex-1">
                <div className="inline-flex items-center gap-2 bg-emerald-500/15 border border-emerald-400/20 rounded-full px-4 py-1.5 text-emerald-300 text-xs font-semibold tracking-wider uppercase mb-5">
                  🌿 Welcome to AgriChat
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold text-white leading-tight mb-4">
                  Paedar Qudratti<br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-300">
                    Nizam-e-Kashtari
                  </span>
                </h1>
                <p className="text-white/60 text-lg leading-relaxed max-w-xl mb-6">
                  <strong className="text-white/80">PQNK</strong> — Pakistan's pioneering sustainable natural farming
                  methodology, developed to free smallholder farmers from chemical dependency through
                  science-backed, nature-first agriculture.
                </p>
                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={() => navigate("/chatbot")}
                    className="px-6 py-3 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 text-white font-semibold shadow-lg shadow-green-500/25 hover:scale-[1.02] transition-all text-sm"
                  >
                    🤖 Ask AI Assistant
                  </button>
                  <button
                    onClick={() => navigate("/browse-repository")}
                    className="px-6 py-3 rounded-xl bg-white/10 border border-white/20 text-white font-semibold hover:bg-white/20 transition text-sm"
                  >
                    📚 Browse Repository
                  </button>
                </div>
              </div>

              {/* Urdu name panel */}
              <div className="flex-shrink-0 text-center lg:text-right">
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur">
                  <p className="text-4xl font-bold text-emerald-300 mb-2" dir="rtl">پائیدار قدرتی</p>
                  <p className="text-3xl font-bold text-white/80" dir="rtl">نظامِ کاشتکاری</p>
                  <div className="mt-4 border-t border-white/10 pt-4">
                    <p className="text-xs text-white/40 uppercase tracking-widest">Pakistan Agriculture Research</p>
                    <p className="text-white/60 text-sm mt-1">Est. Dr. Asif Sharif</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── IMPACT STATS ── */}
        <div className="fade-up-1 grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {BENEFITS.map((b, i) => (
            <div
              key={i}
              className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-5 text-center hover:bg-white/15 hover:scale-[1.02] transition-all shadow-lg"
            >
              <div className="text-3xl mb-2">{b.icon}</div>
              <div className="text-3xl font-bold text-white mb-1">{b.stat}</div>
              <p className="text-white/50 text-xs leading-tight">{b.label}</p>
            </div>
          ))}
        </div>

        {/* ── FOUNDER SECTION ── */}
        <div className="fade-up-2 mb-10">
          <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 flex flex-col lg:flex-row gap-8 items-center shadow-xl">
            {/* Avatar */}
            <div className="flex-shrink-0 text-center">
              <div className="founder-glow w-28 h-28 rounded-full bg-gradient-to-br from-emerald-400 to-green-600 flex items-center justify-center text-5xl mx-auto mb-3 border-4 border-emerald-500/30">
                👨‍🌾
              </div>
              <p className="text-white font-bold text-lg">Dr. Asif Sharif</p>
              <p className="text-emerald-400 text-xs uppercase tracking-wider">Founder, PQNK</p>
              <p className="text-white/40 text-xs mt-1">Agronomist · Researcher</p>
            </div>

            {/* Bio */}
            <div className="flex-1">
              <h2 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
                <span>🎓</span> The Visionary Behind PQNK
              </h2>
              <p className="text-white/60 leading-relaxed mb-4 text-sm">
                Dr. Asif Sharif is a Pakistani agronomist whose decades of research led to the
                formulation of PQNK — a complete natural farming system tailored to Pakistan's
                soil types, climate zones, and cultural farming practices.
              </p>
              <p className="text-white/60 leading-relaxed mb-5 text-sm">
                Working in partnership with <strong className="text-white/80">Pakistan Agriculture Research (PAR)</strong>,
                Dr. Sharif has demonstrated PQNK on hundreds of farms across Punjab and Sindh,
                showing that sustainable yields without chemicals are not just possible — they are
                economically superior. His knowledge, previously accessible only through direct
                consultation, is now digitalised and made available to every farmer through this
                platform.
              </p>
              <div className="flex flex-wrap gap-2">
                {["Natural Farming Pioneer", "PAR Collaborator", "PQNK Founder", "Agricultural Researcher"].map(tag => (
                  <span key={tag} className="bg-emerald-700/40 border border-emerald-500/20 text-emerald-300 text-xs px-3 py-1 rounded-full">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── 6 PILLARS OF PQNK ── */}
        <div className="fade-up-3 mb-10">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white drop-shadow">The 6 Pillars of PQNK</h2>
            <span className="text-white/40 text-sm">Natural · Sustainable · Pakistani</span>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {PQNK_PILLARS.map((pillar, i) => (
              <div
                key={i}
                className={`bg-gradient-to-br ${pillar.color} backdrop-blur-xl border rounded-2xl p-6 hover:scale-[1.02] hover:shadow-2xl transition-all duration-300 group`}
                style={{ animationDelay: `${i * 0.07}s` }}
              >
                <div className="flex items-start justify-between mb-3">
                  <span className="text-3xl">{pillar.icon}</span>
                  <span className="text-[10px] text-white/50 bg-white/10 border border-white/10 px-2.5 py-1 rounded-full uppercase tracking-wider font-medium">
                    {pillar.tag}
                  </span>
                </div>
                <h3 className="font-bold text-white text-base mb-2">{pillar.title}</h3>
                <p className="text-white/55 text-sm leading-relaxed">{pillar.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── CROPS SUPPORTED ── */}
        <div className="fade-up-4 mb-10">
          <h2 className="text-2xl font-bold text-white drop-shadow mb-6">Crops in the PQNK System</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {CROP_CATEGORIES.map((crop, i) => (
              <button
                key={i}
                onClick={() => navigate("/chatbot")}
                className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-xl p-4 text-center hover:bg-white/20 hover:scale-[1.03] hover:border-emerald-400/30 transition-all group"
              >
                <div className="text-3xl mb-2">{crop.icon}</div>
                <p className="text-white font-semibold text-sm">{crop.name}</p>
                <p className="text-white/40 text-[10px] mt-0.5 font-medium" dir="rtl">{crop.urdu}</p>
                <p className="text-emerald-400/70 text-[10px] mt-1">{crop.status}</p>
              </button>
            ))}
          </div>
        </div>

        {/* ── CTA ── */}
        <div className="fade-up-4 bg-gradient-to-r from-emerald-700/80 to-green-800/80 border border-emerald-500/30 backdrop-blur-xl rounded-2xl p-8 flex flex-col sm:flex-row items-center justify-between gap-6 shadow-2xl">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Have an agricultural question?</h2>
            <p className="text-white/60 text-sm">
              Our AI assistant is trained on the full PQNK knowledge base — ask anything about crops,
              soil, irrigation, or pest control.
            </p>
          </div>
          <button
            onClick={() => navigate("/chatbot")}
            className="flex-shrink-0 px-8 py-3.5 rounded-xl bg-gradient-to-r from-amber-400 to-yellow-500 text-white font-bold shadow-lg shadow-amber-400/30 hover:scale-105 transition-all text-sm whitespace-nowrap"
          >
            🤖 Open AI Assistant
          </button>
        </div>
      </>
    </DashboardLayout>
  );
}