import { useNavigate, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import Footer from "../components/layout/Footer";
import { useLanguage, TRANSLATIONS } from "../context/LanguageContext";

const NAV_ITEMS = [
  { labelKey: "dashboard", path: "/dashboard", icon: "📊" },
  { labelKey: "browseRepo", path: "/browse-repository", icon: "📚" },
  { labelKey: "aiAssistant", path: "/chatbot", icon: "🤖" },
  { labelKey: "profile", path: "/profile", icon: "👤" },
];

function getInitials(name = "") {
  return name.split(" ").map((n) => n[0]).slice(0, 2).join("").toUpperCase() || "?";
}

const ROLE_LABELS = {
  farmer: "Farmer", researcher: "Researcher", student: "Student",
  admin: "Admin", superadmin: "Super Admin", seeker: "User",
};

export default function DashboardLayout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { lang, toggleLang } = useLanguage();
  const t = (key) => TRANSLATIONS[lang][key] || TRANSLATIONS.en[key];
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("user")); } catch { return null; }
  });

  useEffect(() => {
    fetch("/api/user", { credentials: "include" })
      .then((r) => {
        if (r.status === 401) { navigate("/login"); return null; }
        return r.json();
      })
      .then((data) => {
        if (data) {
          setUser(data);
          localStorage.setItem("user", JSON.stringify(data));
        }
      })
      .catch(() => { });
  }, [navigate]);

  const handleLogout = async () => {
    try {
      await fetch("/api/logout", { method: "POST", credentials: "include" });
    } catch { }
    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div
      className="flex min-h-screen relative"
      style={{
        backgroundImage: "url('/agri-bg.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundAttachment: "fixed",
      }}
    >
      {/* Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-green-950/60 via-emerald-900/50 to-green-950/70 z-0" />

      {/* ── SIDEBAR ── */}
      <div className="w-[260px] bg-gradient-to-b from-emerald-900 to-green-950 backdrop-blur-xl shadow-2xl z-10 flex flex-col border-r border-emerald-700/30 relative">

        {/* Logo */}
        <div className="px-6 pt-6 pb-5 border-b border-white/10">
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-lg shadow-lg shadow-green-500/20">
              🌿
            </div>
            <div>
              <h2 className="text-lg font-bold leading-none tracking-wide text-white">AgriChat</h2>
              <p className="text-[10px] text-emerald-400/70 font-medium tracking-widest uppercase mt-0.5">
                PQNK Knowledge System
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-5 space-y-1.5 text-sm font-medium overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <div
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`px-4 py-2.5 rounded-xl cursor-pointer transition-all duration-200 flex items-center gap-3 ${isActive
                  ? "bg-white/15 text-white shadow-lg border border-white/10"
                  : "text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
              >
                <span className="text-base">{item.icon}</span>
                {t(item.labelKey)}
              </div>
            );
          })}
        </nav>

        {/* Footer — real user info + logout */}
        <div className="px-5 py-4 border-t border-white/10">
          <div className="flex items-center gap-3 mb-3">
            <div
              onClick={() => navigate("/profile")}
              className="w-9 h-9 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-emerald-950 text-xs font-bold shadow-md cursor-pointer hover:scale-105 transition-transform"
            >
              {getInitials(user?.name)}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-white truncate">{user?.name || "—"}</p>
              <p className="text-[11px] text-emerald-400/60">{ROLE_LABELS[user?.role] || "User"}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-white/50 hover:bg-red-500/15 hover:text-red-400 transition-all text-xs font-medium border border-white/10 hover:border-red-400/20"
          >
            🚪 {t("signOut")}
          </button>
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10">

        <div className="flex items-center justify-between px-8 py-3 bg-emerald-950/70 backdrop-blur-xl border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-1">
            {[
              { labelKey: "home", path: "/dashboard" },
              { labelKey: "about", path: "/about" },
              { labelKey: "resources", path: "/browse-repository" },
              { labelKey: "contact", path: "/contact" },
            ].map((link) => (
              <button
                key={link.path}
                onClick={() => navigate(link.path)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${location.pathname === link.path
                  ? "bg-white/15 text-white border border-white/15"
                  : "text-white/50 hover:text-white hover:bg-white/10"
                  }`}
              >
                {t(link.labelKey)}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4">
            {/* ── Animated Language Toggle ── */}
            <div className="relative flex items-center bg-white/5 border border-white/10 rounded-full p-1 shadow-inner h-9 overflow-hidden">
              <div
                className="absolute top-1 bottom-1 transition-all duration-500 ease-in-out bg-gradient-to-br from-green-400 to-emerald-600 rounded-full shadow-[0_0_15px_rgba(52,211,153,0.3)] z-0"
                style={{
                  width: "calc(50% - 4px)",
                  left: lang === "en" ? "4px" : "calc(50%)"
                }}
              />
              <button
                type="button"
                onClick={() => lang !== "en" && toggleLang()}
                className={`relative z-10 px-4 py-1.5 text-[10px] font-bold tracking-widest transition-colors duration-300 ${lang === "en" ? "text-white" : "text-white/40 hover:text-white/60"}`}
              >
                ENGLISH
              </button>
              <button
                type="button"
                onClick={() => lang !== "ur" && toggleLang()}
                className={`relative z-10 px-4 py-1.5 text-[10px] font-bold tracking-widest transition-colors duration-300 font-urdu ${lang === "ur" ? "text-white" : "text-white/40 hover:text-white/60"}`}
              >
                اردو
              </button>
            </div>

            <input
              type="text"
              placeholder={t("search")}
              className="w-[200px] px-4 py-2 rounded-xl bg-white/10 border border-white/15 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition"
            />

            <div className="relative cursor-pointer">
              <div className="w-8 h-8 rounded-lg bg-white/10 border border-white/15 flex items-center justify-center hover:bg-white/20 transition text-sm">
                🔔
              </div>
              <div className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-red-500 rounded-full text-[8px] flex items-center justify-center font-bold text-white">3</div>
            </div>

            <div
              onClick={() => navigate("/profile")}
              className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 text-emerald-950 flex items-center justify-center font-bold text-xs shadow-lg cursor-pointer hover:scale-105 transition-transform"
            >
              {getInitials(user?.name)}
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto flex flex-col p-8">
          {children}
          <Footer />
        </div>
      </div>
    </div>
  );
}