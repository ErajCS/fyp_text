import { useNavigate, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";

const navItems = [
  { label: "Dashboard", path: "/dashboard", icon: "📊" },
  { label: "Browse Repository", path: "/browse-repository", icon: "📚" },
  { label: "AI Assistant", path: "/chatbot", icon: "🤖" },
  { label: "Profile", path: "/profile", icon: "👤" },
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
          {navItems.map((item) => {
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
                {item.label}
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
            🚪 Sign Out
          </button>
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10">

        {/* Top Bar */}
        <div className="flex items-center justify-between px-8 py-3 bg-emerald-950/70 backdrop-blur-xl border-b border-white/10 flex-shrink-0">

          <div className="flex items-center gap-1">
            {["Home", "About Us", "Resources", "Contact Us"].map((link, i) => (
              <button
                key={i}
                className="px-3 py-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-all text-xs font-medium"
              >
                {link}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <input
              type="text"
              placeholder="Search crops, irrigation, soil…"
              className="w-[240px] px-4 py-2 rounded-xl bg-white/10 border border-white/15 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition"
            />

            <div className="relative cursor-pointer">
              <div className="w-8 h-8 rounded-lg bg-white/10 border border-white/15 flex items-center justify-center hover:bg-white/20 transition text-sm">
                🔔
              </div>
              <div className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-red-500 rounded-full text-[8px] flex items-center justify-center font-bold text-white">3</div>
            </div>

            <button className="px-2 py-1 rounded-lg bg-white/10 border border-white/15 text-white/70 text-xs font-medium hover:bg-white/20 transition">
              اردو
            </button>

            <div
              onClick={() => navigate("/profile")}
              className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 text-emerald-950 flex items-center justify-center font-bold text-xs shadow-lg cursor-pointer hover:scale-105 transition-transform"
            >
              {getInitials(user?.name)}
            </div>
          </div>
        </div>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto p-8">
          {children}
        </div>
      </div>
    </div>
  );
}