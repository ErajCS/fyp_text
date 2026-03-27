import { useNavigate, useLocation } from "react-router-dom";

const navItems = [
    { label: "Dashboard", path: "/admin-dashboard", icon: "📊" },
    { label: "Content Management", path: "/admin-dashboard/content", icon: "📝" },
    { label: "User Management", path: "/admin-dashboard/users", icon: "👥" },
    { label: "Analytics", path: "/admin-dashboard/analytics", icon: "📈" },
    { label: "AI Assistant", path: "/chatbot", icon: "🤖" },
    { label: "Profile", path: "/profile", icon: "👤" },
];

function getInitials(name = "") {
    return name.split(" ").map(n => n[0]).slice(0, 2).join("").toUpperCase() || "A";
}

export default function AdminLayout({ children }) {
    const navigate = useNavigate();
    const location = useLocation();

    // Read real user from localStorage (set by login)
    let user = { name: "Admin", email: "", role: "admin" };
    try { user = JSON.parse(localStorage.getItem("user") || "{}") || user; } catch { }

    const handleLogout = async () => {
        try {
            await fetch("/api/logout", { method: "POST", credentials: "include" });
        } finally {
            localStorage.removeItem("user");
            navigate("/login");
        }
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
            <div className="absolute inset-0 bg-gradient-to-br from-green-950/60 via-emerald-900/50 to-green-950/70 z-0" />

            {/* ── SIDEBAR ── */}
            <div className="w-[260px] bg-gradient-to-b from-emerald-900 to-green-950 backdrop-blur-xl shadow-2xl z-10 flex flex-col border-r border-emerald-700/30 relative">

                <div className="px-6 pt-6 pb-5 border-b border-white/10">
                    <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-lg shadow-lg shadow-green-500/20">
                            🛡️
                        </div>
                        <div>
                            <h2 className="text-lg font-bold leading-none tracking-wide text-white">AgriChat</h2>
                            <p className="text-[10px] text-amber-400/80 font-bold tracking-[0.2em] uppercase mt-0.5">Admin Panel</p>
                        </div>
                    </div>
                </div>

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

                {/* Real user info + logout */}
                <div className="px-5 py-4 border-t border-white/10">
                    <div className="flex items-center gap-3 mb-3">
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-emerald-950 text-xs font-bold shadow-md flex-shrink-0">
                            {getInitials(user.name)}
                        </div>
                        <div className="min-w-0">
                            <p className="text-sm font-semibold text-white truncate">{user.name || "Admin"}</p>
                            <p className="text-[10px] text-amber-400/60 font-medium truncate">{user.email || "Administrator"}</p>
                        </div>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="w-full py-2 rounded-xl bg-red-500/15 border border-red-400/20 text-red-300 text-xs font-semibold hover:bg-red-500/25 transition"
                    >
                        🚪 Sign Out
                    </button>
                </div>
            </div>

            {/* ── MAIN CONTENT ── */}
            <div className="flex-1 flex flex-col min-w-0 relative z-10">
                <div className="flex items-center justify-between px-8 py-3 bg-emerald-950/70 backdrop-blur-xl border-b border-white/10 flex-shrink-0">
                    <div className="flex items-center gap-1">
                        <button onClick={() => navigate("/dashboard")} className="px-3 py-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-all text-xs font-medium">🏠 Home</button>
                        <button onClick={() => navigate("/chatbot")} className="px-3 py-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-all text-xs font-medium">🤖 AI Chat</button>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="px-2.5 py-1 rounded-lg bg-amber-400/15 border border-amber-400/25 text-amber-300 text-xs font-semibold">
                            🛡️ Admin
                        </div>
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 text-emerald-950 flex items-center justify-center font-bold text-xs shadow-lg cursor-pointer"
                            onClick={() => navigate("/profile")}>
                            {getInitials(user.name)}
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-8">
                    {children}
                </div>
            </div>
        </div>
    );
}
