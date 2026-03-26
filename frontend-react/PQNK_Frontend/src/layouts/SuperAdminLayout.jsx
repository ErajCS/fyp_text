import { useNavigate, useLocation } from "react-router-dom";

const navItems = [
    { label: "Command Center", path: "/super-admin-dashboard", icon: "⚡" },
    { label: "Content Management", path: "/super-admin-dashboard/content", icon: "📝" },
    { label: "User Management", path: "/super-admin-dashboard/users", icon: "👥" },
    { label: "Admin Management", path: "/super-admin-dashboard/admins", icon: "🛡️" },
    { label: "System Config", path: "/super-admin-dashboard/config", icon: "⚙️" },
    { label: "Audit Logs", path: "/super-admin-dashboard/audit", icon: "📋" },
    { label: "API Keys", path: "/super-admin-dashboard/api-keys", icon: "🔑" },
    { label: "Analytics", path: "/super-admin-dashboard/analytics", icon: "📈" },
    { label: "AI Assistant", path: "/chatbot", icon: "🤖" },
];

export default function SuperAdminLayout({ children }) {
    const navigate = useNavigate();
    const location = useLocation();

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

            {/* ── SIDEBAR — dark emerald ── */}
            <div className="w-[270px] bg-gradient-to-b from-emerald-900 to-green-950 backdrop-blur-xl shadow-2xl z-10 flex flex-col border-r border-emerald-700/30 relative">

                <div className="px-6 pt-6 pb-5 border-b border-white/10">
                    <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-lg shadow-lg shadow-amber-400/20">
                            ⚡
                        </div>
                        <div>
                            <h2 className="text-lg font-bold leading-none tracking-wide text-white">AgriChat</h2>
                            <p className="text-[10px] text-amber-400/80 font-bold tracking-[0.2em] uppercase mt-0.5">Super Admin</p>
                        </div>
                    </div>
                </div>

                <nav className="flex-1 px-3 py-4 space-y-1 text-sm font-medium overflow-y-auto">
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
                                <span className="truncate">{item.label}</span>
                            </div>
                        );
                    })}
                </nav>

                <div className="px-5 py-4 border-t border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-emerald-950 text-xs font-bold shadow-md">SA</div>
                        <div>
                            <p className="text-sm font-semibold text-white">Super Admin</p>
                            <p className="text-[10px] text-amber-400/60 font-medium">Full System Access</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── MAIN CONTENT ── */}
            <div className="flex-1 flex flex-col min-w-0 relative z-10">

                <div className="flex items-center justify-between px-8 py-3 bg-emerald-950/70 backdrop-blur-xl border-b border-white/10 flex-shrink-0">
                    <div className="flex items-center gap-1">
                        {["Home", "About Us", "Resources", "Contact Us"].map((link, i) => (
                            <button key={i} className="px-3 py-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-all text-xs font-medium">{link}</button>
                        ))}
                    </div>
                    <div className="flex items-center gap-3">
                        <input type="text" placeholder="Search system, admins…" className="w-[240px] px-4 py-2 rounded-xl bg-white/10 border border-white/15 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition" />
                        <div className="relative cursor-pointer">
                            <div className="w-8 h-8 rounded-lg bg-white/10 border border-white/15 flex items-center justify-center hover:bg-white/20 transition text-sm">🔔</div>
                            <div className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-red-500 rounded-full text-[8px] flex items-center justify-center font-bold text-white animate-pulse">3</div>
                        </div>
                        <div className="px-2.5 py-1 rounded-lg bg-green-500/15 border border-green-400/25 flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                            <span className="text-xs text-green-400 font-medium">Online</span>
                        </div>
                        <div className="px-2.5 py-1 rounded-lg bg-amber-400/15 border border-amber-400/25 text-amber-300 text-xs font-semibold">Super Admin</div>
                        <button className="px-2 py-1 rounded-lg bg-white/10 border border-white/15 text-white/70 text-xs font-medium hover:bg-white/20 transition">اردو</button>
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 text-emerald-950 flex items-center justify-center font-bold text-xs shadow-lg shadow-amber-400/20 cursor-pointer">⚡</div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-8">
                    {children}
                </div>
            </div>
        </div>
    );
}
