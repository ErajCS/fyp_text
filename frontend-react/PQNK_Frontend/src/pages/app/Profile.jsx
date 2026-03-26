import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "../../layouts/DashboardLayout";

const ROLE_LABELS = {
    farmer: { label: "Farmer", icon: "🌾", color: "green" },
    researcher: { label: "Researcher", icon: "🔬", color: "blue" },
    student: { label: "Student", icon: "🎓", color: "purple" },
    admin: { label: "Admin", icon: "🛡️", color: "red" },
    superadmin: { label: "Super Admin", icon: "👑", color: "amber" },
    seeker: { label: "User", icon: "👤", color: "emerald" },
};

function getInitials(name = "") {
    return name
        .split(" ")
        .map((n) => n[0])
        .slice(0, 2)
        .join("")
        .toUpperCase();
}

export default function Profile() {
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [editing, setEditing] = useState(false);
    const [saving, setSaving] = useState(false);
    const [msg, setMsg] = useState({ text: "", type: "" });

    const [form, setForm] = useState({ name: "", phone: "", current_password: "", new_password: "", confirm_new_password: "" });

    useEffect(() => {
        fetch("/api/user", { credentials: "include" })
            .then((r) => {
                if (r.status === 401) { navigate("/login"); return null; }
                return r.json();
            })
            .then((data) => {
                if (data) {
                    setUser(data);
                    setForm((f) => ({ ...f, name: data.name, phone: data.phone || "" }));
                }
            })
            .finally(() => setLoading(false));
    }, [navigate]);

    const handleSave = async (e) => {
        e.preventDefault();
        if (form.new_password && form.new_password !== form.confirm_new_password) {
            setMsg({ text: "New passwords do not match", type: "error" });
            return;
        }
        setSaving(true);
        setMsg({ text: "", type: "" });
        try {
            const res = await fetch("/api/user/update", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({
                    name: form.name,
                    phone: form.phone,
                    current_password: form.current_password || undefined,
                    new_password: form.new_password || undefined,
                }),
            });
            const data = await res.json();
            if (data.success) {
                setUser(data.user);
                setEditing(false);
                setForm((f) => ({ ...f, current_password: "", new_password: "", confirm_new_password: "" }));
                setMsg({ text: "Profile updated successfully!", type: "success" });
                // Update localStorage
                localStorage.setItem("user", JSON.stringify(data.user));
            } else {
                setMsg({ text: data.message || "Update failed", type: "error" });
            }
        } catch {
            setMsg({ text: "Could not connect to server", type: "error" });
        } finally {
            setSaving(false);
        }
    };

    const roleInfo = ROLE_LABELS[user?.role] || ROLE_LABELS.seeker;

    if (loading) {
        return (
            <DashboardLayout>
                <div className="flex items-center justify-center h-64">
                    <div className="text-white/60 text-sm animate-pulse">Loading profile…</div>
                </div>
            </DashboardLayout>
        );
    }

    return (
        <DashboardLayout>
            <>
                <style>{`
          @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
          .fade-up { animation: fadeUp 0.5s ease-out both; }
          .fade-up-d1 { animation: fadeUp 0.5s ease-out 0.1s both; }
          .fade-up-d2 { animation: fadeUp 0.5s ease-out 0.2s both; }
        `}</style>

                {/* Page Header */}
                <div className="mb-8 fade-up">
                    <h1 className="text-3xl font-bold text-white drop-shadow-lg mb-1">My Profile</h1>
                    <p className="text-white/60 text-sm">Manage your account information and credentials</p>
                </div>

                <div className="grid lg:grid-cols-3 gap-6">
                    {/* ── LEFT: Avatar Card ── */}
                    <div className="fade-up-d1 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 flex flex-col items-center text-center shadow-xl">
                        {/* Avatar */}
                        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-3xl font-bold text-emerald-950 shadow-xl mb-4">
                            {getInitials(user?.name)}
                        </div>

                        <h2 className="text-xl font-bold text-white mb-1">{user?.name}</h2>
                        <p className="text-white/50 text-sm mb-4">{user?.email}</p>

                        {/* Role badge */}
                        <span className="inline-flex items-center gap-1.5 bg-emerald-700/50 border border-emerald-500/30 text-emerald-300 text-xs px-4 py-1.5 rounded-full font-medium mb-6">
                            <span>{roleInfo.icon}</span>
                            {roleInfo.label}
                        </span>

                        {/* Stats */}
                        <div className="w-full space-y-2 border-t border-white/10 pt-5">
                            {[
                                { label: "Member Since", value: user?.created_at || "—" },
                                { label: "Phone", value: user?.phone || "Not added" },
                                { label: "Status", value: "Active" },
                            ].map((s) => (
                                <div key={s.label} className="flex justify-between items-center text-sm">
                                    <span className="text-white/40">{s.label}</span>
                                    <span className="text-white/80 font-medium">{s.value}</span>
                                </div>
                            ))}
                        </div>

                        {/* Status indicator */}
                        <div className="mt-5 flex items-center gap-2 text-xs text-emerald-400">
                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block" />
                            Online
                        </div>
                    </div>

                    {/* ── RIGHT: Edit Form ── */}
                    <div className="lg:col-span-2 fade-up-d2 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 shadow-xl">
                        <div className="flex items-center justify-between mb-7">
                            <h3 className="text-lg font-semibold text-white">Account Details</h3>
                            {!editing && (
                                <button
                                    onClick={() => { setEditing(true); setMsg({ text: "", type: "" }); }}
                                    className="px-4 py-2 rounded-xl bg-emerald-600/80 hover:bg-emerald-500 text-white text-sm font-medium transition-all"
                                >
                                    ✏️ Edit Profile
                                </button>
                            )}
                        </div>

                        {/* Feedback message */}
                        {msg.text && (
                            <div className={`mb-5 px-4 py-3 rounded-xl text-sm font-medium ${msg.type === "success"
                                ? "bg-green-500/20 border border-green-400/30 text-green-300"
                                : "bg-red-500/20 border border-red-400/30 text-red-300"
                                }`}>
                                {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                            </div>
                        )}

                        <form onSubmit={handleSave} className="space-y-5">
                            {/* Name */}
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Full Name</label>
                                {editing ? (
                                    <input
                                        type="text" value={form.name}
                                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                                        required
                                        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition"
                                    />
                                ) : (
                                    <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.name}</p>
                                )}
                            </div>

                            {/* Email (read-only) */}
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Email Address <span className="text-white/30">(cannot be changed)</span></label>
                                <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm">{user?.email}</p>
                            </div>

                            {/* Phone */}
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Phone Number</label>
                                {editing ? (
                                    <input
                                        type="tel" value={form.phone}
                                        onChange={(e) => setForm({ ...form, phone: e.target.value })}
                                        placeholder="+92 300 0000000"
                                        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition"
                                    />
                                ) : (
                                    <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.phone || "—"}</p>
                                )}
                            </div>

                            {/* Password section — only when editing */}
                            {editing && (
                                <>
                                    <div className="border-t border-white/10 pt-5">
                                        <p className="text-sm text-white/50 mb-4">Change Password <span className="text-white/30">(leave blank to keep current)</span></p>
                                        <div className="space-y-4">
                                            <div>
                                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Current Password</label>
                                                <input
                                                    type="password" value={form.current_password}
                                                    onChange={(e) => setForm({ ...form, current_password: e.target.value })}
                                                    placeholder="••••••••"
                                                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm text-white/60 mb-1.5 font-medium">New Password</label>
                                                <input
                                                    type="password" value={form.new_password}
                                                    onChange={(e) => setForm({ ...form, new_password: e.target.value })}
                                                    placeholder="••••••••"
                                                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Confirm New Password</label>
                                                <input
                                                    type="password" value={form.confirm_new_password}
                                                    onChange={(e) => setForm({ ...form, confirm_new_password: e.target.value })}
                                                    placeholder="••••••••"
                                                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* Buttons */}
                            {editing && (
                                <div className="flex items-center gap-3 pt-2">
                                    <button
                                        type="submit" disabled={saving}
                                        className={`flex-1 py-3 rounded-xl font-semibold text-white transition-all shadow-lg ${saving ? "bg-gray-500 cursor-not-allowed" : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/25"}`}
                                    >
                                        {saving ? "Saving…" : "Save Changes"}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => { setEditing(false); setMsg({ text: "", type: "" }); setForm((f) => ({ ...f, name: user?.name, phone: user?.phone || "", current_password: "", new_password: "", confirm_new_password: "" })); }}
                                        className="px-6 py-3 rounded-xl border border-white/20 text-white/70 hover:bg-white/10 transition text-sm font-medium"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            )}
                        </form>
                    </div>
                </div>
            </>
        </DashboardLayout>
    );
}
