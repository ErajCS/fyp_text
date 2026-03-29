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
    return name.split(" ").map((n) => n[0]).slice(0, 2).join("").toUpperCase();
}

const SURVEY_KEY = "pqnk_user_survey_v1";
const CROPS_LIST = ["Wheat", "Rice", "Cotton", "Maize", "Sugarcane", "Vegetables", "Fruits", "Pulses", "Oilseeds", "Other"];

// ── User Survey Component ─────────────────────────────────────────────────────
function UserSurvey({ userId }) {
    const savedRaw = (() => { try { return JSON.parse(localStorage.getItem(SURVEY_KEY + "_" + userId)); } catch { return null; } })();
    const [expanded, setExpanded] = useState(!savedRaw);
    const [submitted, setSubmitted] = useState(!!savedRaw);
    const [survey, setSurvey] = useState(savedRaw || {
        farmSize: "", crops: [], challenge: "", howHeard: "", yearsExp: "", literacy: "",
    });

    const toggleCrop = (crop) =>
        setSurvey((s) => ({
            ...s, crops: s.crops.includes(crop) ? s.crops.filter((c) => c !== crop) : [...s.crops, crop],
        }));

    const handleSubmit = (e) => {
        e.preventDefault();
        localStorage.setItem(SURVEY_KEY + "_" + userId, JSON.stringify({ ...survey, submittedAt: new Date().toISOString() }));
        setSubmitted(true);
        setExpanded(false);
    };

    const selectCls = "w-full px-4 py-2.5 rounded-xl bg-white/10 border border-white/15 text-white text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/50 transition appearance-none select-dark";

    return (
        <div className="fade-up-d2">
            <div className={`bg-gradient-to-r from-emerald-800/50 to-green-900/50 border rounded-2xl overflow-hidden shadow-xl transition-all ${expanded ? "border-emerald-400/30" : "border-white/10"}`}>
                <button
                    type="button"
                    onClick={() => setExpanded((v) => !v)}
                    className="w-full flex items-center justify-between px-7 py-5 text-left"
                >
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-emerald-700/60 flex items-center justify-center text-xl">📋</div>
                        <div>
                            <p className="text-white font-semibold text-sm">
                                {submitted ? "✅ Survey Completed — Thank You!" : "Help Us Understand You Better"}
                            </p>
                            <p className="text-white/40 text-xs mt-0.5">
                                {submitted ? "Your responses have been saved. Click to review or update." : "A quick 6-question survey to help us serve farmers better (2 min)"}
                            </p>
                        </div>
                    </div>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
                        className={`w-5 h-5 text-white/40 transition-transform duration-300 ${expanded ? "rotate-180" : ""}`}>
                        <polyline points="6 9 12 15 18 9" />
                    </svg>
                </button>

                {expanded && (
                    <div className="px-7 pb-7 border-t border-white/10">
                        {submitted && (
                            <div className="mt-5 mb-4 flex items-center gap-2 bg-green-500/15 border border-green-400/20 rounded-xl px-4 py-3 text-green-300 text-sm">
                                ✅ Your responses are saved. You may update them at any time.
                            </div>
                        )}
                        <form onSubmit={handleSubmit} className="mt-5 space-y-6">

                            {/* Q1 Farm Size */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">1. Farm / Land Holding Size</label>
                                <select className={selectCls} value={survey.farmSize} onChange={(e) => setSurvey({ ...survey, farmSize: e.target.value })} required>
                                    <option value="" disabled>Select land size…</option>
                                    <option value="less_1">Less than 1 acre</option>
                                    <option value="1_5">1 – 5 acres</option>
                                    <option value="5_20">5 – 20 acres</option>
                                    <option value="20_50">20 – 50 acres</option>
                                    <option value="50_plus">More than 50 acres</option>
                                    <option value="no_land">No land (student / researcher)</option>
                                </select>
                            </div>

                            {/* Q2 Crops */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">
                                    2. Primary Crops Grown <span className="text-white/30 font-normal">(select all that apply)</span>
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {CROPS_LIST.map((crop) => (
                                        <button key={crop} type="button" onClick={() => toggleCrop(crop)}
                                            className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${survey.crops.includes(crop)
                                                ? "bg-emerald-600/60 border-emerald-400/50 text-white"
                                                : "bg-white/5 border-white/15 text-white/50 hover:border-emerald-400/30 hover:text-white"}`}>
                                            {crop}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Q3 Challenge */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">3. Your Biggest Farming Challenge</label>
                                <select className={selectCls} value={survey.challenge} onChange={(e) => setSurvey({ ...survey, challenge: e.target.value })} required>
                                    <option value="" disabled>Select main challenge…</option>
                                    <option value="water">Water access / irrigation</option>
                                    <option value="pest">Pest &amp; disease control</option>
                                    <option value="soil">Soil health / fertility</option>
                                    <option value="market">Market access / pricing</option>
                                    <option value="financing">Financing / credit</option>
                                    <option value="knowledge">Lack of agricultural knowledge</option>
                                    <option value="climate">Climate change / weather</option>
                                    <option value="labour">Labour availability</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>

                            {/* Q4 How Heard */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">4. How Did You Hear About PQNK / AgriChat?</label>
                                <select className={selectCls} value={survey.howHeard} onChange={(e) => setSurvey({ ...survey, howHeard: e.target.value })} required>
                                    <option value="" disabled>Select source…</option>
                                    <option value="youtube">YouTube video by Mr. Asif Sharif</option>
                                    <option value="whatsapp">WhatsApp group / broadcast</option>
                                    <option value="par">PAR extension worker / event</option>
                                    <option value="word_of_mouth">Word of mouth (farmer, friend)</option>
                                    <option value="social_media">Social media (Facebook, Instagram)</option>
                                    <option value="university">University / academic context</option>
                                    <option value="pedaver">Pedaver / field demonstration</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>

                            {/* Q5 Experience */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">5. Years of Farming Experience</label>
                                <select className={selectCls} value={survey.yearsExp} onChange={(e) => setSurvey({ ...survey, yearsExp: e.target.value })} required>
                                    <option value="" disabled>Select experience level…</option>
                                    <option value="none">No farming experience</option>
                                    <option value="1_3">1 – 3 years</option>
                                    <option value="3_10">3 – 10 years</option>
                                    <option value="10_25">10 – 25 years</option>
                                    <option value="25_plus">More than 25 years</option>
                                </select>
                            </div>

                            {/* Q6 Literacy */}
                            <div>
                                <label className="block text-white/70 text-sm font-semibold mb-2">
                                    6. Reading / Literacy Level <span className="text-white/30 font-normal">(helps us optimise the platform for you)</span>
                                </label>
                                <div className="space-y-2">
                                    {[
                                        { value: "full", label: "I can read and write comfortably in English or Urdu" },
                                        { value: "partial", label: "I am partially literate — I prefer simple language" },
                                        { value: "audio", label: "I prefer audio responses — I rely on listening rather than reading" },
                                    ].map((opt) => (
                                        <label key={opt.value}
                                            className={`flex items-start gap-3 px-4 py-3 rounded-xl border cursor-pointer transition-all ${survey.literacy === opt.value
                                                ? "bg-emerald-700/40 border-emerald-400/40 text-white"
                                                : "bg-white/5 border-white/10 text-white/55 hover:border-white/20"}`}>
                                            <input type="radio" name="literacy" value={opt.value}
                                                checked={survey.literacy === opt.value}
                                                onChange={(e) => setSurvey({ ...survey, literacy: e.target.value })}
                                                className="mt-0.5 accent-emerald-400" required />
                                            <span className="text-sm">{opt.label}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            <button type="submit"
                                className="w-full py-3.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 text-white font-bold text-sm shadow-lg hover:shadow-green-500/25 hover:scale-[1.01] transition-all">
                                {submitted ? "💾 Update Survey Responses" : "✅ Submit Survey"}
                            </button>
                        </form>
                    </div>
                )}
            </div>
        </div>
    );
}

// ── Main Profile Component ────────────────────────────────────────────────────
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
                if (data) { setUser(data); setForm((f) => ({ ...f, name: data.name, phone: data.phone || "" })); }
            })
            .finally(() => setLoading(false));
    }, [navigate]);

    const handleSave = async (e) => {
        e.preventDefault();
        if (form.new_password && form.new_password !== form.confirm_new_password) {
            setMsg({ text: "New passwords do not match", type: "error" }); return;
        }
        setSaving(true); setMsg({ text: "", type: "" });
        try {
            const res = await fetch("/api/user/update", {
                method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include",
                body: JSON.stringify({ name: form.name, phone: form.phone, current_password: form.current_password || undefined, new_password: form.new_password || undefined }),
            });
            const data = await res.json();
            if (data.success) {
                setUser(data.user); setEditing(false);
                setForm((f) => ({ ...f, current_password: "", new_password: "", confirm_new_password: "" }));
                setMsg({ text: "Profile updated successfully!", type: "success" });
                localStorage.setItem("user", JSON.stringify(data.user));
            } else { setMsg({ text: data.message || "Update failed", type: "error" }); }
        } catch { setMsg({ text: "Could not connect to server", type: "error" }); }
        finally { setSaving(false); }
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
          .fade-up    { animation: fadeUp 0.5s ease-out both; }
          .fade-up-d1 { animation: fadeUp 0.5s ease-out 0.1s both; }
          .fade-up-d2 { animation: fadeUp 0.5s ease-out 0.2s both; }
          .select-dark option { background-color: #064e3b; color: white; }
        `}</style>

                <div className="mb-8 fade-up">
                    <h1 className="text-3xl font-bold text-white drop-shadow-lg mb-1">My Profile</h1>
                    <p className="text-white/60 text-sm">Manage your account information and credentials</p>
                </div>

                <div className="grid lg:grid-cols-3 gap-6">
                    {/* ── LEFT: Avatar Card ── */}
                    <div className="fade-up-d1 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 flex flex-col items-center text-center shadow-xl">
                        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center text-3xl font-bold text-emerald-950 shadow-xl mb-4">
                            {getInitials(user?.name)}
                        </div>
                        <h2 className="text-xl font-bold text-white mb-1">{user?.name}</h2>
                        <p className="text-white/50 text-sm mb-4">{user?.email}</p>
                        <span className="inline-flex items-center gap-1.5 bg-emerald-700/50 border border-emerald-500/30 text-emerald-300 text-xs px-4 py-1.5 rounded-full font-medium mb-6">
                            <span>{roleInfo.icon}</span> {roleInfo.label}
                        </span>
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
                        <div className="mt-5 flex items-center gap-2 text-xs text-emerald-400">
                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block" />
                            Online
                        </div>
                    </div>

                    {/* ── RIGHT: Edit Form + Survey ── */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="fade-up-d1 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 shadow-xl">
                            <div className="flex items-center justify-between mb-7">
                                <h3 className="text-lg font-semibold text-white">Account Details</h3>
                                {!editing && (
                                    <button onClick={() => { setEditing(true); setMsg({ text: "", type: "" }); }}
                                        className="px-4 py-2 rounded-xl bg-emerald-600/80 hover:bg-emerald-500 text-white text-sm font-medium transition-all">
                                        ✏️ Edit Profile
                                    </button>
                                )}
                            </div>

                            {msg.text && (
                                <div className={`mb-5 px-4 py-3 rounded-xl text-sm font-medium ${msg.type === "success"
                                    ? "bg-green-500/20 border border-green-400/30 text-green-300"
                                    : "bg-red-500/20 border border-red-400/30 text-red-300"}`}>
                                    {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                                </div>
                            )}

                            <form onSubmit={handleSave} className="space-y-5">
                                {/* Name */}
                                <div>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">Full Name</label>
                                    {editing
                                        ? <input type="text" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required
                                            className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition" />
                                        : <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.name}</p>
                                    }
                                </div>
                                {/* Email */}
                                <div>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">Email Address <span className="text-white/30">(cannot be changed)</span></label>
                                    <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm">{user?.email}</p>
                                </div>
                                {/* Phone */}
                                <div>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">Phone Number</label>
                                    {editing
                                        ? <input type="tel" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} placeholder="+92 300 0000000"
                                            className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition" />
                                        : <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.phone || "—"}</p>
                                    }
                                </div>
                                {/* Password */}
                                {editing && (
                                    <div className="border-t border-white/10 pt-5">
                                        <p className="text-sm text-white/50 mb-4">Change Password <span className="text-white/30">(leave blank to keep current)</span></p>
                                        <div className="space-y-4">
                                            {[
                                                { key: "current_password", label: "Current Password" },
                                                { key: "new_password", label: "New Password" },
                                                { key: "confirm_new_password", label: "Confirm New Password" },
                                            ].map(({ key, label }) => (
                                                <div key={key}>
                                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">{label}</label>
                                                    <input type="password" value={form[key]} onChange={(e) => setForm({ ...form, [key]: e.target.value })} placeholder="••••••••"
                                                        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition" />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {editing && (
                                    <div className="flex items-center gap-3 pt-2">
                                        <button type="submit" disabled={saving}
                                            className={`flex-1 py-3 rounded-xl font-semibold text-white transition-all shadow-lg ${saving ? "bg-gray-500 cursor-not-allowed" : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/25"}`}>
                                            {saving ? "Saving…" : "Save Changes"}
                                        </button>
                                        <button type="button"
                                            onClick={() => { setEditing(false); setMsg({ text: "", type: "" }); setForm((f) => ({ ...f, name: user?.name, phone: user?.phone || "", current_password: "", new_password: "", confirm_new_password: "" })); }}
                                            className="px-6 py-3 rounded-xl border border-white/20 text-white/70 hover:bg-white/10 transition text-sm font-medium">
                                            Cancel
                                        </button>
                                    </div>
                                )}
                            </form>
                        </div>

                        {/* ── USER SURVEY ── */}
                        <UserSurvey userId={user?.id || user?.email || "guest"} />
                    </div>
                </div>
            </>
        </DashboardLayout>
    );
}
