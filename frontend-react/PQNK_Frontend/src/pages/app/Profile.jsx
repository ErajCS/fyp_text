import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "../../layouts/DashboardLayout";
import { useLanguage, TRANSLATIONS } from "../../context/LanguageContext";

function getInitials(name = "") {
    return name.split(" ").map((n) => n[0]).slice(0, 2).join("").toUpperCase();
}

const SURVEY_KEY = "pqnk_user_survey_v1";

// ── User Survey Component ─────────────────────────────────────────────────────
function UserSurvey({ userId }) {
    const { lang } = useLanguage();
    const t = (key) => TRANSLATIONS[lang][key] || key;
    const isRtl = lang === "ur";

    const CROPS_LIST = [
        t("wheat") || "Wheat", t("rice") || "Rice", t("cotton") || "Cotton", t("maize") || "Maize",
        t("sugarcane") || "Sugarcane", t("vegetables") || "Vegetables", t("fruits") || "Fruits",
        t("pulses") || "Pulses", t("oilseeds") || "Oilseeds", t("other") || "Other"
    ];

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

    const selectCls = `w-full px-4 py-2.5 rounded-xl bg-white/10 border border-white/15 text-white text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/50 transition appearance-none select-dark ${isRtl ? "text-right" : ""}`;

    return (
        <div className="fade-up-d2">
            <div className={`bg-gradient-to-r from-emerald-800/50 to-green-900/50 border rounded-2xl overflow-hidden shadow-xl transition-all ${expanded ? "border-emerald-400/30" : "border-white/10"}`}>
                <button
                    type="button"
                    onClick={() => setExpanded((v) => !v)}
                    className={`w-full flex items-center justify-between px-7 py-5 text-left ${isRtl ? "flex-row-reverse text-right" : ""}`}
                >
                    <div className={`flex items-center gap-3 ${isRtl ? "flex-row-reverse" : ""}`}>
                        <div className="w-10 h-10 rounded-xl bg-emerald-700/60 flex items-center justify-center text-xl">📋</div>
                        <div>
                            <p className="text-white font-semibold text-sm">
                                {submitted ? t("surveyTitleDone") : t("surveyTitle")}
                            </p>
                            <p className="text-white/40 text-xs mt-0.5">
                                {submitted ? t("surveyDescDone") : t("surveyDesc")}
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
                            <div className={`mt-5 mb-4 flex items-center gap-2 bg-green-500/15 border border-green-400/20 rounded-xl px-4 py-3 text-green-300 text-sm ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                                {t("surveyUpdateMsg")}
                            </div>
                        )}
                        <form onSubmit={handleSubmit} className="mt-5 space-y-6">

                            {/* Q1 Farm Size */}
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">{t("surveyQ1")}</label>
                                <select className={selectCls} value={survey.farmSize} onChange={(e) => setSurvey({ ...survey, farmSize: e.target.value })} required>
                                    <option value="" disabled>{t("surveyQ1Select")}</option>
                                    <option value="less_1">{t("surveyQ1Opt1")}</option>
                                    <option value="1_5">{t("surveyQ1Opt2")}</option>
                                    <option value="5_20">{t("surveyQ1Opt3")}</option>
                                    <option value="20_50">{t("surveyQ1Opt4")}</option>
                                    <option value="50_plus">{t("surveyQ1Opt5")}</option>
                                    <option value="no_land">{t("surveyQ1Opt6")}</option>
                                </select>
                            </div>

                            {/* Q2 Crops */}
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">
                                    {t("surveyQ2")}
                                </label>
                                <div className={`flex flex-wrap gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
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
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">{t("surveyQ3")}</label>
                                <select className={selectCls} value={survey.challenge} onChange={(e) => setSurvey({ ...survey, challenge: e.target.value })} required>
                                    <option value="" disabled>{t("surveyQ3Select")}</option>
                                    <option value="water">{t("surveyQ3Opt1")}</option>
                                    <option value="pest">{t("surveyQ3Opt2")}</option>
                                    <option value="soil">{t("surveyQ3Opt3")}</option>
                                    <option value="market">{t("surveyQ3Opt4")}</option>
                                    <option value="financing">{t("surveyQ3Opt5")}</option>
                                    <option value="knowledge">{t("surveyQ3Opt6")}</option>
                                    <option value="climate">{t("surveyQ3Opt7")}</option>
                                    <option value="labour">{t("surveyQ3Opt8")}</option>
                                    <option value="other">{t("surveyQ3Opt9")}</option>
                                </select>
                            </div>

                            {/* Q4 How Heard */}
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">{t("surveyQ4")}</label>
                                <select className={selectCls} value={survey.howHeard} onChange={(e) => setSurvey({ ...survey, howHeard: e.target.value })} required>
                                    <option value="" disabled>{t("surveyQ4Select")}</option>
                                    <option value="youtube">{t("surveyQ4Opt1")}</option>
                                    <option value="whatsapp">{t("surveyQ4Opt2")}</option>
                                    <option value="par">{t("surveyQ4Opt3")}</option>
                                    <option value="word_of_mouth">{t("surveyQ4Opt4")}</option>
                                    <option value="social_media">{t("surveyQ4Opt5")}</option>
                                    <option value="university">{t("surveyQ4Opt6")}</option>
                                    <option value="pedaver">{t("surveyQ4Opt7")}</option>
                                    <option value="other">{t("surveyQ4Opt8")}</option>
                                </select>
                            </div>

                            {/* Q5 Experience */}
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">{t("surveyQ5")}</label>
                                <select className={selectCls} value={survey.yearsExp} onChange={(e) => setSurvey({ ...survey, yearsExp: e.target.value })} required>
                                    <option value="" disabled>{t("surveyQ5Select")}</option>
                                    <option value="none">{t("surveyQ5Opt1")}</option>
                                    <option value="1_3">{t("surveyQ5Opt2")}</option>
                                    <option value="3_10">{t("surveyQ5Opt3")}</option>
                                    <option value="10_25">{t("surveyQ5Opt4")}</option>
                                    <option value="25_plus">{t("surveyQ5Opt5")}</option>
                                </select>
                            </div>

                            {/* Q6 Literacy */}
                            <div className={isRtl ? "text-right" : ""}>
                                <label className="block text-white/70 text-sm font-semibold mb-2">
                                    {t("surveyQ6")}
                                </label>
                                <div className="space-y-2">
                                    {[
                                        { value: "full", label: t("surveyQ6Opt1") },
                                        { value: "partial", label: t("surveyQ6Opt2") },
                                        { value: "audio", label: t("surveyQ6Opt3") },
                                    ].map((opt) => (
                                        <label key={opt.value}
                                            className={`flex items-start gap-3 px-4 py-3 rounded-xl border cursor-pointer transition-all ${survey.literacy === opt.value
                                                ? "bg-emerald-700/40 border-emerald-400/40 text-white"
                                                : "bg-white/5 border-white/10 text-white/55 hover:border-white/20"} ${isRtl ? "flex-row-reverse text-right" : ""}`}>
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
                                {submitted ? t("surveyUpdateBtn") : t("surveySubmitBtn")}
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
    const { lang } = useLanguage();
    const t = (key) => TRANSLATIONS[lang][key] || key;
    const isRtl = lang === "ur";

    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [editing, setEditing] = useState(false);
    const [saving, setSaving] = useState(false);
    const [msg, setMsg] = useState({ text: "", type: "" });
    const [form, setForm] = useState({ name: "", phone: "", current_password: "", new_password: "", confirm_new_password: "" });

    const ROLE_LABELS = {
        farmer: { label: lang === "en" ? "Farmer" : "کسان", icon: "🌾", color: "green" },
        researcher: { label: lang === "en" ? "Researcher" : "محقق", icon: "🔬", color: "blue" },
        student: { label: lang === "en" ? "Student" : "طالب علم", icon: "🎓", color: "purple" },
        admin: { label: lang === "en" ? "Admin" : "ایڈمن", icon: "🛡️", color: "red" },
        superadmin: { label: lang === "en" ? "Super Admin" : "سپر ایڈمن", icon: "👑", color: "amber" },
        seeker: { label: lang === "en" ? "User" : "صارف", icon: "👤", color: "emerald" },
    };

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
            setMsg({ text: t("profilePassMismatch"), type: "error" }); return;
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
                setMsg({ text: t("profileUpdateSuccess"), type: "success" });
                localStorage.setItem("user", JSON.stringify(data.user));
            } else { setMsg({ text: data.message || t("profileUpdateFail"), type: "error" }); }
        } catch { setMsg({ text: t("profileServerErr"), type: "error" }); }
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
            <div dir={isRtl ? "rtl" : "ltr"} className={isRtl ? "font-urdu" : ""}>
                <style>{`
          @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
          .fade-up    { animation: fadeUp 0.5s ease-out both; }
          .fade-up-d1 { animation: fadeUp 0.5s ease-out 0.1s both; }
          .fade-up-d2 { animation: fadeUp 0.5s ease-out 0.2s both; }
          .select-dark option { background-color: #064e3b; color: white; }
          .font-urdu { font-family: 'Noto Nastaliq Urdu', serif; }
        `}</style>

                <div className={`mb-8 fade-up ${isRtl ? "text-right" : ""}`}>
                    <h1 className="text-3xl font-bold text-white drop-shadow-lg mb-1">{t("profileTitle")}</h1>
                    <p className="text-white/60 text-sm">{t("profileDesc")}</p>
                </div>

                <div className={`grid lg:grid-cols-3 gap-6 ${isRtl ? "lg:flex lg:flex-row-reverse" : ""}`}>
                    {/* ── LEFT: Avatar Card ── */}
                    <div className="fade-up-d1 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 lg:w-1/3 flex flex-col items-center text-center shadow-xl">
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
                                { label: t("profileMemberSince"), value: user?.created_at || "—" },
                                { label: t("contactPhone"), value: user?.phone || "—" },
                                { label: t("profileStatus"), value: "Active" },
                            ].map((s) => (
                                <div key={s.label} className={`flex justify-between items-center text-sm ${isRtl ? "flex-row-reverse" : ""}`}>
                                    <span className="text-white/40">{s.label}</span>
                                    <span className="text-white/80 font-medium">{s.value}</span>
                                </div>
                            ))}
                        </div>
                        <div className={`mt-5 flex items-center gap-2 text-xs text-emerald-400 ${isRtl ? "flex-row-reverse" : ""}`}>
                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block" />
                            {t("profileOnline")}
                        </div>
                    </div>

                    {/* ── RIGHT: Edit Form + Survey ── */}
                    <div className="lg:w-2/3 space-y-6">
                        <div className="fade-up-d1 bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 shadow-xl">
                            <div className={`flex items-center justify-between mb-7 ${isRtl ? "flex-row-reverse" : ""}`}>
                                <h3 className="text-lg font-semibold text-white">{t("profileAccountDetails")}</h3>
                                {!editing && (
                                    <button onClick={() => { setEditing(true); setMsg({ text: "", type: "" }); }}
                                        className="px-4 py-2 rounded-xl bg-emerald-600/80 hover:bg-emerald-500 text-white text-sm font-medium transition-all">
                                        {t("profileEditBtn")}
                                    </button>
                                )}
                            </div>

                            {msg.text && (
                                <div className={`mb-5 px-4 py-3 rounded-xl text-sm font-medium ${isRtl ? "text-right" : ""} ${msg.type === "success"
                                    ? "bg-green-500/20 border border-green-400/30 text-green-300"
                                    : "bg-red-500/20 border border-red-400/30 text-red-300"}`}>
                                    {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                                </div>
                            )}

                            <form onSubmit={handleSave} className="space-y-5">
                                {/* Name */}
                                <div className={isRtl ? "text-right" : ""}>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">{t("profileFullName")}</label>
                                    {editing
                                        ? <input type="text" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required
                                            className={`w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition ${isRtl ? "text-right" : ""}`} />
                                        : <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.name}</p>
                                    }
                                </div>
                                {/* Email */}
                                <div className={isRtl ? "text-right" : ""}>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">{t("profileEmailLock")}</label>
                                    <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm">{user?.email}</p>
                                </div>
                                {/* Phone */}
                                <div className={isRtl ? "text-right" : ""}>
                                    <label className="block text-sm text-white/60 mb-1.5 font-medium">{t("contactPhone")}</label>
                                    {editing
                                        ? <input type="tel" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} placeholder="+92 300 0000000"
                                            className={`w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 focus:border-green-400/60 text-sm transition ${isRtl ? "text-right" : ""}`} />
                                        : <p className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm">{user?.phone || "—"}</p>
                                    }
                                </div>
                                {/* Password */}
                                {editing && (
                                    <div className="border-t border-white/10 pt-5 text-right">
                                        <p className={`text-sm text-white/50 mb-4 ${isRtl ? "text-right" : ""}`}>{t("profileChangePass")}</p>
                                        <div className="space-y-4">
                                            {[
                                                { key: "current_password", label: t("profileCurrentPass") },
                                                { key: "new_password", label: t("profileNewPass") },
                                                { key: "confirm_new_password", label: t("profileConfirmPass") },
                                            ].map(({ key, label }) => (
                                                <div key={key}>
                                                    <label className={`block text-sm text-white/60 mb-1.5 font-medium ${isRtl ? "text-right" : ""}`}>{label}</label>
                                                    <input type="password" value={form[key]} onChange={(e) => setForm({ ...form, [key]: e.target.value })} placeholder="••••••••"
                                                        className={`w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-green-400/40 text-sm transition ${isRtl ? "text-right" : ""}`} />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {editing && (
                                    <div className={`flex items-center gap-3 pt-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                        <button type="submit" disabled={saving}
                                            className={`flex-1 py-3 rounded-xl font-semibold text-white transition-all shadow-lg ${saving ? "bg-gray-500 cursor-not-allowed" : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/25"}`}>
                                            {saving ? t("profileSavingBtn") : t("profileSaveBtn")}
                                        </button>
                                        <button type="button"
                                            onClick={() => { setEditing(false); setMsg({ text: "", type: "" }); setForm((f) => ({ ...f, name: user?.name, phone: user?.phone || "", current_password: "", new_password: "", confirm_new_password: "" })); }}
                                            className="px-6 py-3 rounded-xl border border-white/20 text-white/70 hover:bg-white/10 transition text-sm font-medium">
                                            {t("profileCancelBtn")}
                                        </button>
                                    </div>
                                )}
                            </form>
                        </div>

                        {/* ── USER SURVEY ── */}
                        <UserSurvey userId={user?.id || user?.email || "guest"} />
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
}
