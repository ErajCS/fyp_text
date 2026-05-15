import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useLanguage, TRANSLATIONS } from "../context/LanguageContext";

export default function Signup() {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        name: "",
        email: "",
        phone: "",
        password: "",
        confirmPassword: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);
    const [loading, setLoading] = useState(false);
    const [agreed, setAgreed] = useState(false);
    const [error, setError] = useState("");
    const [showPwHint, setShowPwHint] = useState(false);
    
    const { lang, toggleLang } = useLanguage();
    const t = TRANSLATIONS[lang];

    const pwRules = [
        { test: (pw) => pw.length >= 8, label: "At least 8 characters" },
        { test: (pw) => /[A-Z]/.test(pw), label: "At least one uppercase letter" },
        { test: (pw) => /[a-z]/.test(pw), label: "At least one lowercase letter" },
        { test: (pw) => /\d/.test(pw), label: "At least one digit (0–9)" },
        { test: (pw) => /[^A-Za-z0-9]/.test(pw), label: "At least one special character (!@#$%…)" },
    ];
    const pwScore = pwRules.filter((r) => r.test(formData.password)).length;
    const pwColor = pwScore <= 1 ? "#ef4444" : pwScore <= 3 ? "#f59e0b" : "#22c55e";

    const handleChange = (e) => {
        setError("");
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        if (formData.password !== formData.confirmPassword) {
            setError("Passwords do not match");
            return;
        }
        if (pwScore < 5) {
            setError("Password does not meet all requirements.");
            return;
        }
        if (!agreed) {
            setError("Please agree to the Terms & Conditions");
            return;
        }

        setLoading(true);
        try {
            const res = await fetch("/api/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({
                    name: formData.name,
                    email: formData.email,
                    phone: formData.phone,
                    password: formData.password,
                    confirm_password: formData.confirmPassword,
                }),
            });

            let data;
            try {
                data = await res.json();
            } catch {
                setError("Server error — could not parse response");
                return;
            }

            if (res.ok && data.success) {
                // Redirect to OTP verification page
                navigate(`/verify-otp?email=${encodeURIComponent(data.email)}`);
            } else {
                setError(data.message || "Signup failed. Please try again.");
            }
        } catch (err) {
            console.error("Signup error:", err);
            setError("Could not connect to server. Make sure the backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <style>{`
        @keyframes floatIn {
          from { opacity: 0; transform: translateY(24px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .animate-float-in { animation: floatIn 0.7s ease-out both; }
        .animate-float-in-delay { animation: floatIn 0.7s ease-out 0.15s both; }
      `}</style>

            <div
                className="min-h-screen flex relative overflow-hidden"
                style={{
                    backgroundImage: "url('/agri-bg.png')",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                }}
            >
                <div className="absolute inset-0 bg-gradient-to-br from-green-950/70 via-emerald-900/60 to-green-950/80 z-0" />

                {/* ── LEFT PANEL ── */}
                <div className="hidden lg:flex flex-col justify-between w-[48%] relative z-10 p-12">
                    <div className="animate-float-in">
                        <div className="flex items-center gap-3 mb-16">
                            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-2xl shadow-xl shadow-green-500/30">
                                🌿
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold text-white">AgriChat</h1>
                                <p className="text-xs text-emerald-400/70 tracking-widest uppercase">PQNK Knowledge System</p>
                            </div>
                        </div>

                        <h2 className="text-5xl font-bold text-white leading-tight mb-6">
                            Join the Future<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-300">
                                of Agriculture
                            </span>
                        </h2>

                        <p className="text-white/60 text-lg leading-relaxed max-w-md mb-10">
                            Create your account and get access to AI-powered crop intelligence, research resources, and smart advisory systems.
                        </p>

                        <div className="space-y-4">
                            {[
                                { icon: "🤖", title: "AI Assistant", desc: "Get instant answers on crop management" },
                                { icon: "📚", title: "Knowledge Base", desc: "Access curated PQNK agricultural resources" },
                                { icon: "🌦️", title: "Smart Advisory", desc: "Weather-aware farming recommendations" },
                            ].map((feat, i) => (
                                <div key={i} className="flex gap-3.5 items-start">
                                    <div className="w-10 h-10 rounded-xl bg-white/10 border border-white/10 flex items-center justify-center text-lg flex-shrink-0">
                                        {feat.icon}
                                    </div>
                                    <div>
                                        <p className="text-white font-semibold text-sm">{feat.title}</p>
                                        <p className="text-white/50 text-xs">{feat.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="animate-float-in-delay flex gap-8">
                        {[
                            { label: "Free to Join", value: "✓" },
                            { label: "Secure", value: "🔒" },
                            { label: "No Credit Card", value: "✓" },
                        ].map((stat, i) => (
                            <div key={i} className="flex items-center gap-2">
                                <span className="text-lg">{stat.value}</span>
                                <span className="text-xs text-emerald-400/60 uppercase tracking-wider">{stat.label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* ── RIGHT PANEL — form ── */}
                <div className="flex-1 flex items-center justify-center relative z-10 p-8">
                    <div className="w-full max-w-md animate-float-in-delay">
                        <div className="bg-white/95 backdrop-blur-2xl rounded-3xl shadow-2xl p-9 border border-white/50">

                            {/* Mobile logo */}
                            <div className="lg:hidden flex items-center gap-2.5 mb-6">
                                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-lg shadow-md">🌿</div>
                                <h2 className="text-lg font-bold text-green-800">AgriChat</h2>
                            </div>

                            <h3 className="text-2xl font-bold text-gray-800 mb-1">{t.authCreateTitle}</h3>
                            <p className="text-gray-500 text-sm mb-6">{t.authCreateDesc}</p>

                            {/* Error banner */}
                            {error && (
                                <div className="mb-5 px-4 py-3 bg-red-50 border border-red-200 text-red-700 rounded-xl text-sm flex items-start gap-2">
                                    <span className="flex-shrink-0">⚠️</span>
                                    {error}
                                </div>
                            )}

                            <form onSubmit={handleSubmit} className="space-y-4">
                                {/* Full Name */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.authFullName}</label>
                                    <input
                                        type="text" name="name" value={formData.name}
                                        onChange={handleChange} required
                                        placeholder={t.authFullNamePlace}
                                        className="w-full px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition"
                                    />
                                </div>

                                {/* Email */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.authEmailLabel}</label>
                                    <input
                                        type="email" name="email" value={formData.email}
                                        onChange={handleChange} required
                                        placeholder={t.authEmailPlaceholder}
                                        className="w-full px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition"
                                    />
                                </div>

                                {/* Phone */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">
                                        {t.authPhoneLabel} <span className="text-gray-400 font-normal">(optional)</span>
                                    </label>
                                    <input
                                        type="tel" name="phone" value={formData.phone}
                                        onChange={handleChange}
                                        placeholder="+92 300 0000000"
                                        className="w-full px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition"
                                    />
                                </div>

                                {/* Password */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.authPasswordLabel}</label>
                                    <div className="relative">
                                        <input
                                            type={showPassword ? "text" : "password"} name="password"
                                            value={formData.password} onChange={handleChange} required
                                            onFocus={() => setShowPwHint(true)}
                                            placeholder="Min. 8 characters"
                                            className="w-full px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition pr-12"
                                        />
                                        <button type="button" onClick={() => setShowPassword(!showPassword)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 text-sm">
                                            {showPassword ? "🙈" : "👁️"}
                                        </button>
                                    </div>
                                    {showPwHint && formData.password && (
                                        <div className="mt-3 space-y-2 animate-float-in">
                                            {/* Strength bar */}
                                            <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
                                                <div className="h-full rounded-full transition-all duration-300"
                                                    style={{ width: `${(pwScore / 5) * 100}%`, backgroundColor: pwColor }} />
                                            </div>
                                            <div className="grid grid-cols-1 gap-1">
                                                {pwRules.map((r, i) => (
                                                    <p key={i} className={`text-[10px] flex items-center gap-1.5 transition-colors ${r.test(formData.password) ? "text-green-600 font-medium" : "text-gray-400"}`}>
                                                        <span className={`w-3.5 h-3.5 rounded-full flex items-center justify-center border ${r.test(formData.password) ? "bg-green-100 border-green-200 text-green-600" : "border-gray-200 text-transparent"}`}>
                                                            {r.test(formData.password) ? "✓" : ""}
                                                        </span>
                                                        {r.label}
                                                    </p>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Confirm Password */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.authConfirmPassword}</label>
                                    <div className="relative">
                                        <input
                                            type={showConfirm ? "text" : "password"} name="confirmPassword"
                                            value={formData.confirmPassword} onChange={handleChange} required
                                            placeholder="••••••••"
                                            className="w-full px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition pr-12"
                                        />
                                        <button type="button" onClick={() => setShowConfirm(!showConfirm)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 text-sm">
                                            {showConfirm ? "🙈" : "👁️"}
                                        </button>
                                    </div>
                                </div>

                                {/* Terms */}
                                <div className="flex items-start gap-2">
                                    <input
                                        type="checkbox" id="terms" checked={agreed}
                                        onChange={(e) => setAgreed(e.target.checked)}
                                        className="w-4 h-4 mt-0.5 rounded border-gray-300 text-green-600 focus:ring-green-500"
                                    />
                                    <label htmlFor="terms" className="text-xs text-gray-500 leading-relaxed">
                                        {t.authAgreeTerms1} <span className="text-green-600 font-medium cursor-pointer">{t.authAgreeTerms2}</span> {t.authAgreeTerms3} <span className="text-green-600 font-medium cursor-pointer">{t.authAgreeTerms4}</span>
                                    </label>
                                </div>

                                {/* Submit */}
                                <button
                                    type="submit" disabled={loading}
                                    className={`w-full py-3 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg ${loading
                                        ? "bg-gray-400 cursor-not-allowed"
                                        : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/30 hover:scale-[1.01]"
                                        }`}
                                >
                                    {loading ? t.authSigningUp : t.authSignUpBtn}
                                </button>
                            </form>
                            
                            {/* Urdu toggle */}
                            <button 
                                onClick={toggleLang}
                                className="w-full mt-5 py-2.5 rounded-xl border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition flex items-center justify-center gap-2"
                            >
                                🌐 {lang === "en" ? "Switch to Urdu (اردو)" : "Switch to English"}
                            </button>

                            {/* Login link */}
                            <p className="text-center text-sm text-gray-500 mt-6">
                                {t.authHaveAccount}{" "}
                                <Link to="/login" className="text-green-600 font-semibold hover:text-green-700 transition">
                                    {t.authLoginLink}
                                </Link>
                            </p>
                        </div>

                        <p className="text-center text-xs text-white/40 mt-6">
                            © 2026 AgriChat · PQNK Agriculture Repository
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}
