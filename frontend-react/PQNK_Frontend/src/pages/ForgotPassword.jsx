import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function ForgotPassword() {
    const navigate = useNavigate();
    const [step, setStep] = useState("email"); // "email" | "otp"
    const [email, setEmail] = useState("");
    const [otp, setOtp] = useState("");
    const [newPw, setNewPw] = useState("");
    const [confirmPw, setConfirmPw] = useState("");
    const [loading, setLoading] = useState(false);
    const [msg, setMsg] = useState({ text: "", type: "" });
    const [showPwHint, setShowPwHint] = useState(false);

    const pwRules = [
        { test: (pw) => pw.length >= 8, label: "At least 8 characters" },
        { test: (pw) => /[A-Z]/.test(pw), label: "At least one uppercase letter" },
        { test: (pw) => /[a-z]/.test(pw), label: "At least one lowercase letter" },
        { test: (pw) => /\d/.test(pw), label: "At least one digit (0–9)" },
        { test: (pw) => /[^A-Za-z0-9]/.test(pw), label: "At least one special character (!@#$%…)" },
    ];
    const pwScore = pwRules.filter((r) => r.test(newPw)).length;
    const pwColor = pwScore <= 1 ? "#ef4444" : pwScore <= 3 ? "#f59e0b" : "#22c55e";

    const sendOtp = async (e) => {
        e.preventDefault();
        setLoading(true); setMsg({ text: "", type: "" });
        try {
            const res = await fetch("/api/forgot-password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email }),
            });
            const data = await res.json();
            if (data.success) {
                setStep("otp");
                setMsg({ text: "Check your email for the 6-digit reset code.", type: "success" });
            } else {
                setMsg({ text: data.message || "Request failed.", type: "error" });
            }
        } catch {
            setMsg({ text: "Network error. Please try again.", type: "error" });
        } finally { setLoading(false); }
    };

    const resetPassword = async (e) => {
        e.preventDefault();
        if (newPw !== confirmPw) {
            setMsg({ text: "Passwords do not match.", type: "error" }); return;
        }
        if (pwScore < 5) {
            setMsg({ text: "Password does not meet all requirements.", type: "error" }); return;
        }
        setLoading(true); setMsg({ text: "", type: "" });
        try {
            const res = await fetch("/api/reset-password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, otp, new_password: newPw }),
            });
            const data = await res.json();
            if (data.success) {
                setMsg({ text: "Password reset! Redirecting to login…", type: "success" });
                setTimeout(() => navigate("/login"), 2000);
            } else {
                setMsg({ text: data.message || "Reset failed.", type: "error" });
            }
        } catch {
            setMsg({ text: "Network error. Please try again.", type: "error" });
        } finally { setLoading(false); }
    };

    const inputCls = "w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-green-400/50 focus:border-green-400/60 text-sm transition";
    const btnCls = `w-full py-3.5 rounded-xl font-semibold text-white transition-all shadow-lg text-sm ${loading ? "bg-gray-500 cursor-not-allowed" : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500"}`;

    return (
        <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-green-900 to-zinc-900 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Card */}
                <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-3xl p-8 shadow-2xl">
                    <div className="text-center mb-7">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center mx-auto mb-4 shadow-lg">
                            <span className="text-2xl">🔑</span>
                        </div>
                        <h1 className="text-2xl font-bold text-white mb-1">
                            {step === "email" ? "Forgot Password?" : "Reset Password"}
                        </h1>
                        <p className="text-white/50 text-sm">
                            {step === "email"
                                ? "Enter your email to receive a reset code."
                                : `Enter the reset code sent to ${email} and your new password.`}
                        </p>
                    </div>

                    {/* Message */}
                    {msg.text && (
                        <div className={`mb-5 px-4 py-3 rounded-xl text-sm font-medium ${msg.type === "success"
                                ? "bg-green-500/20 border border-green-400/30 text-green-300"
                                : "bg-red-500/20 border border-red-400/30 text-red-300"
                            }`}>
                            {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                        </div>
                    )}

                    {/* Step 1 — Email */}
                    {step === "email" && (
                        <form onSubmit={sendOtp} className="space-y-4">
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Email Address</label>
                                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com" required className={inputCls} />
                            </div>
                            <button type="submit" disabled={loading} className={btnCls}>
                                {loading ? "Sending…" : "Send Reset Code"}
                            </button>
                        </form>
                    )}

                    {/* Step 2 — OTP + New Password */}
                    {step === "otp" && (
                        <form onSubmit={resetPassword} className="space-y-4">
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">6-Digit Reset Code</label>
                                <input type="text" value={otp} onChange={(e) => setOtp(e.target.value)}
                                    placeholder="123456" maxLength={6} required className={inputCls} />
                            </div>
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">New Password</label>
                                <input type="password" value={newPw}
                                    onChange={(e) => setNewPw(e.target.value)}
                                    onFocus={() => setShowPwHint(true)}
                                    placeholder="••••••••" required className={inputCls} />
                                {showPwHint && newPw && (
                                    <div className="mt-2 space-y-1">
                                        {/* Strength bar */}
                                        <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                                            <div className="h-full rounded-full transition-all duration-300"
                                                style={{ width: `${(pwScore / 5) * 100}%`, backgroundColor: pwColor }} />
                                        </div>
                                        {pwRules.map((r, i) => (
                                            <p key={i} className={`text-xs flex items-center gap-1.5 ${r.test(newPw) ? "text-green-400" : "text-white/40"}`}>
                                                {r.test(newPw) ? "✔" : "○"} {r.label}
                                            </p>
                                        ))}
                                    </div>
                                )}
                            </div>
                            <div>
                                <label className="block text-sm text-white/60 mb-1.5 font-medium">Confirm New Password</label>
                                <input type="password" value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)}
                                    placeholder="••••••••" required className={inputCls} />
                            </div>
                            <button type="submit" disabled={loading} className={btnCls}>
                                {loading ? "Resetting…" : "Reset Password"}
                            </button>
                            <button type="button" onClick={() => setStep("email")}
                                className="w-full py-2 text-white/50 hover:text-white text-sm transition">
                                ← Try a different email
                            </button>
                        </form>
                    )}

                    <p className="text-center text-sm text-white/40 mt-5">
                        Remembered it?{" "}
                        <Link to="/login" className="text-emerald-400 hover:text-emerald-300 font-medium transition">
                            Back to Login
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
