import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams, Link } from "react-router-dom";

export default function VerifyOtp() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const email = searchParams.get("email") || "";

    const [otp, setOtp] = useState(["", "", "", "", "", ""]);
    const [loading, setLoading] = useState(false);
    const [resending, setResending] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");
    const [countdown, setCountdown] = useState(60);
    const inputRefs = useRef([]);

    // Countdown timer to enable resend button
    useEffect(() => {
        if (countdown <= 0) return;
        const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
        return () => clearTimeout(t);
    }, [countdown]);

    // Auto-focus first input
    useEffect(() => {
        inputRefs.current[0]?.focus();
    }, []);

    const handleOtpChange = (index, value) => {
        if (!/^\d?$/.test(value)) return; // digits only
        const newOtp = [...otp];
        newOtp[index] = value;
        setOtp(newOtp);
        setError("");

        // Auto-advance
        if (value && index < 5) {
            inputRefs.current[index + 1]?.focus();
        }
    };

    const handleKeyDown = (index, e) => {
        if (e.key === "Backspace" && !otp[index] && index > 0) {
            inputRefs.current[index - 1]?.focus();
        }
    };

    const handlePaste = (e) => {
        e.preventDefault();
        const text = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
        if (text.length === 6) {
            setOtp(text.split(""));
            inputRefs.current[5]?.focus();
        }
    };

    const handleVerify = async () => {
        const code = otp.join("");
        if (code.length !== 6) {
            setError("Please enter the complete 6-digit OTP");
            return;
        }

        setLoading(true);
        setError("");
        try {
            const res = await fetch("/api/verify-otp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ email, otp: code }),
            });
            const data = await res.json();
            if (data.success) {
                setSuccess("✅ Email verified! Redirecting to login…");
                setTimeout(() => navigate("/login"), 2000);
            } else {
                setError(data.message || "Verification failed");
            }
        } catch {
            setError("Could not connect to server");
        } finally {
            setLoading(false);
        }
    };

    const handleResend = async () => {
        if (countdown > 0) return;
        setResending(true);
        setError("");
        try {
            const res = await fetch("/api/resend-otp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ email }),
            });
            const data = await res.json();
            if (data.success) {
                setSuccess("A new OTP has been sent to your email.");
                setOtp(["", "", "", "", "", ""]);
                setCountdown(60);
                inputRefs.current[0]?.focus();
            } else {
                setError(data.message || "Could not resend OTP");
            }
        } catch {
            setError("Could not connect to server");
        } finally {
            setResending(false);
        }
    };

    return (
        <>
            <style>{`
        @keyframes floatIn {
          from { opacity: 0; transform: translateY(24px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseRing {
          0%, 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.4); }
          50% { box-shadow: 0 0 0 12px rgba(52, 211, 153, 0); }
        }
        .animate-float-in { animation: floatIn 0.6s ease-out both; }
        .otp-input:focus { animation: pulseRing 1.5s ease-in-out infinite; }
      `}</style>

            <div
                className="min-h-screen flex items-center justify-center relative overflow-hidden"
                style={{
                    backgroundImage: "url('/agri-bg.png')",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                }}
            >
                <div className="absolute inset-0 bg-gradient-to-br from-green-950/80 via-emerald-900/70 to-green-950/85 z-0" />

                <div className="relative z-10 w-full max-w-md px-6 animate-float-in">
                    {/* Logo */}
                    <div className="text-center mb-8">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-green-400 to-emerald-500 text-3xl shadow-2xl shadow-green-500/40 mb-4">
                            🌿
                        </div>
                        <h1 className="text-2xl font-bold text-white">AgriChat</h1>
                        <p className="text-emerald-400/70 text-xs tracking-widest uppercase">PQNK Knowledge System</p>
                    </div>

                    <div className="bg-white/95 backdrop-blur-2xl rounded-3xl shadow-2xl p-9 border border-white/50">

                        {/* Header */}
                        <div className="text-center mb-8">
                            <div className="w-14 h-14 rounded-full bg-emerald-50 border-2 border-emerald-200 flex items-center justify-center text-2xl mx-auto mb-4">
                                📧
                            </div>
                            <h2 className="text-2xl font-bold text-gray-800 mb-2">Verify Your Email</h2>
                            <p className="text-gray-500 text-sm leading-relaxed">
                                We sent a 6-digit verification code to
                            </p>
                            <p className="text-emerald-700 font-semibold text-sm mt-1 break-all">{email}</p>
                            <p className="text-gray-400 text-xs mt-2">Check your inbox (and spam folder)</p>
                        </div>

                        {/* Success message */}
                        {success && (
                            <div className="mb-5 px-4 py-3 bg-green-50 border border-green-200 text-green-700 rounded-xl text-sm text-center font-medium">
                                {success}
                            </div>
                        )}

                        {/* Error message */}
                        {error && (
                            <div className="mb-5 px-4 py-3 bg-red-50 border border-red-200 text-red-700 rounded-xl text-sm flex items-start gap-2">
                                <span className="flex-shrink-0">⚠️</span> {error}
                            </div>
                        )}

                        {/* OTP Input Boxes */}
                        <div className="flex gap-3 justify-center mb-6" onPaste={handlePaste}>
                            {otp.map((digit, i) => (
                                <input
                                    key={i}
                                    ref={(el) => (inputRefs.current[i] = el)}
                                    type="text"
                                    inputMode="numeric"
                                    maxLength={1}
                                    value={digit}
                                    onChange={(e) => handleOtpChange(i, e.target.value)}
                                    onKeyDown={(e) => handleKeyDown(i, e)}
                                    className="otp-input w-12 h-14 text-center text-xl font-bold rounded-xl border-2 border-gray-200 bg-gray-50 text-gray-800 focus:outline-none focus:border-emerald-400 focus:bg-white transition-all"
                                />
                            ))}
                        </div>

                        {/* Verify button */}
                        <button
                            onClick={handleVerify}
                            disabled={loading || otp.join("").length !== 6}
                            className={`w-full py-3.5 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg mb-4 ${loading || otp.join("").length !== 6
                                ? "bg-gray-300 cursor-not-allowed"
                                : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/30 hover:scale-[1.01]"
                                }`}
                        >
                            {loading ? "Verifying…" : "Verify Email"}
                        </button>

                        {/* Resend */}
                        <div className="text-center">
                            <p className="text-gray-500 text-sm mb-2">Didn't receive the code?</p>
                            {countdown > 0 ? (
                                <p className="text-gray-400 text-sm">
                                    Resend in <span className="font-semibold text-emerald-600">{countdown}s</span>
                                </p>
                            ) : (
                                <button
                                    onClick={handleResend}
                                    disabled={resending}
                                    className="text-emerald-600 font-semibold text-sm hover:text-emerald-700 transition disabled:opacity-50"
                                >
                                    {resending ? "Sending…" : "Resend OTP"}
                                </button>
                            )}
                        </div>

                        {/* Back to signup */}
                        <div className="border-t border-gray-100 mt-6 pt-5 text-center">
                            <Link to="/signup" className="text-xs text-gray-400 hover:text-gray-600 transition">
                                ← Back to Sign Up
                            </Link>
                        </div>
                    </div>

                    <p className="text-center text-xs text-white/40 mt-6">
                        © 2026 AgriChat · PQNK Agriculture Repository
                    </p>
                </div>
            </div>
        </>
    );
}
