import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setError("");
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            const res = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify(formData),
            });
            const data = await res.json();
            if (res.ok && data.success) {
                localStorage.setItem("user", JSON.stringify(data.user));
                // Route based on role
                if (data.user.role === "admin") {
                    navigate("/admin-dashboard");
                } else if (data.user.role === "superadmin") {
                    navigate("/super-admin-dashboard");
                } else {
                    navigate("/dashboard");
                }
            } else if (res.status === 403 && data.needs_otp) {
                // User exists but not verified
                navigate(`/verify-otp?email=${encodeURIComponent(data.email)}`);
            } else {
                setError(data.message || "Login failed");
            }
        } catch (err) {
            console.error("Login error:", err);
            setError("Could not connect to server");
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
        @keyframes shimmerBg {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
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
                {/* Full-screen dark overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-green-950/70 via-emerald-900/60 to-green-950/80 z-0" />

                {/* ── LEFT PANEL — branding ── */}
                <div className="hidden lg:flex flex-col justify-between w-[50%] relative z-10 p-12">
                    {/* Top branding */}
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
                            Empowering<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-300">
                                Pakistan's Agriculture
                            </span>
                            <br />with AI
                        </h2>

                        <p className="text-white/60 text-lg leading-relaxed max-w-md">
                            Access crop intelligence, soil analysis, irrigation planning, and AI-powered advisory — all in one platform.
                        </p>
                    </div>

                    {/* Bottom stats */}
                    <div className="animate-float-in-delay flex gap-8">
                        {[
                            { label: "Active Farmers", value: "54+" },
                            { label: "Crop Resources", value: "128+" },
                            { label: "AI Responses", value: "10K+" },
                        ].map((stat, i) => (
                            <div key={i}>
                                <p className="text-2xl font-bold text-white">{stat.value}</p>
                                <p className="text-xs text-emerald-400/60 uppercase tracking-wider">{stat.label}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* ── RIGHT PANEL — login form ── */}
                <div className="flex-1 flex items-center justify-center relative z-10 p-8">
                    <div className="w-full max-w-md animate-float-in-delay">

                        {/* Card */}
                        <div className="bg-white/95 backdrop-blur-2xl rounded-3xl shadow-2xl p-10 border border-white/50">

                            {/* Mobile logo */}
                            <div className="lg:hidden flex items-center gap-2.5 mb-8">
                                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-lg shadow-md">🌿</div>
                                <h2 className="text-lg font-bold text-green-800">AgriChat</h2>
                            </div>

                            <h3 className="text-2xl font-bold text-gray-800 mb-1">Welcome back</h3>
                            <p className="text-gray-500 text-sm mb-8">Sign in to continue to your dashboard</p>

                            <form onSubmit={handleSubmit} className="space-y-5">
                                {/* Email */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1.5">Email Address</label>
                                    <input
                                        type="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        required
                                        placeholder="you@example.com"
                                        className="w-full px-4 py-3 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition"
                                    />
                                </div>

                                {/* Password */}
                                <div>
                                    <div className="flex justify-between items-center mb-1.5">
                                        <label className="text-sm font-medium text-gray-700">Password</label>
                                        <Link to="/forgot-password" className="text-xs text-green-600 hover:text-green-700 font-medium transition">
                                            Forgot password?
                                        </Link>
                                    </div>
                                    <div className="relative">
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            name="password"
                                            value={formData.password}
                                            onChange={handleChange}
                                            required
                                            placeholder="••••••••"
                                            className="w-full px-4 py-3 rounded-xl bg-gray-50 border border-gray-200 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-400 text-sm transition pr-12"
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowPassword(!showPassword)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 text-sm"
                                        >
                                            {showPassword ? "🙈" : "👁️"}
                                        </button>
                                    </div>
                                </div>

                                {/* Remember me */}
                                <div className="flex items-center gap-2">
                                    <input type="checkbox" id="remember" className="w-4 h-4 rounded border-gray-300 text-green-600 focus:ring-green-500" />
                                    <label htmlFor="remember" className="text-sm text-gray-600">Remember me</label>
                                </div>

                                {/* Submit */}
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className={`w-full py-3 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg ${loading
                                        ? "bg-gray-400 cursor-not-allowed"
                                        : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:shadow-green-500/30 hover:scale-[1.01]"
                                        }`}
                                >
                                    {loading ? "Signing in…" : "Sign In"}
                                </button>
                            </form>

                            {/* Divider */}
                            <div className="flex items-center gap-3 my-6">
                                <div className="flex-1 h-px bg-gray-200" />
                                <span className="text-xs text-gray-400 uppercase tracking-wider">or</span>
                                <div className="flex-1 h-px bg-gray-200" />
                            </div>

                            {/* Urdu toggle */}
                            <button className="w-full py-2.5 rounded-xl border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition flex items-center justify-center gap-2">
                                🌐 Switch to Urdu (اردو)
                            </button>

                            {/* Sign up link */}
                            <p className="text-center text-sm text-gray-500 mt-6">
                                Don't have an account?{" "}
                                <Link to="/signup" className="text-green-600 font-semibold hover:text-green-700 transition">
                                    Create Account
                                </Link>
                            </p>
                        </div>

                        {/* Footer */}
                        <p className="text-center text-xs text-white/40 mt-6">
                            © 2026 AgriChat · PQNK Agriculture Repository
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}
