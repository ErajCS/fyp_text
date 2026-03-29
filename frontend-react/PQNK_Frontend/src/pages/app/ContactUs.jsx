import { useState } from "react";
import DashboardLayout from "../../layouts/DashboardLayout";

const CONTACT_INFO = [
    {
        icon: "🏛️",
        label: "Affiliated Institution",
        value: "Habib University — Dhanani School of Science & Engineering",
        sub: "Karachi, Pakistan",
    },
    {
        icon: "🌿",
        label: "Industry Partner",
        value: "Pakistan Agriculture Research (PAR)",
        sub: "Pedaver — The Transformative Producer",
    },
    {
        icon: "📧",
        label: "Research Enquiries",
        value: "pqnk@habib.edu.pk",
        sub: "Response within 2–3 business days",
    },
    {
        icon: "🎓",
        label: "Academic Conference",
        value: "DURS 2026 — AI for Transforming Agriculture",
        sub: "Dhanani Undergraduate Research Symposium",
    },
];

export default function ContactUs() {
    const [form, setForm] = useState({
        firstName: "", lastName: "", email: "", phone: "", subject: "", message: "",
    });
    const [submitted, setSubmitted] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSubmitting(true);
        // Simulate submission delay (no backend endpoint needed — could be hooked up later)
        await new Promise((r) => setTimeout(r, 1200));
        setSubmitting(false);
        setSubmitted(true);
    };

    if (submitted) {
        return (
            <DashboardLayout>
                <div className="flex flex-col items-center justify-center min-h-[60vh]" style={{ animation: "fadeUp .5s ease-out" }}>
                    <style>{`@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}`}</style>
                    <div className="w-24 h-24 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-5xl mb-6 shadow-2xl shadow-green-500/30">
                        ✅
                    </div>
                    <h2 className="text-3xl font-bold text-white mb-3">Message Sent!</h2>
                    <p className="text-white/60 text-base text-center max-w-md mb-8 leading-relaxed">
                        Thank you for reaching out. Our team will review your message and respond within 2–3 business days.
                    </p>
                    <button
                        onClick={() => { setSubmitted(false); setForm({ firstName: "", lastName: "", email: "", phone: "", subject: "", message: "" }); }}
                        className="px-8 py-3 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 text-white font-semibold shadow-lg transition-all hover:scale-[1.02]"
                    >
                        Send Another Message
                    </button>
                </div>
            </DashboardLayout>
        );
    }

    return (
        <DashboardLayout>
            <>
                <style>{`
          @keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
          .fu{animation:fadeUp .55s ease-out both;}
          .fu1{animation:fadeUp .55s ease-out .08s both;}
          .fu2{animation:fadeUp .55s ease-out .16s both;}
          .select-dark option { background-color: #064e3b; color: white; }
          .input-field{ width:100%; padding:12px 16px; border-radius:12px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:white; font-size:14px; outline:none; transition:border-color .2s, box-shadow .2s; }
          .input-field::placeholder{ color:rgba(255,255,255,0.3); }
          .input-field:focus{ border-color:rgba(52,211,153,0.6); box-shadow:0 0 0 3px rgba(52,211,153,0.12); }
        `}</style>

                {/* ── PAGE HEADER ── */}
                <div className="fu mb-10">
                    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-900/90 to-green-950/95 border border-emerald-500/20 shadow-2xl p-10">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(52,211,153,0.1),_transparent_60%)]" />
                        <div className="relative z-10">
                            <div className="inline-flex items-center gap-2 bg-emerald-500/15 border border-emerald-400/25 rounded-full px-4 py-1.5 text-emerald-300 text-xs font-semibold tracking-widest uppercase mb-4">
                                📬 Contact Us
                            </div>
                            <h1 className="text-4xl font-extrabold text-white mb-3">Get in Touch</h1>
                            <p className="text-white/55 text-base max-w-xl leading-relaxed">
                                Have questions about PQNK, the platform, collaboration opportunities, or research enquiries? We'd love to
                                hear from you — let's start a conversation.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="fu1 grid lg:grid-cols-3 gap-8">
                    {/* ── LEFT: Contact Info ── */}
                    <div className="space-y-4">
                        <h2 className="text-lg font-bold text-white mb-5 flex items-center gap-2">
                            <span className="w-1 h-6 rounded-full bg-emerald-400 inline-block" /> Contact Information
                        </h2>
                        {CONTACT_INFO.map((item, i) => (
                            <div
                                key={i}
                                className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-5 shadow-md hover:bg-white/15 transition-colors"
                            >
                                <div className="flex items-start gap-4">
                                    <div className="w-10 h-10 rounded-xl bg-emerald-700/50 flex items-center justify-center text-xl flex-shrink-0">
                                        {item.icon}
                                    </div>
                                    <div>
                                        <p className="text-white/40 text-xs font-semibold uppercase tracking-wider mb-0.5">{item.label}</p>
                                        <p className="text-white text-sm font-semibold">{item.value}</p>
                                        <p className="text-emerald-400/70 text-xs mt-0.5">{item.sub}</p>
                                    </div>
                                </div>
                            </div>
                        ))}

                        {/* Social/Web Links */}
                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
                            <p className="text-white/40 text-xs font-semibold uppercase tracking-wider mb-3">Follow Our Research</p>
                            <div className="space-y-2">
                                {[
                                    { icon: "🌐", label: "pedaver.com", url: "https://www.pedaver.com" },
                                    { icon: "📘", label: "facebook/pedaver", url: "https://facebook.com/pedaver" },
                                    { icon: "🐙", label: "github.com/ErajCS/fyp_text", url: "#" },
                                ].map((link, i) => (
                                    <a
                                        key={i}
                                        href={link.url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="flex items-center gap-3 py-2 text-white/55 hover:text-emerald-300 transition-colors text-sm"
                                    >
                                        <span>{link.icon}</span>
                                        <span>{link.label}</span>
                                    </a>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* ── RIGHT: Contact Form ── */}
                    <div className="lg:col-span-2 fu2">
                        <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-3xl p-8 shadow-2xl">
                            <h2 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
                                <span className="w-1 h-6 rounded-full bg-emerald-400 inline-block" /> Send Us a Message
                            </h2>
                            <p className="text-white/40 text-sm mb-7">All fields marked with * are required.</p>

                            <form onSubmit={handleSubmit} className="space-y-5">
                                {/* Name Row */}
                                <div className="grid sm:grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                            First Name *
                                        </label>
                                        <input
                                            className="input-field"
                                            name="firstName"
                                            value={form.firstName}
                                            onChange={handleChange}
                                            placeholder="First Name"
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                            Last Name *
                                        </label>
                                        <input
                                            className="input-field"
                                            name="lastName"
                                            value={form.lastName}
                                            onChange={handleChange}
                                            placeholder="Last Name"
                                            required
                                        />
                                    </div>
                                </div>

                                {/* Email */}
                                <div>
                                    <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                        Email Address *
                                    </label>
                                    <input
                                        className="input-field"
                                        name="email"
                                        type="email"
                                        value={form.email}
                                        onChange={handleChange}
                                        placeholder="your@email.com"
                                        required
                                    />
                                </div>

                                {/* Phone */}
                                <div>
                                    <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                        Phone Number
                                    </label>
                                    <input
                                        className="input-field"
                                        name="phone"
                                        type="tel"
                                        value={form.phone}
                                        onChange={handleChange}
                                        placeholder="+92 300 0000000"
                                    />
                                </div>

                                {/* Subject */}
                                <div>
                                    <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                        Subject *
                                    </label>
                                    <select
                                        className="input-field select-dark"
                                        name="subject"
                                        value={form.subject}
                                        onChange={handleChange}
                                        required
                                        style={{ appearance: "none" }}
                                    >
                                        <option value="" disabled>Select a topic…</option>
                                        <option value="general">General Enquiry</option>
                                        <option value="pqnk">PQNK Farming Practices</option>
                                        <option value="research">Research &amp; Academic Collaboration</option>
                                        <option value="technical">Platform / Technical Support</option>
                                        <option value="media">Media &amp; Press</option>
                                        <option value="other">Other</option>
                                    </select>
                                </div>

                                {/* Message */}
                                <div>
                                    <label className="block text-white/60 text-xs font-semibold uppercase tracking-wide mb-2">
                                        Message *
                                    </label>
                                    <textarea
                                        className="input-field resize-none"
                                        name="message"
                                        rows={5}
                                        value={form.message}
                                        onChange={handleChange}
                                        placeholder="Write your message here…"
                                        required
                                    />
                                </div>

                                {/* Submit */}
                                <button
                                    type="submit"
                                    disabled={submitting}
                                    className={`w-full py-3.5 rounded-xl font-bold text-white text-sm shadow-lg transition-all duration-200 ${submitting
                                        ? "bg-gray-500 cursor-not-allowed"
                                        : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 hover:scale-[1.01] hover:shadow-green-500/25"
                                        }`}
                                >
                                    {submitting ? (
                                        <span className="flex items-center justify-center gap-2">
                                            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><circle cx="12" cy="12" r="10" strokeOpacity={0.25} /><path d="M12 2a10 10 0 0 1 10 10" /></svg>
                                            Sending…
                                        </span>
                                    ) : (
                                        "📤 Send Message"
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </>
        </DashboardLayout>
    );
}
