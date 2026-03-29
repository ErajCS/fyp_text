import { useNavigate } from "react-router-dom";

const QUICK_LINKS = [
    { label: "Dashboard", path: "/dashboard" },
    { label: "AI Assistant", path: "/chatbot" },
    { label: "Browse Repository", path: "/browse-repository" },
    { label: "About Us", path: "/about" },
    { label: "Contact Us", path: "/contact" },
    { label: "My Profile", path: "/profile" },
];

const PARTNERS = [
    { name: "Pakistan Agriculture Research", abbr: "PAR" },
    { name: "Habib University — DSSE", abbr: "HU" },
    { name: "Pedaver", abbr: "PV" },
];

export default function Footer() {
    const navigate = useNavigate();
    const year = new Date().getFullYear();

    return (
        <footer className="mt-auto border-t border-white/10 bg-emerald-950/80 backdrop-blur-xl">
            <div className="max-w-[1400px] mx-auto px-8 lg:px-12 py-12 lg:py-16">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8 mb-12">

                    {/* Brand */}
                    <div className="md:col-span-2 lg:pr-20">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-xl shadow-xl shadow-green-500/20">
                                🌿
                            </div>
                            <div>
                                <h3 className="text-white font-extrabold text-2xl tracking-tight leading-none">AgriChat</h3>
                                <p className="text-emerald-400 text-[10px] tracking-[0.2em] uppercase font-bold mt-1.5 opacity-80">PQNK Knowledge System</p>
                            </div>
                        </div>
                        <p className="text-white/50 text-sm leading-relaxed mb-8 max-w-md">
                            An AI-powered expert knowledge repository for sustainable natural farming in
                            Pakistan. Grounded exclusively in Dr. Asif Sharif's PQNK corpus — hallucination-free, bilingual, and
                            accessible to every farmer across the nation.
                        </p>
                        <div className="flex flex-wrap gap-2">
                            {["RAG · GPT-4o", "Urdu + English", "Zero Chemicals", "Open Access"].map((tag) => (
                                <span
                                    key={tag}
                                    className="text-[10px] font-semibold tracking-wide text-emerald-300 border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 rounded-full whitespace-nowrap"
                                >
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div className="flex flex-col">
                        <h4 className="text-white/80 text-xs font-black uppercase tracking-[0.15em] mb-7">Quick Navigation</h4>
                        <ul className="grid grid-cols-1 gap-y-3.5">
                            {QUICK_LINKS.map((link) => (
                                <li key={link.path}>
                                    <button
                                        onClick={() => navigate(link.path)}
                                        className="group flex items-center gap-2 text-white/45 hover:text-emerald-400 text-[13px] transition-all duration-200"
                                    >
                                        <span className="w-1 h-1 rounded-full bg-emerald-500/0 group-hover:bg-emerald-500 transition-all" />
                                        {link.label}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Partners & Awards */}
                    <div className="flex flex-col">
                        <h4 className="text-white/80 text-xs font-black uppercase tracking-[0.15em] mb-7">Partners & Faculty</h4>
                        <div className="space-y-4 mb-8">
                            {PARTNERS.map((p) => (
                                <div key={p.name} className="flex items-center gap-3 group">
                                    <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-[10px] font-black text-emerald-500/60 group-hover:text-emerald-400 transition-colors flex-shrink-0">
                                        {p.abbr}
                                    </div>
                                    <p className="text-white/40 text-[11px] leading-tight group-hover:text-white/60 transition-colors uppercase tracking-wider">{p.name}</p>
                                </div>
                            ))}
                        </div>

                        <div className="relative group overflow-hidden bg-gradient-to-br from-amber-500/10 to-transparent border border-amber-500/20 rounded-2xl p-5 transition-all hover:border-amber-500/40">
                            <div className="absolute top-0 right-0 p-2 opacity-20 group-hover:opacity-40 transition-opacity">🏆</div>
                            <p className="text-amber-400 text-[11px] font-black tracking-widest uppercase mb-1">DURS 2026 Nomination</p>
                            <p className="text-white/60 text-xs leading-snug">AI for Transforming Sustainable Agriculture</p>
                            <div className="mt-3 pt-3 border-t border-amber-500/10">
                                <p className="text-white/30 text-[10px] italic">Prototypes for Humanity — Dubai 2026</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="border-t border-white/5 pt-8 flex flex-col lg:flex-row items-center justify-between gap-6">
                    <div className="flex flex-col items-center lg:items-start gap-1">
                        <p className="text-white/20 text-[11px] tracking-wide uppercase">
                            © {year} PQNK Knowledge Intelligence System
                        </p>
                        <p className="text-white/15 text-[10px]">
                            Habib University — DSSE, Karachi, Pakistan.
                        </p>
                    </div>

                    <div className="flex items-center gap-8">
                        <a
                            href="https://www.pedaver.com"
                            target="_blank"
                            rel="noreferrer"
                            className="text-white/25 hover:text-emerald-400 text-xs font-medium transition-all hover:tracking-widest"
                        >
                            pedaver.com
                        </a>
                        <div className="h-4 w-px bg-white/10 hidden sm:block" />
                        <p className="text-white/20 text-xs flex items-center gap-1.5">
                            Built with <span className="text-emerald-500/60 animate-pulse">❤️</span> at Habib University
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
