import DashboardLayout from "../../layouts/DashboardLayout";
import { useLanguage, TRANSLATIONS } from "../../context/LanguageContext";

export default function AboutUs() {
    const { lang } = useLanguage();
    const t = (key) => TRANSLATIONS[lang][key] || key;
    const isRtl = lang === "ur";

    const PQNK_PRACTICES = [
        { icon: "🌾", title: t("aboutPractice0Title"), desc: t("aboutPractice0Desc") },
        { icon: "🛏️", title: t("aboutPractice1Title"), desc: t("aboutPractice1Desc") },
        { icon: "🍂", title: t("aboutPractice2Title"), desc: t("aboutPractice2Desc") },
        { icon: "📐", title: t("aboutPractice3Title"), desc: t("aboutPractice3Desc") },
        { icon: "🌳", title: t("aboutPractice4Title"), desc: t("aboutPractice4Desc") },
        { icon: "🚜", title: t("aboutPractice5Title"), desc: t("aboutPractice5Desc") },
    ];

    const TIMELINE = [
        { year: "1980s", event: lang === "en" ? "Began agricultural engineering career, designing innovative machinery for Pakistani farms" : "زرعی انجینئرنگ کے کیریئر کا آغاز، پاکستانی فارمز کے لیے جدید مشینری ڈیزائن کی" },
        { year: "1990s", event: lang === "en" ? "Introduced global agribusiness brands to Pakistan — Ford New Holland, Case IH, Sperry, and others" : "پاکستان میں عالمی زرعی برانڈز متعارف کروائے — فورڈ، کیس آئی ایچ اور دیگر" },
        { year: "2000s", event: lang === "en" ? "Developed raised-bed rice cultivation achieving 70% water-use reduction" : "اٹھائے ہوئے بیڈز پر چاول کی کاشت تیار کی جس سے پانی کے استعمال میں 70٪ کمی آئی" },
        { year: "2005", event: lang === "en" ? "Launched the 'One Acre Prosperity' (OAP) initiative to empower smallholder farmers" : "'ون ایکڑ پراسپیرٹی' (OAP) اقدام کا آغاز کیا تاکہ چھوٹے کسانوں کو بااختیار بنایا جا سکے" },
        { year: "2010s", event: lang === "en" ? "Formulated and refined PQNK — the complete no-till, no-agrichemical sustainable farming system" : "PQNK کی تشکیل اور بہتری — مکمل بغیر ہل چلائے (no-till) اور بغیر کیمیکل والا پائیدار زرعی نظام" },
        { year: "2020s", event: lang === "en" ? "Founded Pedaver to scale the PQNK system with precision machinery and field implementation" : "پداور (Pedaver) کی بنیاد رکھی تاکہ مشینری کے ذریعے PQNK کو بڑے پیمانے پر پھیلایا جا سکے" },
        { year: "2026", event: lang === "en" ? "PQNK Knowledge Intelligence System — AI-powered platform built with Habib University & PAR" : "PQNK نالج انٹیلی جنس سسٹم — حبیب یونیورسٹی اور PAR کے ساتھ مل کر بنایا گیا AI پلیٹ فارم" },
    ];

    const CORE_PRINCIPLES = [
        { icon: "🚫", label: t("aboutPrinciple0"), desc: t("aboutPrinciple0Desc") },
        { icon: "🛏️", label: t("aboutPrinciple1"), desc: t("aboutPrinciple1Desc") },
        { icon: "💧", label: t("aboutPrinciple2"), desc: t("aboutPrinciple2Desc") },
        { icon: "🍂", label: t("aboutPrinciple3"), desc: t("aboutPrinciple3Desc") },
        { icon: "📐", label: t("aboutPrinciple4"), desc: t("aboutPrinciple4Desc") },
        { icon: "🌳", label: t("aboutPrinciple5"), desc: t("aboutPrinciple5Desc") },
    ];

    const MISSIONS = [
        { icon: "🌍", title: t("aboutMission0Title"), desc: t("aboutMission0Desc") },
        { icon: "🤖", title: t("aboutMission1Title"), desc: t("aboutMission1Desc") },
        { icon: "🌱", title: t("aboutMission2Title"), desc: t("aboutMission2Desc") },
    ];

    return (
        <DashboardLayout>
            <div dir={isRtl ? "rtl" : "ltr"} className={isRtl ? "font-urdu" : ""}>
                <style>{`
          @keyframes fadeUp { from {opacity:0;transform:translateY(22px);} to {opacity:1;transform:translateY(0);} }
          .fu  { animation: fadeUp .55s ease-out both; }
          .fu1 { animation: fadeUp .55s ease-out .08s both; }
          .fu2 { animation: fadeUp .55s ease-out .16s both; }
          .fu3 { animation: fadeUp .55s ease-out .24s both; }
          .fu4 { animation: fadeUp .55s ease-out .32s both; }
          .fu5 { animation: fadeUp .55s ease-out .40s both; }
          .card-hover { transition: transform 0.25s ease, box-shadow 0.25s ease; }
          .card-hover:hover { transform: translateY(-4px); box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
          .font-urdu { font-family: 'Noto Nastaliq Urdu', serif; }
        `}</style>

                {/* ── HERO ── */}
                <div className="fu mb-10">
                    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-900/90 to-green-950/95 border border-emerald-500/20 shadow-2xl p-10 lg:p-12">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(52,211,153,0.12),_transparent_60%)]" />
                        <div className="relative z-10 max-w-4xl">
                            <div className="inline-flex items-center gap-2 bg-emerald-500/15 border border-emerald-400/25 rounded-full px-5 py-2 text-emerald-300 text-xs font-semibold tracking-widest uppercase mb-6">
                                {t("aboutHeroBadge")}
                            </div>
                            <h1 className="text-4xl lg:text-5xl font-extrabold text-white leading-tight mb-5">
                                {t("heroTitle1")}{" "}
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-300">
                                    {t("heroTitle2")}
                                </span>
                            </h1>
                            <p className="text-white/65 text-lg lg:text-xl leading-relaxed mb-4">
                                {t("aboutHeroDesc1")}
                            </p>
                            <p className="text-white/55 text-base leading-relaxed">
                                {t("aboutHeroDesc2")}
                            </p>
                        </div>
                    </div>
                </div>

                {/* ── WHAT IS PQNK ── */}
                <div className="fu1 mb-12">
                    <div className={`flex items-center gap-3 mb-5 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">{t("aboutWhatIsTitle")}</h2>
                    </div>
                    <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 shadow-xl">
                        <div className="grid lg:grid-cols-2 gap-8 items-start">
                            <div className={`space-y-4 text-white/70 text-[15px] leading-relaxed ${isRtl ? "text-right" : ""}`}>
                                <p>{t("aboutWhatIsDesc1")}</p>
                                <p>{t("aboutWhatIsDesc2")}</p>
                                <p>{t("aboutWhatIsDesc3")}</p>
                                <p>{t("aboutWhatIsDesc4")}</p>
                                <p>{t("aboutWhatIsDesc5")}</p>
                            </div>
                            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                                <h3 className={`text-emerald-300 font-bold text-lg mb-4 ${isRtl ? "text-right" : ""}`}>{t("aboutCorePrinciplesTitle")}</h3>
                                <div className="space-y-3">
                                    {CORE_PRINCIPLES.map((item, i) => (
                                        <div key={i} className={`flex items-start gap-4 py-2 border-b border-white/5 last:border-0 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                                            <span className="text-xl">{item.icon}</span>
                                            <div>
                                                <p className="text-white font-semibold text-sm">{item.label}</p>
                                                <p className="text-white/45 text-xs">{item.desc}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── PQNK PRACTICES GRID ── */}
                <div className="fu2 mb-12">
                    <div className={`flex items-center gap-3 mb-5 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">{t("aboutPracticesTitle")}</h2>
                    </div>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
                        {PQNK_PRACTICES.map((p, i) => (
                            <div
                                key={i}
                                className={`card-hover bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-6 shadow-lg ${isRtl ? "text-right" : ""}`}
                            >
                                <div className="text-4xl mb-4">{p.icon}</div>
                                <h3 className="text-white font-bold text-base mb-3">{p.title}</h3>
                                <p className="text-white/55 text-sm leading-relaxed">{p.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* ── MISSION ── */}
                <div className="fu3 mb-12">
                    <div className="bg-gradient-to-r from-emerald-800/70 to-green-900/80 border border-emerald-500/25 backdrop-blur-xl rounded-2xl p-10 shadow-2xl">
                        <div className={`flex items-center gap-3 mb-6 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                            <div className="w-1 h-8 rounded-full bg-gradient-to-b from-amber-400 to-yellow-600" />
                            <h2 className="text-2xl font-bold text-white">{t("aboutMissionTitle")}</h2>
                        </div>
                        <div className="grid lg:grid-cols-3 gap-8">
                            {MISSIONS.map((m, i) => (
                                <div key={i} className="text-center">
                                    <div className="text-5xl mb-4">{m.icon}</div>
                                    <h3 className="text-white font-bold text-base mb-2">{m.title}</h3>
                                    <p className="text-white/55 text-sm leading-relaxed">{m.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* ── FOUNDER PROFILE ── */}
                <div className="fu4 mb-12">
                    <div className={`flex items-center gap-3 mb-6 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">{t("aboutFounderTitle")}</h2>
                    </div>

                    <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-3xl p-8 lg:p-10 shadow-2xl">
                        {/* Top Section — Name + Title */}
                        <div className={`flex flex-col lg:flex-row gap-8 items-center mb-8 pb-8 border-b border-white/10 ${isRtl ? "lg:flex-row-reverse" : ""}`}>
                            <div className="flex-shrink-0 text-center">
                                <div
                                    className="w-36 h-36 rounded-full bg-gradient-to-br from-emerald-500 to-green-700 flex items-center justify-center text-6xl mx-auto mb-4 shadow-2xl"
                                    style={{ boxShadow: "0 0 60px rgba(52,211,153,0.3)" }}
                                >
                                    👨‍🌾
                                </div>
                                <h3 className="text-white font-extrabold text-2xl">{t("aboutFounderName")}</h3>
                                <p className="text-emerald-400 text-sm font-semibold mt-1">{t("aboutFounderRole")}</p>
                                <p className="text-white/40 text-xs mt-0.5">{lang === "en" ? "PQNK Creator · Agricultural Pioneer" : "PQNK کے خالق · زرعی علمبردار"}</p>
                                <div className="flex flex-wrap justify-center gap-2 mt-4">
                                    {["Innovator", "Philanthropist", "Global Citizen", "Engineer"].map((tag) => (
                                        <span
                                            key={tag}
                                            className="bg-emerald-700/40 border border-emerald-500/25 text-emerald-300 text-[11px] px-3 py-1 rounded-full font-medium"
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className={`flex-1 ${isRtl ? "text-right" : ""}`}>
                                <div className="inline-flex items-center gap-2 bg-amber-500/15 border border-amber-400/20 rounded-full px-4 py-1.5 text-amber-300 text-xs font-semibold tracking-wider uppercase mb-4">
                                    🏆 {t("aboutFounderHeroTitle")}
                                </div>
                                <p className="text-white/70 text-[15px] leading-relaxed mb-4">
                                    {t("aboutFounderBio1")}
                                </p>
                                <p className="text-white/70 text-[15px] leading-relaxed">
                                    {t("aboutFounderBio2")}
                                </p>
                            </div>
                        </div>

                        {/* Main Sections */}
                        <div className="grid lg:grid-cols-2 gap-8 mb-8">
                            <div className={isRtl ? "text-right" : ""}>
                                <h4 className={`text-emerald-300 font-bold text-base mb-4 flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                    <span>⚙️</span> {t("aboutFounderMachineryTitle")}
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    {t("aboutFounderMachineryDesc1")}
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    {t("aboutFounderMachineryDesc2")}
                                </p>
                            </div>

                            <div className={isRtl ? "text-right" : ""}>
                                <h4 className={`text-emerald-300 font-bold text-base mb-4 flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                    <span>🌾</span> {t("aboutFounderInitiativesTitle")}
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    {t("aboutFounderInitiativesDesc1")}
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    {t("aboutFounderInitiativesDesc2")}
                                </p>
                            </div>
                        </div>

                        {/* Timeline */}
                        <div className="mb-8">
                            <h4 className={`text-white font-bold text-base mb-5 flex items-center gap-2 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                                <span>📅</span> {t("aboutTimelineTitle")}
                            </h4>
                            <div className={`relative ${isRtl ? "pr-6 border-r" : "pl-6 border-l"} border-emerald-500/30 space-y-5`}>
                                {TIMELINE.map((t, i) => (
                                    <div key={i} className="relative">
                                        <div className={`absolute ${isRtl ? "-right-[31px]" : "-left-[31px]"} w-4 h-4 rounded-full bg-emerald-500 border-2 border-emerald-300 shadow-lg shadow-emerald-500/40`} />
                                        <div className={`flex flex-col sm:flex-row sm:items-baseline gap-2 ${isRtl ? "sm:flex-row-reverse text-right" : ""}`}>
                                            <span className="text-emerald-400 font-bold text-sm w-20 flex-shrink-0">{t.year}</span>
                                            <p className="text-white/65 text-sm leading-relaxed">{t.event}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Business & Global Roles */}
                        <div className="grid lg:grid-cols-2 gap-8 pt-6 border-t border-white/10">
                            <div className={isRtl ? "text-right" : ""}>
                                <h4 className={`text-emerald-300 font-bold text-base mb-4 flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                    <span>🌐</span> {t("aboutBusinessTitle")}
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">{t("aboutBusinessDesc1")}</p>
                                <p className="text-white/60 text-sm leading-relaxed">{t("aboutBusinessDesc2")}</p>
                            </div>

                            <div className={isRtl ? "text-right" : ""}>
                                <h4 className={`text-emerald-300 font-bold text-base mb-4 flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                    <span>❤️</span> {t("aboutPhilanthropyTitle")}
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">{t("aboutPhilanthropyDesc1")}</p>
                                <p className="text-white/60 text-sm leading-relaxed">{t("aboutPhilanthropyDesc2")}</p>
                            </div>
                        </div>

                        {/* Quote */}
                        <div className={`mt-8 bg-emerald-900/60 border-emerald-400 rounded-3xl p-6 ${isRtl ? "border-r-4 text-right" : "border-l-4"}`}>
                            <p className="text-white/80 text-base italic leading-relaxed">
                                {lang === "en"
                                    ? '"Mr. Asif Sharif is an exemplary individual who has dedicated his life to improving agriculture and empowering communities. His contributions have had a profound impact on the lives of millions of people around the world."'
                                    : '"مسٹر آصف شریف ایک مثالی شخصیت ہیں جنہوں نے اپنی زندگی زراعت کی بہتری اور کمیونٹیز کو بااختیار بنانے کے لیے وقف کر دی ہے۔ ان کی خدمات نے دنیا بھر کے لاکھوں لوگوں کی زندگیوں پر گہرا اثر ڈالا ہے۔"'}
                            </p>
                            <p className="text-emerald-400 text-xs mt-2 font-semibold tracking-wide uppercase">
                                — Pedaver Profile Document, November 2023
                            </p>
                        </div>
                    </div>
                </div>

                {/* ── PARTNERSHIP STRIP ── */}
                <div className="fu5">
                    <div className={`bg-white/5 border border-white/10 rounded-2xl px-8 py-6 flex flex-wrap items-center justify-between gap-4 ${isRtl ? "flex-row-reverse" : ""}`}>
                        <div className={isRtl ? "text-right" : ""}>
                            <p className="text-white/40 text-xs uppercase tracking-widest mb-1 font-semibold">{t("aboutCollaborationTitle")}</p>
                            <div className={`flex items-center gap-6 flex-wrap ${isRtl ? "flex-row-reverse" : ""}`}>
                                {[
                                    { icon: "🏛️", name: lang === "en" ? "Pakistan Agriculture Research (PAR)" : "پاکستان ایگریکلچر ریسرچ (PAR)" },
                                    { icon: "🎓", name: lang === "en" ? "Habib University — DSSE" : "حبیب یونیورسٹی — ڈی ایس ایس ای" },
                                    { icon: "🌿", name: lang === "en" ? "Pedaver — The Transformative Producer" : "پداور — تبدیلی لانے والا پروڈیوسر" },
                                ].map((p, i) => (
                                    <div key={i} className={`flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
                                        <span>{p.icon}</span>
                                        <span className="text-white/60 text-sm font-medium">{p.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className={`hidden sm:block ${isRtl ? "text-left" : "text-right"}`}>
                            <p className="text-emerald-400/60 text-xs tracking-wider">DURS 2026</p>
                            <p className="text-white/40 text-xs">{lang === "en" ? "AI for Transforming Agriculture" : "زراعت کی تبدیلی کے لیے AI"}</p>
                        </div>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
}
