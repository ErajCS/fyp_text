import DashboardLayout from "../../layouts/DashboardLayout";

const PQNK_PRACTICES = [
    {
        icon: "🌾",
        title: "PQNK Agricultural Practices",
        desc: "Under PQNK, crops are cultivated on permanent no-till raised beds that are formed on levelled, non-compacted land. The beds' size is adapted to the size of the tractor and machinery wheels, so that the wheels are driving through the furrows, hence not compacting the beds. Beds are covered with organic mulch that can be locally procured by leaving previous crop residues on the field.",
        image: null,
    },
    {
        icon: "🛏️",
        title: "Why Raised Beds?",
        desc: "Growing crops on raised beds minimises soil disturbance to protect soil biota, improves soil structure, and promotes carbon sequestration. With machines developed to the size of the raised beds, PQNK facilitates the large-scale implementation of no-till. In PQNK, soil compaction is completely avoided by letting the tires of the heavy machinery drive in the furrows between the beds.",
        image: null,
    },
    {
        icon: "🍂",
        title: "Why Organic Mulch?",
        desc: "In a PQNK system, the layer of organic mulch should be thick enough to cover the soil surface and prevent sunlight from reaching the soil. This inhibits weed germination, resulting in a non-chemical strategy for weed management. Maintaining permanent biomass cover on the soil surface also protects the land from overheating in direct sunlight, which adversely affects the soil biota.",
        image: null,
    },
    {
        icon: "📐",
        title: "Why Optimise Plant Spacing?",
        desc: "Increasing the spacing between plants — reducing plant density per m² — should be done when soil is managed according to the principles of Conservation Agriculture (CA) as in PQNK systems. This is because it is natural for tillering crops to grow more profusely in the more fertile soil environment created by PQNK practices: more soil organic matter, more soil microbiome activity.",
        image: null,
    },
    {
        icon: "🌳",
        title: "Why Diversification?",
        desc: "Promoting diversification within the farming system increases biodiversity and results in greater land use efficiency and higher overall farm yield while enhancing carbon sequestration, especially when trees are included. Within a PQNK system, it is important to consider multiple strategies for achieving greater crop biodiversity — introducing cover crops between seasons, intercropping, alley cropping, relay cropping.",
        image: null,
    },
    {
        icon: "🚜",
        title: "What is Mechanisation?",
        desc: "Pedaver has developed machinery for mechanically forming raised beds and furrows, and for mechanised direct planting through heavy mulch with precise plant spacing. Transplanting of young single seedlings in permanent no-till raised beds with precise spacing is also possible thanks to Pedaver's innovations. Recent machinery even allows for forming the raised permanent beds while direct seeding in one single passage.",
        image: null,
    },
];

const TIMELINE = [
    { year: "1980s", event: "Began agricultural engineering career, designing innovative machinery for Pakistani farms" },
    { year: "1990s", event: "Introduced global agribusiness brands to Pakistan — Ford New Holland, Case IH, Sperry, and others" },
    { year: "2000s", event: "Developed raised-bed rice cultivation achieving 70% water-use reduction" },
    { year: "2005", event: "Launched the 'One Acre Prosperity' (OAP) initiative to empower smallholder farmers" },
    { year: "2010s", event: "Formulated and refined PQNK — the complete no-till, no-agrichemical sustainable farming system" },
    { year: "2020s", event: "Founded Pedaver to scale the PQNK system with precision machinery and field implementation" },
    { year: "2026", event: "PQNK Knowledge Intelligence System — AI-powered platform built with Habib University & PAR" },
];

export default function AboutUs() {
    return (
        <DashboardLayout>
            <>
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
        `}</style>

                {/* ── HERO ── */}
                <div className="fu mb-10">
                    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-900/90 to-green-950/95 border border-emerald-500/20 shadow-2xl p-12">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(52,211,153,0.12),_transparent_60%)]" />
                        <div className="absolute bottom-0 left-0 w-72 h-72 bg-emerald-500/5 rounded-full translate-y-1/2 -translate-x-1/4 blur-3xl" />
                        <div className="relative z-10 max-w-3xl">
                            <div className="inline-flex items-center gap-2 bg-emerald-500/15 border border-emerald-400/25 rounded-full px-5 py-2 text-emerald-300 text-xs font-semibold tracking-widest uppercase mb-6">
                                🌿 About PQNK
                            </div>
                            <h1 className="text-5xl font-extrabold text-white leading-tight mb-5">
                                Paedar Qudratti{" "}
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-300">
                                    Nizam-e-Kashtari
                                </span>
                            </h1>
                            <p className="text-white/65 text-xl leading-relaxed mb-4">
                                PQNK — pronounced{" "}
                                <em className="text-emerald-300 not-italic font-semibold">"picnic"</em> — is the abbreviation of{" "}
                                <strong className="text-white/90">Paedar Qudratti Nizam Kashatqari</strong>, meaning{" "}
                                <strong className="text-white/90">Sustainable Natural Farming System</strong> in Urdu.
                            </p>
                            <p className="text-white/55 text-base leading-relaxed">
                                It is also referred to as <em className="text-emerald-300">Paradoxical Agriculture</em> — a pioneering
                                methodology that replicates natural dynamics in agricultural production, eliminating synthetic chemicals
                                while achieving superior yields.
                            </p>
                        </div>
                    </div>
                </div>

                {/* ── WHAT IS PQNK ── */}
                <div className="fu1 mb-12">
                    <div className="flex items-center gap-3 mb-5">
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">What is PQNK?</h2>
                    </div>
                    <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-8 shadow-xl">
                        <div className="grid lg:grid-cols-2 gap-8 items-start">
                            <div className="space-y-4 text-white/70 text-[15px] leading-relaxed">
                                <p>
                                    PQNK is the abbreviation of <strong className="text-white">Paedar Qudratti Nizam Kashatqari</strong> —
                                    a sustainable natural farming system in Urdu, also referred to as{" "}
                                    <strong className="text-emerald-300">Paradoxical Agriculture</strong>. Mr. Asif Sharif has
                                    conceptualised PQNK through first-hand experience on his own land in the Indo-Gangetic Plains of
                                    Pakistan.
                                </p>
                                <p>
                                    He later founded <strong className="text-white">Pedaver</strong>, a company that works with thousands
                                    of farmers to disseminate the PQNK system and develop appropriate mechanisation. Initially, after
                                    identifying that inundation, soil disturbance, and bare land are contrary to the natural processes of
                                    soil fertility and vegetation, Mr. Asif developed the raised bed planting system.
                                </p>
                                <p>
                                    Rice was the first crop tested on Asif's raised beds — conventionally grown under flooded conditions.
                                    This was to demonstrate that water serves as a nutrient carrier, and only the water absorbed by the
                                    roots and transpired from the leaves is needed for plant development.
                                </p>
                                <p>
                                    At this stage, Asif gained insights into optimal plant population — row spacing and plant-to-plant
                                    distance. After achieving success by reducing rice plant density, the next objective was to maintain
                                    soil coverage. Rice crop residues had been utilised for this purpose.
                                </p>
                                <p>
                                    Soon PQNK developed into a more comprehensive farming method for all crops and trees that combines{" "}
                                    <strong className="text-white">SRI</strong> and{" "}
                                    <strong className="text-white">Conservation Agriculture (CA)</strong>, with natural farming as an
                                    overarching principle.
                                </p>
                            </div>
                            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                                <h3 className="text-emerald-300 font-bold text-lg mb-4">Core Principles</h3>
                                <div className="space-y-3">
                                    {[
                                        { icon: "🚫", label: "Zero Synthetic Amendments", desc: "No chemical fertilisers or pesticides" },
                                        { icon: "🛏️", label: "Permanent Raised Beds", desc: "No-till, non-compacted raised bed system" },
                                        { icon: "💧", label: "70%+ Water Reduction", desc: "Precision moisture management" },
                                        { icon: "🍂", label: "Organic Mulch Cover", desc: "Permanent biomass on soil surface" },
                                        { icon: "📐", label: "Optimal Plant Spacing", desc: "Adapted density for natural soil fertility" },
                                        { icon: "🌳", label: "Agro-biodiversity", desc: "Intercropping, trees, and diversification" },
                                    ].map((item, i) => (
                                        <div key={i} className="flex items-start gap-3 py-2 border-b border-white/5 last:border-0">
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
                    <div className="flex items-center gap-3 mb-5">
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">PQNK Agricultural Practices</h2>
                    </div>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
                        {PQNK_PRACTICES.map((p, i) => (
                            <div
                                key={i}
                                className="card-hover bg-white/10 backdrop-blur-xl border border-white/15 rounded-2xl p-6 shadow-lg"
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
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-1 h-8 rounded-full bg-gradient-to-b from-amber-400 to-yellow-600" />
                            <h2 className="text-2xl font-bold text-white">Our Mission</h2>
                        </div>
                        <div className="grid lg:grid-cols-3 gap-6">
                            {[
                                {
                                    icon: "🌍",
                                    title: "Scale Expert Knowledge",
                                    desc: "Digitise Dr. Sharif's PQNK expertise — accessible to every farmer in Pakistan 24/7, in Urdu and English.",
                                },
                                {
                                    icon: "🤖",
                                    title: "AI-Powered Accessibility",
                                    desc: "Through RAG-powered AI, provide reliable, hallucination-free agronomic guidance grounded entirely in the PQNK knowledge corpus.",
                                },
                                {
                                    icon: "🌱",
                                    title: "Sustainable Agriculture",
                                    desc: "Equip Pakistan's 37% agricultural workforce with tools to transition from chemical-dependent farming to natural, sustainable practices.",
                                },
                            ].map((m, i) => (
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
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-1 h-8 rounded-full bg-gradient-to-b from-green-400 to-emerald-600" />
                        <h2 className="text-2xl font-bold text-white">Meet the Founder</h2>
                    </div>

                    <div className="bg-white/10 backdrop-blur-xl border border-white/15 rounded-3xl p-10 shadow-2xl">
                        {/* Top Section — Name + Title */}
                        <div className="flex flex-col lg:flex-row gap-8 items-center mb-8 pb-8 border-b border-white/10">
                            <div className="flex-shrink-0 text-center">
                                <div
                                    className="w-36 h-36 rounded-full bg-gradient-to-br from-emerald-500 to-green-700 flex items-center justify-center text-6xl mx-auto mb-4 shadow-2xl"
                                    style={{ boxShadow: "0 0 60px rgba(52,211,153,0.3)" }}
                                >
                                    👨‍🌾
                                </div>
                                <h3 className="text-white font-extrabold text-2xl">Mr. Asif Sharif</h3>
                                <p className="text-emerald-400 text-sm font-semibold mt-1">Founder & Chairman — Pedaver</p>
                                <p className="text-white/40 text-xs mt-0.5">PQNK Creator · Agricultural Pioneer</p>
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

                            <div className="flex-1">
                                <div className="inline-flex items-center gap-2 bg-amber-500/15 border border-amber-400/20 rounded-full px-4 py-1.5 text-amber-300 text-xs font-semibold tracking-wider uppercase mb-4">
                                    🏆 A Trailblazing Innovator
                                </div>
                                <p className="text-white/70 text-[15px] leading-relaxed mb-4">
                                    Mr. Asif Sharif is a visionary and innovative agricultural pioneer, entrepreneur, engineer, inventor,
                                    and philanthropist. Born into the esteemed <strong className="text-white">SirGroh family of
                                        Faisalabad</strong>, renowned for its contributions to education, agriculture, and community service,
                                    Mr. Sharif has upheld the family's legacy and virtues.
                                </p>
                                <p className="text-white/70 text-[15px] leading-relaxed">
                                    His expertise encompasses marketing, project development in agriculture, production process development
                                    &amp; mechanisation, forward engineering, alternate energies, value addition, and responsible
                                    citizenship. He has transformed numerous farms by implementing sound programmes for mechanisation,
                                    management, farm design, irrigation, and knowledge-based skills development.
                                </p>
                            </div>
                        </div>

                        {/* Main Sections */}
                        <div className="grid lg:grid-cols-2 gap-8 mb-8">
                            <div>
                                <h4 className="text-emerald-300 font-bold text-base mb-4 flex items-center gap-2">
                                    <span>⚙️</span> Machinery &amp; Innovation
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    Mr. Sharif has revolutionised the agricultural industry with numerous inventions. He designed,
                                    developed, and introduced a multitude of agricultural machinery including soil scrapers, fine levellers,
                                    ridgers, reversible disc ploughs, plate planters, tractor mounted sprayers, sugarcane planters, stubble
                                    shavers, combine harvesters, cotton pickers, and peanut harvesters.
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    His ingenuity extends to machinery for land levelling, customised precision laser levelling,
                                    raised-bed cropping, precision seeding, compost banding, fertiliser side dressing, transplanting,
                                    weeding, and soil aerating. He introduced improved seeds and hybrids of Corn, Cotton, and Sunflower.
                                </p>
                            </div>

                            <div>
                                <h4 className="text-emerald-300 font-bold text-base mb-4 flex items-center gap-2">
                                    <span>🌾</span> Groundbreaking Initiatives
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    Mr. Sharif challenged the conventional tillage-based crop production process used for millennia. He
                                    developed the PQNK system which requires no tillage or agrochemicals and reduces water use by over
                                    <strong className="text-emerald-300"> 70%</strong>. It produces food at the lowest cost and highest
                                    quality, surpassing even organic standards.
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    He developed a method for growing rice on raised beds in moist soil, achieving a remarkable 70%
                                    reduction in water usage. Additionally, he spearheaded the{" "}
                                    <strong className="text-white">"One Acre Prosperity" (OAP)</strong> process, empowering smallholders
                                    to achieve financial independence.
                                </p>
                            </div>
                        </div>

                        {/* Timeline */}
                        <div className="mb-8">
                            <h4 className="text-white font-bold text-base mb-5 flex items-center gap-2">
                                <span>📅</span> Career Timeline
                            </h4>
                            <div className="relative pl-6 border-l border-emerald-500/30 space-y-5">
                                {TIMELINE.map((t, i) => (
                                    <div key={i} className="relative">
                                        <div className="absolute -left-[25px] w-4 h-4 rounded-full bg-emerald-500 border-2 border-emerald-300 shadow-lg shadow-emerald-500/40" />
                                        <div className="flex flex-col sm:flex-row sm:items-baseline gap-2">
                                            <span className="text-emerald-400 font-bold text-sm w-16 flex-shrink-0">{t.year}</span>
                                            <p className="text-white/65 text-sm leading-relaxed">{t.event}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Business & Global Roles */}
                        <div className="grid lg:grid-cols-2 gap-8 pt-6 border-t border-white/10">
                            <div>
                                <h4 className="text-emerald-300 font-bold text-base mb-4 flex items-center gap-2">
                                    <span>🌐</span> Business &amp; Diplomacy
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    Beyond his technical expertise, Mr. Sharif has made significant contributions to business and
                                    agriculture. He served as <strong className="text-white">President of the Pakistan-Belgium Business
                                        Development Forum</strong>, <strong className="text-white">Honorary Consul for the Republic of
                                            Poland</strong>, and a member of the Board of Directors of Zarai Taraqiati Bank.
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    He played a pivotal role in introducing renowned global companies to Pakistan, including{" "}
                                    <strong className="text-white">Ford Motor Company</strong>, Sperry New Holland, Ford New Holland,
                                    Case IH, Toft, Long Manufacturing, Butler, Howard, Ursus, and Kenwood.
                                </p>
                            </div>

                            <div>
                                <h4 className="text-emerald-300 font-bold text-base mb-4 flex items-center gap-2">
                                    <span>❤️</span> Philanthropy &amp; Values
                                </h4>
                                <p className="text-white/60 text-sm leading-relaxed mb-3">
                                    Mr. Sharif is a global citizen who has undergone training in the{" "}
                                    <strong className="text-white">United States, Australia, and Europe</strong>, travelling across five
                                    continents for study, training, business, and research. He is a trustworthy and compassionate leader
                                    earning the respect of colleagues and communities he serves.
                                </p>
                                <p className="text-white/60 text-sm leading-relaxed">
                                    He is particularly passionate about animal welfare, life in the soil, and all lifeforms — embodying a
                                    wholistic philosophy of ecological stewardship extending far beyond agriculture.
                                </p>
                            </div>
                        </div>

                        {/* Quote */}
                        <div className="mt-8 bg-emerald-900/60 border-l-4 border-emerald-400 rounded-r-2xl pl-6 pr-6 py-5">
                            <p className="text-white/80 text-base italic leading-relaxed">
                                "Mr. Asif Sharif is an exemplary individual who has dedicated his life to improving agriculture and
                                empowering communities. His contributions have had a profound impact on the lives of millions of people
                                around the world."
                            </p>
                            <p className="text-emerald-400 text-xs mt-2 font-semibold tracking-wide uppercase">
                                — Pedaver Profile Document, November 2023
                            </p>
                        </div>
                    </div>
                </div>

                {/* ── PARTNERSHIP STRIP ── */}
                <div className="fu5">
                    <div className="bg-white/5 border border-white/10 rounded-2xl px-8 py-6 flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <p className="text-white/40 text-xs uppercase tracking-widest mb-1 font-semibold">In Collaboration With</p>
                            <div className="flex items-center gap-6 flex-wrap">
                                {[
                                    { icon: "🏛️", name: "Pakistan Agriculture Research (PAR)" },
                                    { icon: "🎓", name: "Habib University — DSSE" },
                                    { icon: "🌿", name: "Pedaver — The Transformative Producer" },
                                ].map((p, i) => (
                                    <div key={i} className="flex items-center gap-2">
                                        <span>{p.icon}</span>
                                        <span className="text-white/60 text-sm font-medium">{p.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="text-right hidden sm:block">
                            <p className="text-emerald-400/60 text-xs tracking-wider">DURS 2026</p>
                            <p className="text-white/40 text-xs">AI for Transforming Agriculture</p>
                        </div>
                    </div>
                </div>
            </>
        </DashboardLayout>
    );
}
