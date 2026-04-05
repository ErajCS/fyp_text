import DashboardLayout from "../../layouts/DashboardLayout";
import { useState, useEffect, useCallback } from "react";

const FILE_TYPE_ICONS = { document: "📄", image: "🖼️", video: "🎥" };

export default function BrowseRepository() {
    const [resources, setResources] = useState([]);
    const [categories, setCategories] = useState([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [search, setSearch] = useState("");
    const [typeFilter, setTypeFilter] = useState("");
    const [catFilter, setCatFilter] = useState("");
    const [loading, setLoading] = useState(true);

    const PER_PAGE = 15;

    const fetchResources = useCallback(() => {
        setLoading(true);
        const params = new URLSearchParams({
            page, per_page: PER_PAGE,
            ...(search && { q: search }),
            ...(typeFilter && { file_type: typeFilter }),
            ...(catFilter && { category: catFilter }),
        });
        fetch(`/api/repository?${params}`, { credentials: "include" })
            .then(r => r.json())
            .then(d => {
                if (d.success) {
                    setResources(d.resources);
                    setTotal(d.total);
                    setCategories(d.categories || []);
                }
            })
            .finally(() => setLoading(false));
    }, [page, search, typeFilter, catFilter]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { fetchResources(); }, [fetchResources]);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { setPage(1); }, [search, typeFilter, catFilter]);

    const totalPages = Math.ceil(total / PER_PAGE);

    return (
        <DashboardLayout>
            <>
                <style>{`
          @keyframes fadeUp { from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);} }
          .fade-up   { animation: fadeUp .45s ease-out both; }
          .fade-up-1 { animation: fadeUp .45s ease-out .08s both; }
        `}</style>

                {/* Header */}
                <div className="fade-up mb-7">
                    <h1 className="text-3xl font-bold text-white drop-shadow-lg mb-1">📚 PQNK Repository</h1>
                    <p className="text-white/50 text-sm">{total.toLocaleString()} resources available — documents, images, and video guidance</p>
                </div>

                {/* Filters */}
                <div className="fade-up-1 flex flex-wrap gap-3 mb-5">
                    <input
                        type="text" value={search} onChange={e => setSearch(e.target.value)}
                        placeholder="Search by keyword, title…"
                        className="flex-1 min-w-[200px] px-4 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-green-400/40 shadow"
                    />
                    <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                        className="px-3 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none shadow">
                        <option value="">All Types</option>
                        <option value="document">📄 Documents</option>
                        <option value="image">🖼️ Images</option>
                        <option value="video">🎥 Videos</option>
                    </select>
                    {categories.length > 0 && (
                        <select value={catFilter} onChange={e => setCatFilter(e.target.value)}
                            className="px-3 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none shadow">
                            <option value="">All Categories</option>
                            {categories.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    )}
                </div>

                {/* Resource Grid */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    {loading ? (
                        <div className="space-y-3">
                            {[...Array(6)].map((_, i) => <div key={i} className="h-16 bg-gray-100 rounded-xl animate-pulse" />)}
                        </div>
                    ) : resources.length === 0 ? (
                        <div className="text-center py-16 text-gray-400">
                            <p className="text-5xl mb-4">📭</p>
                            <p className="font-semibold text-gray-500">No resources found</p>
                            <p className="text-sm mt-1">Try a different keyword or clear your filters</p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {resources.map(r => (
                                <div key={r.id}
                                    className="flex items-center justify-between p-4 rounded-xl bg-gray-50 border border-gray-100 hover:bg-white hover:shadow-md transition group"
                                >
                                    <div className="flex items-center gap-4 min-w-0 flex-1">
                                        <div className="text-2xl w-10 text-center flex-shrink-0">{FILE_TYPE_ICONS[r.file_type] || "📁"}</div>
                                        <div className="min-w-0">
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <p className="font-semibold text-sm text-gray-800 truncate">{r.title}</p>
                                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-100 text-green-700 border border-green-200 font-medium flex-shrink-0">
                                                    {r.category}
                                                </span>
                                            </div>
                                            <p className="text-xs text-gray-400 mt-0.5 truncate">
                                                {r.description && <span className="mr-2">{r.description}</span>}
                                                {r.keywords && <span className="text-emerald-600">🏷️ {r.keywords}</span>}
                                            </p>
                                        </div>
                                    </div>
                                    {/* Actions — view only */}
                                    <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                                        {r.file_type === "video" && r.video_link ? (
                                            <a href={r.video_link} target="_blank" rel="noreferrer"
                                                className="px-3 py-1.5 rounded-lg bg-blue-100 text-blue-700 text-xs font-medium hover:bg-blue-200 transition flex items-center gap-1">
                                                ▶ Watch Video
                                            </a>
                                        ) : r.filename ? (
                                            <a href={`/api/repository/file/${r.filename}`} target="_blank" rel="noreferrer"
                                                className="px-3 py-1.5 rounded-lg bg-green-100 text-green-700 text-xs font-medium hover:bg-green-200 transition">
                                                ⬇ View / Download
                                            </a>
                                        ) : null}
                                        {r.drive_view_link && (
                                            <a href={r.drive_view_link} target="_blank" rel="noreferrer"
                                                className="px-3 py-1.5 rounded-lg bg-yellow-100 text-yellow-700 text-xs font-medium hover:bg-yellow-200 transition flex items-center gap-1">
                                                📂 Drive
                                            </a>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-between mt-5 pt-4 border-t border-gray-100">
                            <p className="text-xs text-gray-400">
                                Showing {((page - 1) * PER_PAGE) + 1}–{Math.min(page * PER_PAGE, total)} of {total.toLocaleString()}
                            </p>
                            <div className="flex gap-2">
                                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                                    className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs disabled:opacity-40 hover:bg-gray-50">← Prev</button>
                                <span className="px-3 py-1.5 text-xs text-gray-500">{page} / {totalPages}</span>
                                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                                    className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs disabled:opacity-40 hover:bg-gray-50">Next →</button>
                            </div>
                        </div>
                    )}
                </div>
            </>
        </DashboardLayout>
    );
}
