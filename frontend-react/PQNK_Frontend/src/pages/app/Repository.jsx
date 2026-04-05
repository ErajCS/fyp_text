import AdminLayout from "../../layouts/AdminLayout";
import { useState, useEffect, useCallback } from "react";
import useAuth from "../../hooks/useAuth";

const FILE_TYPES = ["document", "image", "video", "audio"];
const FILE_TYPE_ICONS = { document: "📄", image: "🖼️", video: "🎥", audio: "🎵" };
const ACCEPT = {
    document: ".pdf,.doc,.docx,.txt,.pptx,.xlsx",
    image: ".jpg,.jpeg,.png,.gif,.webp,.svg",
    video: ".mp4,.webm,.mov,.avi,.mkv",
    audio: ".mp3,.m4a,.wav,.ogg,.flac",
};

const DEFAULT_FORM = {
    title: "", description: "", category: "", newCategory: "",
    keywords: "", file_type: "document", video_link: "", file: null,
};

export default function Repository() {
    useAuth(["admin", "superadmin"]);

    const [resources, setResources] = useState([]);
    const [categories, setCategories] = useState([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [search, setSearch] = useState("");
    const [typeFilter, setTypeFilter] = useState("");
    const [catFilter, setCatFilter] = useState("");
    const [loading, setLoading] = useState(true);
    const [showUpload, setShowUpload] = useState(false);
    const [form, setForm] = useState(DEFAULT_FORM);
    const [uploading, setUploading] = useState(false);
    const [deleting, setDeleting] = useState(null);
    const [msg, setMsg] = useState({ text: "", type: "" });

    const PER_PAGE = 15;

    const showMsg = (text, type = "success") => {
        setMsg({ text, type });
        setTimeout(() => setMsg({ text: "", type: "" }), 3500);
    };

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

    useEffect(() => { fetchResources(); }, [fetchResources]);
    useEffect(() => { setPage(1); }, [search, typeFilter, catFilter]);

    const handleUpload = async (e) => {
        e.preventDefault();
        setUploading(true);
        try {
            const fd = new FormData();
            fd.append("title", form.title);
            fd.append("description", form.description);
            fd.append("category", form.newCategory || form.category || "General");
            fd.append("keywords", form.keywords);
            fd.append("file_type", form.file_type);
            fd.append("video_link", form.video_link);
            if (form.file) fd.append("file", form.file);

            const res = await fetch("/api/repository/upload", {
                method: "POST", credentials: "include", body: fd,
            });
            const d = await res.json();
            if (d.success) {
                showMsg("Resource uploaded successfully!");
                setForm(DEFAULT_FORM);
                setShowUpload(false);
                fetchResources();
            } else {
                showMsg(d.message || "Upload failed", "error");
            }
        } catch {
            showMsg("Could not connect to server", "error");
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (id, title) => {
        if (!window.confirm(`Delete "${title}"? This cannot be undone.`)) return;
        setDeleting(id);
        try {
            const res = await fetch(`/api/repository/${id}`, {
                method: "DELETE", credentials: "include",
            });
            const d = await res.json();
            if (d.success) { showMsg("Resource deleted"); fetchResources(); }
            else showMsg(d.message, "error");
        } catch {
            showMsg("Could not connect to server", "error");
        } finally {
            setDeleting(null);
        }
    };

    const totalPages = Math.ceil(total / PER_PAGE);

    return (
        <AdminLayout>
            <>
                <style>{`
          @keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
          .fade-up   { animation: fadeUp .45s ease-out both; }
          .fade-up-1 { animation: fadeUp .45s ease-out .08s both; }
        `}</style>

                {/* Header */}
                <div className="fade-up mb-7 flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-white drop-shadow-lg mb-1">📁 Repository Manager</h1>
                        <p className="text-white/50 text-sm">{total.toLocaleString()} resources · Upload and manage all PQNK content</p>
                    </div>
                    <button
                        onClick={() => setShowUpload(true)}
                        className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold text-sm shadow-lg hover:scale-[1.02] transition-all"
                    >
                        + Upload Resource
                    </button>
                </div>

                {/* Feedback */}
                {msg.text && (
                    <div className={`mb-4 px-4 py-3 rounded-xl text-sm font-medium fade-up ${msg.type === "success" ? "bg-green-500/20 border border-green-400/30 text-green-300" : "bg-red-500/20 border border-red-400/30 text-red-300"}`}>
                        {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                    </div>
                )}

                {/* ── Upload Modal ── */}
                {showUpload && (
                    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
                        <div className="bg-white rounded-2xl shadow-2xl p-7 max-w-lg w-full my-4">
                            <div className="flex items-center justify-between mb-5">
                                <h2 className="text-lg font-bold text-gray-800">📤 Upload New Resource</h2>
                                <button onClick={() => setShowUpload(false)} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
                            </div>
                            <form onSubmit={handleUpload} className="space-y-4">
                                {/* Title */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Title *</label>
                                    <input value={form.title} onChange={e => setForm({ ...form, title: e.target.value })} required
                                        className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40" />
                                </div>
                                {/* Description */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                                    <textarea rows={2} value={form.description} onChange={e => setForm({ ...form, description: e.target.value })}
                                        className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40 resize-none" />
                                </div>
                                {/* File Type */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Resource Type *</label>
                                    <div className="flex gap-2">
                                        {FILE_TYPES.map(t => (
                                            <button key={t} type="button"
                                                onClick={() => setForm({ ...form, file_type: t, file: null, video_link: "" })}
                                                className={`flex-1 py-2 rounded-xl text-xs font-semibold border transition ${form.file_type === t ? "bg-green-500 text-white border-green-500" : "border-gray-200 text-gray-600 hover:bg-gray-50"}`}
                                            >
                                                {FILE_TYPE_ICONS[t]} {t.charAt(0).toUpperCase() + t.slice(1)}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                {/* Category */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Existing Category</label>
                                        <select value={form.category} onChange={e => setForm({ ...form, category: e.target.value })}
                                            className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40">
                                            <option value="">— Select —</option>
                                            {categories.map(c => <option key={c} value={c}>{c}</option>)}
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Or New Category</label>
                                        <input placeholder="e.g. Wheat" value={form.newCategory}
                                            onChange={e => setForm({ ...form, newCategory: e.target.value })}
                                            className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40" />
                                    </div>
                                </div>
                                {/* Keywords */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Keywords (comma-separated)</label>
                                    <input value={form.keywords} onChange={e => setForm({ ...form, keywords: e.target.value })}
                                        placeholder="wheat, irrigation, soil"
                                        className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40" />
                                </div>
                                {/* File Upload */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        {form.file_type === "video" ? "Video File (optional)" : "File *"}
                                    </label>
                                    <input type="file" accept={ACCEPT[form.file_type] || ""}
                                        onChange={e => setForm({ ...form, file: e.target.files[0] || null })}
                                        className="w-full text-sm text-gray-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-green-50 file:text-green-700 file:text-sm file:font-medium hover:file:bg-green-100"
                                        required={form.file_type !== "video"} />
                                </div>
                                {/* Video Link (video only) */}
                                {form.file_type === "video" && (
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Video Link * <span className="text-gray-400 font-normal">(YouTube or direct URL)</span></label>
                                        <input value={form.video_link}
                                            onChange={e => setForm({ ...form, video_link: e.target.value })}
                                            placeholder="https://youtube.com/watch?v=..."
                                            className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-green-400/40"
                                            required />
                                    </div>
                                )}
                                {/* Audio notice: invisible in repository, AI-only */}
                                {form.file_type === "audio" && (
                                    <div className="px-3 py-2 rounded-xl bg-amber-50 border border-amber-200 text-amber-700 text-xs">
                                        🎵 <strong>Audio-only ingestion:</strong> This recording will be transcribed, translated, and added to the AI knowledge base. It will <em>not</em> appear in the Browse Repository.
                                    </div>
                                )}
                                {/* Submit */}
                                <div className="flex gap-3 pt-2">
                                    <button type="submit" disabled={uploading}
                                        className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold text-sm hover:from-green-400 hover:to-emerald-500 transition disabled:opacity-50">
                                        {uploading ? "Uploading…" : "📤 Upload Resource"}
                                    </button>
                                    <button type="button" onClick={() => setShowUpload(false)}
                                        className="flex-1 py-2.5 rounded-xl border border-gray-200 text-gray-600 font-semibold text-sm hover:bg-gray-50 transition">
                                        Cancel
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Filters */}
                <div className="fade-up-1 flex flex-wrap gap-3 mb-5">
                    <input type="text" value={search} onChange={e => setSearch(e.target.value)}
                        placeholder="Search title, keywords…"
                        className="flex-1 min-w-[180px] px-4 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-green-400/40 shadow" />
                    <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                        className="px-3 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none shadow">
                        <option value="">All Types</option>
                        {FILE_TYPES.map(t => <option key={t} value={t}>{FILE_TYPE_ICONS[t]} {t.charAt(0).toUpperCase() + t.slice(1)}</option>)}
                    </select>
                    {categories.length > 0 && (
                        <select value={catFilter} onChange={e => setCatFilter(e.target.value)}
                            className="px-3 py-2 rounded-xl bg-white/90 border border-white/50 text-sm text-gray-700 focus:outline-none shadow">
                            <option value="">All Categories</option>
                            {categories.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    )}
                </div>

                {/* Resource Table */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    {loading ? (
                        <div className="space-y-2">
                            {[...Array(8)].map((_, i) => <div key={i} className="h-14 bg-gray-100 rounded-xl animate-pulse" />)}
                        </div>
                    ) : resources.length === 0 ? (
                        <div className="text-center py-16 text-gray-400">
                            <p className="text-5xl mb-4">📂</p>
                            <p className="font-semibold text-gray-500">No resources found</p>
                            <p className="text-sm mt-1">Try adjusting your filters or upload a new resource</p>
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
                                                {r.file_type === "video" && r.video_link && (
                                                    <a href={r.video_link} target="_blank" rel="noreferrer"
                                                        className="text-[10px] px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 border border-blue-200 font-medium flex-shrink-0 hover:bg-blue-200"
                                                    >
                                                        🔗 Link
                                                    </a>
                                                )}
                                            </div>
                                            <p className="text-xs text-gray-400 truncate mt-0.5">
                                                {r.keywords && <span className="mr-3">🏷️ {r.keywords}</span>}
                                                by {r.uploaded_by} · {r.created_at}
                                                {r.original_name && <span className="ml-2">· {r.original_name}</span>}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2 ml-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition">
                                        {r.filename && (
                                            <a href={`/api/repository/file/${r.filename}`} target="_blank" rel="noreferrer"
                                                className="px-3 py-1.5 rounded-lg bg-blue-100 text-blue-700 text-xs font-medium hover:bg-blue-200 transition">
                                                ⬇ Download
                                            </a>
                                        )}
                                        <button
                                            onClick={() => handleDelete(r.id, r.title)}
                                            disabled={deleting === r.id}
                                            className="px-3 py-1.5 rounded-lg bg-red-100 text-red-600 text-xs font-medium hover:bg-red-200 transition disabled:opacity-50"
                                        >
                                            {deleting === r.id ? "…" : "🗑 Delete"}
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-between mt-5 pt-4 border-t border-gray-100">
                            <p className="text-xs text-gray-400">Showing {((page - 1) * PER_PAGE) + 1}–{Math.min(page * PER_PAGE, total)} of {total.toLocaleString()}</p>
                            <div className="flex gap-2">
                                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs disabled:opacity-40 hover:bg-gray-50">← Prev</button>
                                <span className="px-3 py-1.5 text-xs text-gray-500">{page} / {totalPages}</span>
                                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs disabled:opacity-40 hover:bg-gray-50">Next →</button>
                            </div>
                        </div>
                    )}
                </div>
            </>
        </AdminLayout>
    );
}
