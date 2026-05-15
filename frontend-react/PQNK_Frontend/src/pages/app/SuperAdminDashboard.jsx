import SuperAdminLayout from "../../layouts/SuperAdminLayout";
import { useState, useEffect, useCallback } from "react";
import useAuth from "../../hooks/useAuth";

const ROLE_OPTIONS = ["seeker", "farmer", "researcher", "admin", "superadmin"];
const ROLE_COLORS = {
    seeker: "bg-gray-100 text-gray-700 border-gray-200",
    farmer: "bg-green-100 text-green-700 border-green-200",
    researcher: "bg-blue-100 text-blue-700 border-blue-200",
    admin: "bg-amber-100 text-amber-700 border-amber-200",
    superadmin: "bg-red-100 text-red-700 border-red-200",
};

function getInitials(name = "") {
    return name.split(" ").map(n => n[0]).slice(0, 2).join("").toUpperCase() || "?";
}

export default function SuperAdminDashboard() {
    // ── Auth guard: validates session, redirects if role insufficient ──
    const currentUser = useAuth(["admin", "superadmin"]);

    const [stats, setStats] = useState(null);
    const [users, setUsers] = useState([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [search, setSearch] = useState("");
    const [roleFilter, setRoleFilter] = useState("");
    const [loading, setLoading] = useState(true);
    const [statsLoading, setStatsLoading] = useState(true);
    const [roleChangeId, setRoleChangeId] = useState(null);
    const [deleteId, setDeleteId] = useState(null);
    const [confirmDelete, setConfirmDelete] = useState(null); // user to confirm delete
    const [msg, setMsg] = useState({ text: "", type: "" });

    const PER_PAGE = 20;

    // Fetch stats
    const fetchStats = useCallback(() => {
        setStatsLoading(true);
        fetch("/api/admin/stats", { credentials: "include" })
            .then(r => r.json())
            .then(d => { if (d.success) setStats(d.stats); })
            .finally(() => setStatsLoading(false));
    }, []);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { fetchStats(); }, [fetchStats]);

    // Fetch users
    const fetchUsers = useCallback(() => {
        setLoading(true);
        const params = new URLSearchParams({
            page, per_page: PER_PAGE,
            ...(search && { search }),
            ...(roleFilter && { role: roleFilter }),
        });
        fetch(`/api/admin/users?${params}`, { credentials: "include" })
            .then(r => r.json())
            .then(d => { if (d.success) { setUsers(d.users); setTotal(d.total); } })
            .finally(() => setLoading(false));
    }, [page, search, roleFilter]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { fetchUsers(); }, [fetchUsers]);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { setPage(1); }, [search, roleFilter]);

    const showMsg = (text, type = "success") => {
        setMsg({ text, type });
        setTimeout(() => setMsg({ text: "", type: "" }), 3500);
    };

    const handleRoleChange = async (userId, newRole) => {
        setRoleChangeId(userId);
        try {
            const res = await fetch(`/api/admin/users/${userId}/role`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ role: newRole }),
            });
            const d = await res.json();
            if (d.success) { showMsg(`Role updated to ${newRole}`); fetchUsers(); fetchStats(); }
            else showMsg(d.message, "error");
        } catch { showMsg("Could not connect to server", "error"); }
        finally { setRoleChangeId(null); }
    };

    const handleDelete = async (userId) => {
        setDeleteId(userId);
        setConfirmDelete(null);
        try {
            const res = await fetch(`/api/admin/users/${userId}`, {
                method: "DELETE", credentials: "include",
            });
            const d = await res.json();
            if (d.success) { showMsg("User deleted"); fetchUsers(); fetchStats(); }
            else showMsg(d.message, "error");
        } catch { showMsg("Could not connect to server", "error"); }
        finally { setDeleteId(null); }
    };

    const totalPages = Math.ceil(total / PER_PAGE);

    const statCards = stats ? [
        { title: "Total Users", value: stats.total_users, icon: "👥", sub: `${stats.new_this_week} this week`, color: "border-blue-200 bg-blue-50/50" },
        { title: "Verified", value: stats.verified, icon: "✅", sub: `${stats.unverified} pending`, color: "border-green-200 bg-green-50/50" },
        { title: "Admins", value: stats.admins, icon: "🛡️", sub: "Admin + Super", color: "border-amber-200 bg-amber-50/50" },
        { title: "Farmers", value: stats.farmers, icon: "🌾", sub: "Active farmers", color: "border-emerald-200 bg-emerald-50/50" },
        { title: "Researchers", value: stats.researchers, icon: "🔬", sub: "Research accounts", color: "border-purple-200 bg-purple-50/50" },
    ] : [];

    return (
        <SuperAdminLayout>
            <>
                <style>{`
          @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
          .fade-up   { animation: fadeUp .5s ease-out both; }
          .fade-up-1 { animation: fadeUp .5s ease-out .1s both; }
          .fade-up-2 { animation: fadeUp .5s ease-out .2s both; }
        `}</style>

                {/* Delete Confirmation Modal */}
                {confirmDelete && (
                    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                        <div className="bg-white rounded-2xl shadow-2xl p-7 max-w-sm w-full">
                            <div className="text-3xl mb-4">⚠️</div>
                            <h3 className="text-lg font-bold text-gray-800 mb-2">Delete User Account?</h3>
                            <p className="text-gray-500 text-sm mb-6">
                                This will permanently delete <strong>{confirmDelete.name}</strong> ({confirmDelete.email}).
                                This action cannot be undone.
                            </p>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => handleDelete(confirmDelete.id)}
                                    disabled={deleteId === confirmDelete.id}
                                    className="flex-1 py-2.5 rounded-xl bg-red-600 text-white font-semibold text-sm hover:bg-red-700 transition disabled:opacity-50"
                                >
                                    {deleteId === confirmDelete.id ? "Deleting…" : "Delete User"}
                                </button>
                                <button
                                    onClick={() => setConfirmDelete(null)}
                                    className="flex-1 py-2.5 rounded-xl border border-gray-200 text-gray-700 font-semibold text-sm hover:bg-gray-50 transition"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Header */}
                <div className="fade-up mb-8">
                    <div className="flex items-center gap-3 mb-2">
                        <h1 className="text-3xl font-bold text-white drop-shadow-lg">Command Center</h1>
                        <span className="px-3 py-1 rounded-lg bg-red-50 text-red-700 text-xs font-semibold border border-red-200">Super Admin</span>
                    </div>
                    <p className="text-white/60 text-sm">Full system oversight — manage all accounts, roles, and monitor platform activity.</p>
                </div>

                {/* Feedback */}
                {msg.text && (
                    <div className={`mb-5 px-4 py-3 rounded-xl text-sm font-medium ${msg.type === "success" ? "bg-green-500/20 border border-green-400/30 text-green-300" : "bg-red-500/20 border border-red-400/30 text-red-300"}`}>
                        {msg.type === "success" ? "✅ " : "❌ "}{msg.text}
                    </div>
                )}

                {/* Stats */}
                <div className="fade-up-1 grid md:grid-cols-5 gap-4 mb-8">
                    {statsLoading ? (
                        [...Array(5)].map((_, i) => (
                            <div key={i} className="bg-white/90 border border-gray-200 rounded-2xl p-5 animate-pulse">
                                <div className="h-5 bg-gray-200 rounded mb-3 w-1/2" />
                                <div className="h-7 bg-gray-200 rounded mb-2" />
                            </div>
                        ))
                    ) : statCards.map((item, i) => (
                        <div key={i} className={`bg-white/90 backdrop-blur-lg border ${item.color} p-5 rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all duration-300`}>
                            <span className="text-xl">{item.icon}</span>
                            <h3 className="text-2xl font-bold mt-2 text-gray-800">{item.value?.toLocaleString()}</h3>
                            <p className="text-sm text-gray-600 mt-1">{item.title}</p>
                            <p className="text-[11px] text-gray-400 mt-0.5">{item.sub}</p>
                        </div>
                    ))}
                </div>

                {/* Full User Management */}
                <div className="fade-up-2 bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6 mb-6">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
                        <div>
                            <h2 className="text-lg font-semibold text-gray-800">🛡️ Full User Registry</h2>
                            <p className="text-xs text-gray-400 mt-1">{total.toLocaleString()} accounts · Role changes apply immediately</p>
                        </div>
                        <div className="flex gap-3 w-full sm:w-auto">
                            <input
                                type="text" value={search}
                                onChange={e => setSearch(e.target.value)}
                                placeholder="Search name or email…"
                                className="flex-1 sm:w-56 px-3 py-2 rounded-xl border border-gray-200 bg-gray-50 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-green-400/30 focus:border-green-400"
                            />
                            <select
                                value={roleFilter} onChange={e => setRoleFilter(e.target.value)}
                                className="px-3 py-2 rounded-xl border border-gray-200 bg-gray-50 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-green-400/30"
                            >
                                <option value="">All Roles</option>
                                {ROLE_OPTIONS.map(r => (
                                    <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {loading ? (
                        <div className="space-y-2">
                            {[...Array(10)].map((_, i) => (
                                <div key={i} className="h-14 bg-gray-100 rounded-xl animate-pulse" />
                            ))}
                        </div>
                    ) : users.length === 0 ? (
                        <div className="text-center py-12 text-gray-400">
                            <p className="text-4xl mb-3">🔍</p>
                            <p className="font-medium">No users found</p>
                        </div>
                    ) : (
                        <div className="space-y-1.5">
                            {users.map(user => (
                                <div
                                    key={user.id}
                                    className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 border border-gray-100 hover:bg-white hover:shadow-md transition group"
                                >
                                    <div className="flex items-center gap-3 min-w-0 flex-1">
                                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-sm font-bold text-white flex-shrink-0">
                                            {getInitials(user.name)}
                                        </div>
                                        <div className="min-w-0">
                                            <div className="flex items-center gap-2">
                                                <p className="text-sm font-semibold text-gray-800 truncate">{user.name}</p>
                                                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-bold flex-shrink-0 ${user.is_verified ? "bg-green-100 text-green-700 border-green-200" : "bg-amber-100 text-amber-700 border-amber-200"}`}>
                                                    {user.is_verified ? "Verified" : "Pending"}
                                                </span>
                                            </div>
                                            <p className="text-xs text-gray-400 truncate">{user.email} · {user.created_at}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                                        {/* Role selector — superadmin can set any role */}
                                        <select
                                            value={user.role}
                                            disabled={roleChangeId === user.id}
                                            onChange={e => handleRoleChange(user.id, e.target.value)}
                                            className={`text-xs px-2.5 py-1.5 rounded-lg border font-semibold cursor-pointer focus:outline-none transition ${ROLE_COLORS[user.role] || "bg-gray-100 text-gray-600"} disabled:opacity-50`}
                                        >
                                            {ROLE_OPTIONS.map(r => (
                                                <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                                            ))}
                                        </select>
                                        {roleChangeId === user.id && (
                                            <span className="text-xs text-gray-400 animate-pulse">Saving…</span>
                                        )}
                                        {/* Delete button — appears on hover */}
                                        <button
                                            onClick={() => setConfirmDelete(user)}
                                            disabled={deleteId === user.id}
                                            className="opacity-0 group-hover:opacity-100 w-7 h-7 rounded-lg bg-red-100 text-red-500 hover:bg-red-200 flex items-center justify-center transition text-xs"
                                            title="Delete user"
                                        >
                                            🗑️
                                        </button>
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
                                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition">← Prev</button>
                                <span className="px-3 py-1.5 text-xs text-gray-500">{page} / {totalPages}</span>
                                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="px-3 py-1.5 rounded-lg border border-gray-200 text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-40 transition">Next →</button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Role Distribution */}
                {stats && (
                    <div className="bg-gradient-to-r from-green-600 to-emerald-700 p-6 rounded-2xl shadow-2xl">
                        <h2 className="text-base font-semibold text-white mb-4">🔐 Role & Permission Overview</h2>
                        <div className="grid md:grid-cols-3 gap-5">
                            {[
                                { role: "Users", count: stats.seekers + stats.farmers + stats.researchers, perms: ["View Resources", "AI Chatbot", "Search", "Profile"], badge: "text-green-200" },
                                { role: "Admins", count: stats.admins, perms: ["User Management", "Role Assignment", "Platform Analytics"], badge: "text-blue-200" },
                                { role: "Super Admins", count: "-", perms: ["Full Access", "Delete Users", "Promote to Admin", "System Overview"], badge: "text-amber-200" },
                            ].map((item, i) => (
                                <div key={i} className="bg-white/15 border border-white/20 rounded-xl p-5 backdrop-blur">
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className={`font-bold ${item.badge}`}>{item.role}</h4>
                                        <span className="text-sm text-white/50">{item.count} users</span>
                                    </div>
                                    <div className="space-y-1.5">
                                        {item.perms.map((p, j) => (
                                            <p key={j} className={`text-xs text-white/60 flex items-center gap-2`}>
                                                <span className={`text-[10px] ${item.badge}`}>✓</span> {p}
                                            </p>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </>
        </SuperAdminLayout>
    );
}
