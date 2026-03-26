import SuperAdminLayout from "../../layouts/SuperAdminLayout";
import { useState } from "react";

export default function SuperAdminDashboard() {
    const [maintenanceMode, setMaintenanceMode] = useState(false);

    const systemStats = [
        { title: "Total Admins", value: "12", icon: "🛡️", color: "border-indigo-200 bg-indigo-50/50", detail: "3 Super, 9 Admin" },
        { title: "Platform Uptime", value: "99.7%", icon: "🟢", color: "border-green-200 bg-green-50/50", detail: "Last 30 days" },
        { title: "API Calls (24h)", value: "48.2K", icon: "🔗", color: "border-cyan-200 bg-cyan-50/50", detail: "+12% vs yesterday" },
        { title: "Storage Used", value: "2.4 TB", icon: "💾", color: "border-purple-200 bg-purple-50/50", detail: "of 5 TB quota" },
        { title: "Error Rate", value: "0.03%", icon: "⚠️", color: "border-red-200 bg-red-50/50", detail: "Last 24 hours" },
    ];

    const adminList = [
        { id: 1, name: "Dr. Ahmed Khan", email: "ahmed@pqnk.edu", role: "Super Admin", lastActive: "Online now", status: "Active", actions: 142 },
        { id: 2, name: "Sara Malik", email: "sara@pqnk.edu", role: "Admin", lastActive: "2 hrs ago", status: "Active", actions: 89 },
        { id: 3, name: "Usman Tariq", email: "usman@pqnk.edu", role: "Admin", lastActive: "5 hrs ago", status: "Active", actions: 67 },
        { id: 4, name: "Ayesha Noor", email: "ayesha@pqnk.edu", role: "Admin", lastActive: "1 day ago", status: "Active", actions: 45 },
        { id: 5, name: "Hassan Raza", email: "hassan@pqnk.edu", role: "Admin", lastActive: "3 days ago", status: "Inactive", actions: 12 },
    ];

    const auditLogs = [
        { time: "08:24:11", admin: "Dr. Ahmed Khan", action: "System config updated", category: "Config", severity: "Info" },
        { time: "08:12:45", admin: "Sara Malik", action: "Approved 5 resources in bulk", category: "Content", severity: "Info" },
        { time: "07:58:32", admin: "System", action: "Auto-backup completed (2.4 TB)", category: "System", severity: "Info" },
        { time: "07:45:19", admin: "Usman Tariq", action: "Banned user spam_bot_99", category: "Users", severity: "Warning" },
        { time: "07:30:05", admin: "System", action: "Rate limit triggered: API endpoint /search", category: "Security", severity: "Warning" },
        { time: "07:15:48", admin: "Dr. Ahmed Khan", action: "API key regenerated for mobile app", category: "API", severity: "Critical" },
        { time: "06:50:22", admin: "Ayesha Noor", action: "Created new category: Organic Farming", category: "Content", severity: "Info" },
        { time: "06:30:10", admin: "System", action: "SSL certificate renewed successfully", category: "System", severity: "Info" },
    ];

    const apiKeys = [
        { name: "Mobile App v2", key: "pk_live_...4f8a", calls: "12,340", status: "Active", created: "Jan 15, 2026" },
        { name: "Research Portal", key: "pk_live_...9b2c", calls: "8,920", status: "Active", created: "Feb 01, 2026" },
        { name: "Web Dashboard", key: "pk_live_...1d7e", calls: "26,940", status: "Active", created: "Dec 20, 2025" },
        { name: "Legacy API (v1)", key: "pk_test_...5a3f", calls: "102", status: "Deprecated", created: "Jun 10, 2025" },
    ];

    const healthMetrics = [
        { name: "Web Server", status: "Healthy", load: "23%", icon: "🌐" },
        { name: "Database (Primary)", status: "Healthy", load: "41%", icon: "🗄️" },
        { name: "Database (Replica)", status: "Healthy", load: "18%", icon: "🗄️" },
        { name: "Cache (Redis)", status: "Healthy", load: "12%", icon: "⚡" },
        { name: "Queue Worker", status: "Healthy", load: "7%", icon: "📮" },
        { name: "AI Model Server", status: "Healthy", load: "56%", icon: "🧠" },
    ];

    const getSeverityColor = (severity) => {
        switch (severity) {
            case "Critical": return "text-red-600 bg-red-100";
            case "Warning": return "text-amber-600 bg-amber-100";
            case "Info": return "text-blue-600 bg-blue-100";
            default: return "text-gray-500";
        }
    };

    const getRoleColor = (role) => {
        return role === "Super Admin"
            ? "text-amber-700 bg-amber-100 border-amber-200"
            : "text-blue-700 bg-blue-100 border-blue-200";
    };

    return (
        <SuperAdminLayout>

            {/* Header */}
            <div className="mb-10">
                <div className="flex items-center gap-3 mb-3">
                    <h1 className="text-4xl font-bold text-white drop-shadow-lg">Command Center</h1>
                    <span className="px-3 py-1 rounded-lg bg-amber-50 text-amber-700 text-xs font-semibold border border-amber-200 shadow-sm">
                        Super Admin
                    </span>
                </div>
                <p className="text-white/70 text-base max-w-2xl drop-shadow">
                    Full system oversight — manage admins, audit operations, configure the platform, and monitor infrastructure health.
                </p>
            </div>

            {/* System Stats */}
            <div className="grid md:grid-cols-5 gap-4 mb-10">
                {systemStats.map((item, index) => (
                    <div
                        key={index}
                        className={`bg-white/90 backdrop-blur-lg border ${item.color} p-5 rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all duration-300`}
                    >
                        <span className="text-xl">{item.icon}</span>
                        <h3 className="text-xl font-bold mt-2 text-gray-800">{item.value}</h3>
                        <p className="text-sm text-gray-600 mt-1">{item.title}</p>
                        <p className="text-[11px] text-gray-400 mt-0.5">{item.detail}</p>
                    </div>
                ))}
            </div>

            {/* Admin Management + Audit Logs */}
            <div className="grid lg:grid-cols-5 gap-6 mb-10">

                {/* Admin Management — 3 cols */}
                <div className="lg:col-span-3 bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <div className="flex items-center justify-between mb-5">
                        <h2 className="text-lg font-semibold text-gray-800">🛡️ Admin Management</h2>
                        <button className="text-xs bg-amber-100 text-amber-700 border border-amber-200 px-4 py-2 rounded-xl font-semibold hover:bg-amber-200 transition">
                            + Add Admin
                        </button>
                    </div>

                    <div className="space-y-2">
                        {adminList.map((admin) => (
                            <div
                                key={admin.id}
                                className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 border border-gray-100 hover:bg-white hover:shadow-md transition group"
                            >
                                <div className="flex items-center gap-3.5">
                                    <div className="w-9 h-9 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-sm font-bold text-white">
                                        {admin.name.split(" ").map(n => n[0]).join("")}
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h4 className="font-semibold text-sm text-gray-800">{admin.name}</h4>
                                            <span className={`text-[10px] px-2 py-0.5 rounded-md border font-bold ${getRoleColor(admin.role)}`}>
                                                {admin.role}
                                            </span>
                                        </div>
                                        <p className="text-xs text-gray-400">{admin.email} · {admin.actions} actions · {admin.lastActive}</p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition">
                                    <button className="text-xs px-3 py-1.5 rounded-lg bg-blue-100 text-blue-700 hover:bg-blue-200 transition">Edit</button>
                                    <button className="text-xs px-3 py-1.5 rounded-lg bg-red-100 text-red-600 hover:bg-red-200 transition">Remove</button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Audit Log Feed — 2 cols */}
                <div className="lg:col-span-2 bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <div className="flex items-center justify-between mb-5">
                        <h2 className="text-lg font-semibold text-gray-800">📋 Audit Log</h2>
                        <button className="text-[10px] text-gray-400 hover:text-gray-600 transition uppercase tracking-wider font-medium">
                            Export All
                        </button>
                    </div>

                    <div className="space-y-2.5 max-h-[400px] overflow-y-auto pr-1">
                        {auditLogs.map((log, index) => (
                            <div key={index} className="p-3 rounded-xl bg-gray-50 border border-gray-100">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs text-gray-400 font-mono">{log.time}</span>
                                    <span className={`text-[10px] px-2 py-0.5 rounded-md font-medium ${getSeverityColor(log.severity)}`}>
                                        {log.severity}
                                    </span>
                                </div>
                                <p className="text-sm font-medium text-gray-700">{log.action}</p>
                                <p className="text-[11px] text-gray-400">{log.admin} · {log.category}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* System Config + API Keys */}
            <div className="grid lg:grid-cols-2 gap-6 mb-10">

                {/* System Configuration */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <h2 className="text-lg font-semibold mb-5 text-gray-800">⚙️ System Configuration</h2>

                    <div className="space-y-3">
                        {/* Maintenance Mode Toggle */}
                        <div className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 border border-gray-100">
                            <div>
                                <h4 className="font-semibold text-sm text-gray-700">Maintenance Mode</h4>
                                <p className="text-xs text-gray-400">Temporarily disable platform access for all users</p>
                            </div>
                            <button
                                onClick={() => setMaintenanceMode(!maintenanceMode)}
                                className={`w-12 h-6 rounded-full transition-all duration-300 relative ${maintenanceMode ? "bg-red-500" : "bg-gray-300"}`}
                            >
                                <div className={`w-[18px] h-[18px] rounded-full bg-white absolute top-[3px] transition-all duration-300 shadow ${maintenanceMode ? "left-[26px]" : "left-[3px]"}`} />
                            </button>
                        </div>

                        {/* Feature Toggles */}
                        <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-100">
                            <h4 className="font-semibold text-sm mb-3 text-gray-700">Feature Flags</h4>
                            <div className="space-y-2">
                                {[
                                    { label: "AI Chatbot", enabled: true },
                                    { label: "Resource Upload", enabled: true },
                                    { label: "Video Lectures", enabled: true },
                                    { label: "Public Search", enabled: false },
                                ].map((feat, i) => (
                                    <div key={i} className="flex items-center justify-between py-1">
                                        <span className="text-sm text-gray-600">{feat.label}</span>
                                        <div className={`w-9 h-5 rounded-full ${feat.enabled ? "bg-green-500" : "bg-gray-300"} relative cursor-pointer`}>
                                            <div className={`w-3.5 h-3.5 rounded-full bg-white absolute top-[3px] shadow ${feat.enabled ? "left-[19px]" : "left-[3px]"}`} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Rate Limiting */}
                        <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-100">
                            <h4 className="font-semibold text-sm mb-2 text-gray-700">Rate Limits</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <p className="text-xs text-gray-400 mb-1">API (per min)</p>
                                    <p className="text-lg font-bold text-green-600">120</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-400 mb-1">Uploads (per hr)</p>
                                    <p className="text-lg font-bold text-green-600">50</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* API Key Management */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <div className="flex items-center justify-between mb-5">
                        <h2 className="text-lg font-semibold text-gray-800">🔑 API Keys</h2>
                        <button className="text-xs bg-green-100 text-green-700 border border-green-200 px-4 py-2 rounded-xl font-semibold hover:bg-green-200 transition">
                            + Generate Key
                        </button>
                    </div>

                    <div className="space-y-2.5">
                        {apiKeys.map((key, index) => (
                            <div key={index} className="p-3.5 rounded-xl bg-gray-50 border border-gray-100 hover:bg-white hover:shadow-md transition group">
                                <div className="flex items-center justify-between mb-1.5">
                                    <h4 className="font-semibold text-sm text-gray-800">{key.name}</h4>
                                    <span className={`text-[10px] px-2 py-0.5 rounded-md font-bold ${key.status === "Active" ? "text-green-600 bg-green-100" : "text-red-600 bg-red-100"}`}>
                                        {key.status}
                                    </span>
                                </div>
                                <p className="text-sm text-gray-400 font-mono mb-1.5">{key.key}</p>
                                <div className="flex items-center justify-between">
                                    <p className="text-xs text-gray-400">{key.calls} calls · Created {key.created}</p>
                                    <div className="flex gap-1.5 opacity-0 group-hover:opacity-100 transition">
                                        <button className="text-[10px] px-2 py-1 rounded bg-blue-100 text-blue-700 hover:bg-blue-200 transition">Regenerate</button>
                                        <button className="text-[10px] px-2 py-1 rounded bg-red-100 text-red-600 hover:bg-red-200 transition">Revoke</button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Data Export + Platform Health */}
            <div className="grid lg:grid-cols-2 gap-6 mb-10">

                {/* Data Export */}
                <div className="bg-gradient-to-br from-green-600 to-emerald-700 rounded-2xl shadow-2xl p-6">
                    <h2 className="text-lg font-semibold mb-5 text-white">📦 Data Export & Bulk Operations</h2>

                    <div className="grid grid-cols-2 gap-3">
                        {[
                            { label: "Export Users", icon: "👥", format: "CSV", records: "1,284" },
                            { label: "Export Resources", icon: "📚", format: "JSON", records: "342" },
                            { label: "Export Audit Logs", icon: "📋", format: "CSV", records: "8,420" },
                            { label: "Full DB Backup", icon: "💾", format: "SQL", records: "2.4 TB" },
                        ].map((item, index) => (
                            <button
                                key={index}
                                className="p-4 rounded-xl bg-white/15 border border-white/20 hover:bg-white/25 hover:scale-[1.02] transition-all text-left text-white backdrop-blur"
                            >
                                <span className="text-xl">{item.icon}</span>
                                <h4 className="font-semibold mt-2 text-sm">{item.label}</h4>
                                <p className="text-[11px] text-white/60 mt-1">{item.records} records · {item.format}</p>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Platform Health Monitor */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <h2 className="text-lg font-semibold mb-5 text-gray-800">💚 Platform Health</h2>

                    <div className="space-y-2.5">
                        {healthMetrics.map((metric, index) => (
                            <div key={index} className="flex items-center justify-between p-3 rounded-xl bg-gray-50 border border-gray-100">
                                <div className="flex items-center gap-2.5">
                                    <span className="text-base">{metric.icon}</span>
                                    <span className="text-sm font-medium text-gray-700">{metric.name}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="w-20 h-1.5 rounded-full bg-gray-200 overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all ${parseInt(metric.load) > 70 ? "bg-red-500" : parseInt(metric.load) > 40 ? "bg-amber-500" : "bg-green-500"}`}
                                            style={{ width: metric.load }}
                                        />
                                    </div>
                                    <span className="text-xs text-gray-500 w-8 text-right">{metric.load}</span>
                                    <div className="w-2 h-2 rounded-full bg-green-500" />
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-5 p-3.5 rounded-xl bg-green-50 border border-green-200 text-center">
                        <p className="text-sm text-green-700 font-medium">All systems operational</p>
                        <p className="text-[11px] text-gray-400 mt-1">Last checked: 2 minutes ago</p>
                    </div>
                </div>
            </div>

            {/* Role & Permission Summary */}
            <div className="bg-gradient-to-r from-green-600 to-emerald-700 p-6 rounded-2xl shadow-2xl">
                <h2 className="text-lg font-bold mb-5 text-white">🔐 Role & Permission Overview</h2>
                <div className="grid md:grid-cols-3 gap-5">
                    {[
                        { role: "User", count: "1,284", permissions: ["View Resources", "AI Chatbot", "Search", "Profile"], color: "bg-white/15 border-white/20", badge: "text-green-200" },
                        { role: "Admin", count: "9", permissions: ["Content Mgmt", "User Mgmt", "Moderation", "Analytics"], color: "bg-white/15 border-white/20", badge: "text-blue-200" },
                        { role: "Super Admin", count: "3", permissions: ["System Config", "Admin Mgmt", "Audit Logs", "API Keys", "Full Access"], color: "bg-white/15 border-white/20", badge: "text-amber-200" },
                    ].map((item, index) => (
                        <div key={index} className={`p-4 rounded-xl border ${item.color} backdrop-blur`}>
                            <div className="flex items-center justify-between mb-2.5">
                                <h4 className={`font-bold ${item.badge}`}>{item.role}</h4>
                                <span className="text-xs text-white/50">{item.count} users</span>
                            </div>
                            <div className="space-y-1">
                                {item.permissions.map((p, i) => (
                                    <p key={i} className="text-xs text-white/60 flex items-center gap-2">
                                        <span className={`text-[10px] ${item.badge}`}>✓</span> {p}
                                    </p>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

        </SuperAdminLayout>
    );
}
