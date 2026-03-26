import AdminLayout from "../../layouts/AdminLayout";
import { useNavigate } from "react-router-dom";

export default function AdminDashboard() {
    const navigate = useNavigate();

    const stats = [
        { title: "Total Users", value: "1,284", change: "+48 this week", icon: "👥", color: "border-blue-200 bg-blue-50/50" },
        { title: "Pending Approvals", value: "23", change: "5 urgent", icon: "⏳", color: "border-amber-200 bg-amber-50/50" },
        { title: "Flagged Content", value: "8", change: "2 critical", icon: "🚩", color: "border-red-200 bg-red-50/50" },
        { title: "Resources Published", value: "342", change: "+18 this month", icon: "📚", color: "border-green-200 bg-green-50/50" },
    ];

    const moderationQueue = [
        { id: 1, title: "Wheat Irrigation Guide v3", author: "Dr. Ahmed Khan", type: "Document", status: "Pending", date: "2 hours ago", priority: "High" },
        { id: 2, title: "Rice Pest Management Video", author: "Farmer Ali", type: "Video", status: "Under Review", date: "5 hours ago", priority: "Medium" },
        { id: 3, title: "Cotton Harvesting Techniques", author: "AgriExpert", type: "Article", status: "Pending", date: "1 day ago", priority: "Low" },
        { id: 4, title: "Soil Nutrient Analysis Report", author: "Lab Team", type: "Report", status: "Flagged", date: "1 day ago", priority: "High" },
        { id: 5, title: "Sugarcane Growth Cycle PDF", author: "Prof. Hassan", type: "Document", status: "Pending", date: "2 days ago", priority: "Medium" },
    ];

    const recentActivity = [
        { action: "New user registered", detail: "farmer_ali_92 joined the platform", time: "12 min ago", icon: "🆕" },
        { action: "Resource uploaded", detail: "Wheat_Disease_Guide.pdf by Dr. Ahmed", time: "34 min ago", icon: "📤" },
        { action: "Content flagged", detail: "Inappropriate comment on Pest Control thread", time: "1 hr ago", icon: "🚩" },
        { action: "User verified", detail: "agri_researcher_pk email confirmed", time: "2 hrs ago", icon: "✅" },
        { action: "Bulk upload completed", detail: "15 lecture videos processed", time: "3 hrs ago", icon: "📦" },
        { action: "Report generated", detail: "Weekly analytics report exported", time: "5 hrs ago", icon: "📊" },
    ];

    const analyticsCards = [
        { label: "Weekly Uploads", value: "47", trend: "+12%", icon: "📁" },
        { label: "Active Users (7d)", value: "389", trend: "+8%", icon: "👤" },
        { label: "Avg. Session Time", value: "14m", trend: "+3%", icon: "⏱️" },
        { label: "Top Category", value: "Wheat", trend: "34% share", icon: "🌾" },
    ];

    const getPriorityColor = (priority) => {
        switch (priority) {
            case "High": return "text-red-600 bg-red-50 border-red-200";
            case "Medium": return "text-amber-600 bg-amber-50 border-amber-200";
            case "Low": return "text-green-600 bg-green-50 border-green-200";
            default: return "text-gray-500";
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case "Pending": return "text-amber-700 bg-amber-100";
            case "Under Review": return "text-blue-700 bg-blue-100";
            case "Flagged": return "text-red-700 bg-red-100";
            default: return "text-gray-600";
        }
    };

    return (
        <AdminLayout>

            {/* Header */}
            <div className="mb-10">
                <div className="flex items-center gap-3 mb-3">
                    <h1 className="text-4xl font-bold text-white drop-shadow-lg">Admin Dashboard</h1>
                    <span className="px-3 py-1 rounded-lg bg-amber-50 text-amber-700 text-xs font-semibold border border-amber-200 shadow-sm">
                        Admin
                    </span>
                </div>
                <p className="text-white/70 text-base max-w-2xl drop-shadow">
                    Manage content, moderate submissions, and monitor user activity across the PQNK AgriChat platform.
                </p>
            </div>

            {/* Stats */}
            <div className="grid md:grid-cols-4 gap-5 mb-10">
                {stats.map((item, index) => (
                    <div
                        key={index}
                        className={`bg-white/90 backdrop-blur-lg border ${item.color} p-6 rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all duration-300`}
                    >
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-2xl">{item.icon}</span>
                            <span className="text-xs text-green-600 font-semibold bg-green-100 px-2.5 py-0.5 rounded-lg">
                                {item.change}
                            </span>
                        </div>
                        <h3 className="text-2xl font-bold text-gray-800">{item.value}</h3>
                        <p className="text-sm text-gray-500 mt-1">{item.title}</p>
                    </div>
                ))}
            </div>

            {/* Content Moderation Queue + User Activity */}
            <div className="grid lg:grid-cols-3 gap-6 mb-10">

                {/* Moderation Queue — 2 cols */}
                <div className="lg:col-span-2 bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <div className="flex items-center justify-between mb-5">
                        <h2 className="text-lg font-semibold text-gray-800">📋 Content Moderation Queue</h2>
                        <span className="text-xs text-amber-700 bg-amber-100 px-3 py-1 rounded-lg font-medium">
                            {moderationQueue.length} items
                        </span>
                    </div>

                    <div className="space-y-2">
                        {moderationQueue.map((item) => (
                            <div
                                key={item.id}
                                className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 border border-gray-100 hover:bg-white hover:shadow-md transition group"
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2.5 mb-1">
                                        <h4 className="font-semibold text-sm text-gray-800 truncate">{item.title}</h4>
                                        <span className={`text-[10px] px-2 py-0.5 rounded-md border font-bold uppercase ${getPriorityColor(item.priority)}`}>
                                            {item.priority}
                                        </span>
                                    </div>
                                    <p className="text-xs text-gray-400">
                                        by {item.author} · {item.type} · {item.date}
                                    </p>
                                </div>

                                <div className="flex items-center gap-2.5 ml-4">
                                    <span className={`text-xs px-2.5 py-1 rounded-lg font-medium ${getStatusColor(item.status)}`}>
                                        {item.status}
                                    </span>
                                    <div className="flex gap-1.5 opacity-0 group-hover:opacity-100 transition">
                                        <button className="w-7 h-7 rounded-lg bg-green-100 text-green-600 hover:bg-green-200 transition flex items-center justify-center text-sm" title="Approve">
                                            ✓
                                        </button>
                                        <button className="w-7 h-7 rounded-lg bg-red-100 text-red-500 hover:bg-red-200 transition flex items-center justify-center text-sm" title="Reject">
                                            ✕
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* User Activity Feed */}
                <div className="bg-white/90 backdrop-blur-lg border border-gray-200/50 rounded-2xl shadow-xl p-6">
                    <h2 className="text-lg font-semibold mb-5 text-gray-800">🟢 Live Activity</h2>

                    <div className="space-y-3.5">
                        {recentActivity.map((item, index) => (
                            <div key={index} className="flex gap-3 items-start">
                                <div className="w-7 h-7 rounded-lg bg-gray-100 flex items-center justify-center text-sm flex-shrink-0 mt-0.5">
                                    {item.icon}
                                </div>
                                <div className="min-w-0">
                                    <p className="text-sm font-medium text-gray-700">{item.action}</p>
                                    <p className="text-xs text-gray-400 truncate">{item.detail}</p>
                                    <p className="text-[10px] text-green-600/70 mt-0.5">{item.time}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Analytics Snapshot */}
            <div className="mb-10">
                <h2 className="text-lg font-semibold mb-5 text-white drop-shadow">📈 Analytics Snapshot</h2>
                <div className="grid md:grid-cols-4 gap-4">
                    {analyticsCards.map((item, index) => (
                        <div
                            key={index}
                            className="bg-white/90 backdrop-blur-lg border border-gray-200/50 p-5 rounded-2xl shadow-lg hover:shadow-xl transition"
                        >
                            <div className="flex items-center justify-between mb-2.5">
                                <span className="text-xl">{item.icon}</span>
                                <span className="text-xs text-green-600 font-medium">{item.trend}</span>
                            </div>
                            <h3 className="text-xl font-bold text-gray-800">{item.value}</h3>
                            <p className="text-sm text-gray-500 mt-1">{item.label}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gradient-to-r from-green-600 to-emerald-700 p-6 rounded-2xl shadow-2xl">
                <h2 className="text-lg font-bold mb-5 text-white">⚡ Quick Actions</h2>
                <div className="flex flex-wrap gap-3">
                    <button onClick={() => navigate("/admin/moderation")} className="bg-white/20 text-white border border-white/30 px-5 py-2.5 rounded-xl font-semibold hover:bg-white/30 hover:scale-[1.02] transition-all flex items-center gap-2 text-sm backdrop-blur">
                        🛡️ Review Queue
                    </button>
                    <button onClick={() => navigate("/admin/users")} className="bg-white/20 text-white border border-white/30 px-5 py-2.5 rounded-xl font-semibold hover:bg-white/30 hover:scale-[1.02] transition-all flex items-center gap-2 text-sm backdrop-blur">
                        👥 User Management
                    </button>
                    <button className="bg-white/20 text-white border border-white/30 px-5 py-2.5 rounded-xl font-semibold hover:bg-white/30 hover:scale-[1.02] transition-all flex items-center gap-2 text-sm backdrop-blur">
                        📚 Publish Resource
                    </button>
                    <button onClick={() => navigate("/chatbot")} className="bg-gradient-to-r from-amber-400 to-yellow-500 text-white px-5 py-2.5 rounded-xl font-semibold hover:scale-[1.02] transition-all flex items-center gap-2 text-sm shadow-lg shadow-amber-400/30">
                        🤖 AI Assistant
                    </button>
                </div>
            </div>

        </AdminLayout>
    );
}
