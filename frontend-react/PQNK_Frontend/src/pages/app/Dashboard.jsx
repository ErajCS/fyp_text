import DashboardLayout from "../../layouts/DashboardLayout";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const navigate = useNavigate();

  const stats = [
    { title: "Total Resources", value: "128", growth: "+12%", icon: "📚" },
    { title: "Crop Categories", value: "24", growth: "+3", icon: "🌾" },
    { title: "Video Lectures", value: "76", growth: "+8%", icon: "🎥" },
    { title: "Active Farmers", value: "54", growth: "+5", icon: "👨‍🌾" },
  ];

  const quickActions = [
    { label: "Crop Problem", icon: "🌿", desc: "Identify and solve crop issues" },
    { label: "When to Water?", icon: "💧", desc: "Smart irrigation scheduling" },
    { label: "Fertilizer Info", icon: "🧪", desc: "NPK planning & ratios" },
    { label: "Weather Forecast", icon: "🌦️", desc: "7-day climate advisory" },
    { label: "Open Chatbot", icon: "💬", desc: "Ask our AI assistant", action: () => navigate("/chatbot") },
  ];

  return (
    <DashboardLayout>

      {/* Header */}
      <div className="mb-10">
        <h1 className="text-4xl font-bold mb-3 text-white drop-shadow-lg">
          Welcome to AgriChat Dashboard
        </h1>
        <p className="text-white/70 text-base max-w-2xl drop-shadow">
          AI-powered agricultural knowledge ecosystem — crop insights,
          soil analysis, irrigation intelligence, and smart advisory systems.
        </p>
      </div>

      {/* Stats */}
      <div className="grid md:grid-cols-4 gap-5 mb-10">
        {stats.map((item, index) => (
          <div
            key={index}
            className="bg-white/90 backdrop-blur-lg border border-white/50 p-6 rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-3xl">{item.icon}</span>
              <span className="text-xs text-green-600 font-semibold bg-green-100 px-2.5 py-0.5 rounded-lg">
                {item.growth}
              </span>
            </div>
            <h3 className="text-2xl font-bold text-gray-800">{item.value}</h3>
            <p className="text-sm text-gray-500 mt-1">{item.title}</p>
          </div>
        ))}
      </div>

      {/* Quick Actions — wireframe-inspired icon grid */}
      <div className="mb-10">
        <h2 className="text-xl font-semibold mb-5 text-white drop-shadow">Quick Actions</h2>
        <div className="grid grid-cols-5 gap-4">
          {quickActions.map((item, index) => (
            <button
              key={index}
              onClick={item.action || undefined}
              className="bg-white/90 backdrop-blur-lg border border-white/50 p-5 rounded-2xl shadow-lg hover:shadow-2xl hover:scale-[1.03] transition-all duration-300 flex flex-col items-center text-center group"
            >
              <div className="w-14 h-14 rounded-full bg-green-100 border-2 border-green-200 flex items-center justify-center text-2xl mb-3 group-hover:bg-green-200 group-hover:scale-110 transition-all">
                {item.icon}
              </div>
              <h4 className="text-sm font-semibold text-gray-700 mb-1">{item.label}</h4>
              <p className="text-[11px] text-gray-400 leading-tight">{item.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Smart Insights Panel */}
      <div className="grid md:grid-cols-2 gap-6 mb-10">
        <div className="bg-white/90 backdrop-blur-lg p-7 rounded-2xl border border-green-200/50 shadow-xl">
          <h2 className="text-lg font-semibold mb-3 flex items-center gap-2 text-gray-800">
            🌾 Seasonal Crop Advisory
          </h2>
          <p className="text-gray-600 leading-relaxed text-sm">
            Wheat sowing season approaching. Soil moisture currently optimal.
            Recommended nitrogen supplementation within next 10 days.
          </p>
        </div>

        <div className="bg-white/90 backdrop-blur-lg p-7 rounded-2xl border border-amber-200/50 shadow-xl">
          <h2 className="text-lg font-semibold mb-3 flex items-center gap-2 text-gray-800">
            🌱 Soil Health Indicator
          </h2>
          <p className="text-gray-600 leading-relaxed text-sm">
            Current soil index: <span className="text-green-600 font-bold">78%</span>.
            Organic matter improving. Recommend balanced NPK application.
          </p>
        </div>
      </div>

      {/* Crop Focus */}
      <div className="mb-10">
        <h2 className="text-xl font-semibold mb-5 text-white drop-shadow">High Demand Crops</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {["Wheat", "Rice", "Cotton", "Maize", "Sugarcane", "Vegetables"].map(
            (crop, index) => (
              <div
                key={index}
                className="bg-white/90 backdrop-blur-lg p-5 rounded-2xl border border-white/50 shadow-lg hover:shadow-2xl hover:scale-[1.02] transition-all cursor-pointer"
              >
                <h3 className="text-base font-semibold mb-2 text-gray-800">{crop}</h3>
                <p className="text-gray-500 text-xs leading-relaxed">
                  AI recommendations, research papers, irrigation cycles, and
                  fertilizer planning for {crop}.
                </p>
              </div>
            )
          )}
        </div>
      </div>

      {/* AI Quick Action Panel */}
      <div className="bg-gradient-to-r from-green-600 to-emerald-700 p-7 rounded-2xl shadow-2xl flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold mb-1.5 text-white">
            Need Smart Recommendation?
          </h2>
          <p className="text-green-100 text-sm">
            Ask our AI assistant about crop disease, irrigation planning, or fertilizer optimization.
          </p>
        </div>

        <button
          onClick={() => navigate("/chatbot")}
          className="bg-gradient-to-r from-amber-400 to-yellow-500 text-white px-6 py-3 rounded-xl font-semibold hover:scale-105 transition-all shadow-lg shadow-amber-400/30 flex-shrink-0"
        >
          Launch AI Assistant
        </button>
      </div>

    </DashboardLayout>
  );
}