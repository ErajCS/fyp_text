import { Home, MessageSquare, Settings } from "lucide-react"
import { motion } from "framer-motion"
import { Link } from "react-router-dom"

export default function Sidebar() {
  return (
    <motion.div
      initial={{ x: -60, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="w-64 bg-zinc-950 text-white h-screen p-6 flex flex-col justify-between"
    >
      <div>
        <h1 className="text-2xl font-bold mb-10 tracking-tight">
          YourApp
        </h1>

        <nav className="space-y-4">
          <Link
            to="/dashboard"
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-zinc-800 transition-all"
          >
            <Home size={18} />
            Dashboard
          </Link>

          <Link
            to="/chat"
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-zinc-800 transition-all"
          >
            <MessageSquare size={18} />
            Chat
          </Link>

          <Link
            to="#"
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-zinc-800 transition-all"
          >
            <Settings size={18} />
            Settings
          </Link>
        </nav>
      </div>

      <p className="text-xs text-zinc-500">
        © 2026 YourApp
      </p>
    </motion.div>
  )
}