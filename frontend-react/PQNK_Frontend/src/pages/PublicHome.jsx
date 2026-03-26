import { motion } from "framer-motion"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function PublicHome() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-50 to-white">

      {/* NAVBAR */}
      <div className="flex justify-between items-center px-10 py-6 bg-white shadow-sm">
        <h1 className="text-2xl font-bold text-emerald-700">
          PQNK Agriculture Repository
        </h1>

        <div className="space-x-6">
          <button className="text-zinc-600 hover:text-emerald-600 transition">
            Resources
          </button>
          <button className="text-zinc-600 hover:text-emerald-600 transition">
            About
          </button>
          <Button className="bg-emerald-600 hover:bg-emerald-700">
            Login
          </Button>
        </div>
      </div>

      {/* HERO SECTION */}
      <div className="text-center mt-20 px-6">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-5xl font-extrabold text-zinc-800 leading-tight"
        >
          Centralized Knowledge Hub for <br />
          <span className="text-emerald-600">PQNK Agriculture</span>
        </motion.h2>

        <p className="mt-6 text-lg text-zinc-600 max-w-2xl mx-auto">
          Search agricultural documents, lecture slides, and curated YouTube resources.
          Empowering farmers and researchers with structured knowledge.
        </p>

        {/* SEARCH BAR */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-10 flex justify-center"
        >
          <div className="flex w-full max-w-2xl shadow-lg rounded-xl overflow-hidden">
            <Input
              placeholder="Search for wheat, irrigation, soil fertility..."
              className="h-14 text-lg border-none focus:ring-0"
            />
            <Button className="h-14 px-6 bg-emerald-600 hover:bg-emerald-700">
              <Search size={20} />
            </Button>
          </div>
        </motion.div>
      </div>

      {/* FEATURED SECTION */}
      <div className="mt-24 px-10 pb-20">
        <h3 className="text-2xl font-semibold text-zinc-800 mb-8">
          Featured Categories
        </h3>

        <div className="grid md:grid-cols-3 gap-8">
          {["Wheat", "Soil Science", "Irrigation Systems"].map((item, index) => (
            <motion.div
              key={index}
              whileHover={{ y: -8 }}
              className="bg-white rounded-2xl shadow-md p-8 hover:shadow-xl transition"
            >
              <h4 className="text-xl font-semibold text-emerald-700 mb-3">
                {item}
              </h4>
              <p className="text-zinc-600">
                Access curated documents, slides and research material.
              </p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* FOOTER */}
      <div className="bg-emerald-700 text-white text-center py-6">
        © 2026 PQNK Agriculture Repository
      </div>

    </div>
  )
}