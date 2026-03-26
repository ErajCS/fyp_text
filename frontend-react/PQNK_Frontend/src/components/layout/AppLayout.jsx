import Sidebar from "./Sidebar"
import Topbar from "./Topbar"

export default function AppLayout({ children }) {
  return (
    <div className="flex bg-zinc-100">
      <Sidebar />
      <div className="flex-1 min-h-screen">
        <Topbar />
        <div className="p-8">
          {children}
        </div>
      </div>
    </div>
  )
}