import { Routes, Route } from "react-router-dom"
import PublicHome from "@/pages/PublicHome"
import Login from "@/pages/Login"
import Signup from "@/pages/Signup"
import VerifyOtp from "@/pages/VerifyOtp"
import Dashboard from "@/pages/app/Dashboard"
import Chatbot from "@/pages/app/Chatbot"
import AdminDashboard from "@/pages/app/AdminDashboard"
import SuperAdminDashboard from "@/pages/app/SuperAdminDashboard"
import Profile from "@/pages/app/Profile"
import Repository from "@/pages/app/Repository"
import BrowseRepository from "@/pages/app/BrowseRepository"

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<PublicHome />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/verify-otp" element={<VerifyOtp />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/user-dashboard" element={<Dashboard />} />
      <Route path="/chatbot" element={<Chatbot />} />
      <Route path="/profile" element={<Profile />} />
      <Route path="/browse-repository" element={<BrowseRepository />} />
      <Route path="/admin-dashboard" element={<AdminDashboard />} />
      <Route path="/admin-dashboard/content" element={<Repository />} />
      <Route path="/super-admin-dashboard" element={<SuperAdminDashboard />} />
      <Route path="/super-admin-dashboard/content" element={<Repository />} />
    </Routes>
  )
}
