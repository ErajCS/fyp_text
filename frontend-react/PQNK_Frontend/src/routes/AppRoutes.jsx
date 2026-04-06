import { Routes, Route, Navigate } from "react-router-dom"
import { useEffect, useState } from "react"
import PublicHome from "@/pages/PublicHome"
import Login from "@/pages/Login"
import Signup from "@/pages/Signup"
import VerifyOtp from "@/pages/VerifyOtp"
import ForgotPassword from "@/pages/ForgotPassword"
import Dashboard from "@/pages/app/Dashboard"
import Chatbot from "@/pages/app/Chatbot"
import AdminDashboard from "@/pages/app/AdminDashboard"
import SuperAdminDashboard from "@/pages/app/SuperAdminDashboard"
import Profile from "@/pages/app/Profile"
import Repository from "@/pages/app/Repository"
import BrowseRepository from "@/pages/app/BrowseRepository"
import AboutUs from "@/pages/app/AboutUs"
import ContactUs from "@/pages/app/ContactUs"

/**
 * ProtectedRoute — renders `element` only if the authenticated user has one of
 * the `allowedRoles`. Falls back to `/dashboard` with an unauthorised redirect.
 * Uses the cached user from localStorage for instant render (no flicker), and
 * re-validates against the server in the background.
 */
function ProtectedRoute({ element, allowedRoles }) {
  const [status, setStatus] = useState("loading"); // "loading" | "ok" | "denied"

  useEffect(() => {
    // Quick check from cache
    const cached = (() => {
      try { return JSON.parse(localStorage.getItem("user")); } catch { return null; }
    })();
    if (cached?.role && allowedRoles.includes(cached.role)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setStatus("ok");
    }
    // Always verify with server (server is the source of truth)
    fetch("/api/user", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((u) => {
        if (u && allowedRoles.includes(u.role)) {
          setStatus("ok");
        } else {
          setStatus("denied");
        }
      })
      .catch(() => setStatus("denied"));
  }, [allowedRoles]);

  if (status === "loading") return null; // brief blank avoids flash
  if (status === "denied") return <Navigate to="/dashboard" replace />;
  return element;
}

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/verify-otp" element={<VerifyOtp />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/user-dashboard" element={<Dashboard />} />
      <Route path="/chatbot" element={<Chatbot />} />
      <Route path="/profile" element={<Profile />} />
      <Route path="/browse-repository" element={<BrowseRepository />} />
      <Route path="/about" element={<AboutUs />} />
      <Route path="/contact" element={<ContactUs />} />

      {/* Admin routes — role-guarded */}
      <Route path="/admin-dashboard" element={
        <ProtectedRoute element={<AdminDashboard />} allowedRoles={["admin", "superadmin"]} />
      } />
      <Route path="/admin-dashboard/content" element={
        <ProtectedRoute element={<Repository />} allowedRoles={["admin", "superadmin"]} />
      } />

      {/* Super-Admin routes — superadmin only */}
      <Route path="/super-admin-dashboard" element={
        <ProtectedRoute element={<SuperAdminDashboard />} allowedRoles={["superadmin"]} />
      } />
      <Route path="/super-admin-dashboard/content" element={
        <ProtectedRoute element={<Repository />} allowedRoles={["superadmin"]} />
      } />
    </Routes>
  )
}
