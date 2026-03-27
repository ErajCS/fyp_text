import { useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";

/**
 * useAuth — Validates current session against /api/user.
 * On 401 (not logged in) → redirects to /login.
 * On role mismatch → redirects to the correct dashboard.
 *
 * Usage:
 *   const user = useAuth("admin");
 *   const user = useAuth(["admin","superadmin"]);
 */
export default function useAuth(requiredRole = null) {
    const navigate = useNavigate();

    const getStoredUser = useCallback(() => {
        try {
            return JSON.parse(localStorage.getItem("user") || "null");
        } catch {
            return null;
        }
    }, []);

    useEffect(() => {
        let cancelled = false;

        fetch("/api/user", { credentials: "include" })
            .then((r) => {
                if (cancelled) return null;
                if (r.status === 401) {
                    localStorage.removeItem("user");
                    navigate("/login");
                    return null;
                }
                return r.json();
            })
            .then((data) => {
                if (!data || cancelled) return;

                // /api/user returns the user object directly (no `success` wrapper)
                // It has id, name, email, role, etc.
                const userRole = data.role || data.user?.role;
                if (!userRole) {
                    // Response doesn't have a role — not a valid user session
                    navigate("/login");
                    return;
                }

                // Update cached user
                const userObj = data.role ? data : data.user;
                localStorage.setItem("user", JSON.stringify(userObj));

                // Role check
                if (requiredRole) {
                    const allowed = Array.isArray(requiredRole) ? requiredRole : [requiredRole];
                    if (!allowed.includes(userRole)) {
                        // Wrong role — redirect to the right dashboard
                        if (userRole === "superadmin") navigate("/super-admin-dashboard");
                        else if (userRole === "admin") navigate("/admin-dashboard");
                        else navigate("/dashboard");
                    }
                }
            })
            .catch(() => {
                if (!cancelled) navigate("/login");
            });

        return () => { cancelled = true; };
    }, [navigate, requiredRole]);

    return getStoredUser();
}
