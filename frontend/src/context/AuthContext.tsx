import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { getMe, login as apiLogin, logout as apiLogout, signup as apiSignup } from '../api';

interface AuthContextValue {
  email: string | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

const TOKEN_KEY = 'medassist_token';
const EMAIL_KEY = 'medassist_email';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [email, setEmail] = useState<string | null>(() => localStorage.getItem(EMAIL_KEY));
  const [isLoading, setIsLoading] = useState(true);

  const persistAuth = (accessToken: string, userEmail: string) => {
    localStorage.setItem(TOKEN_KEY, accessToken);
    localStorage.setItem(EMAIL_KEY, userEmail);
    setToken(accessToken);
    setEmail(userEmail);
  };

  const clearAuth = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EMAIL_KEY);
    setToken(null);
    setEmail(null);
  };

  useEffect(() => {
    const verifySession = async () => {
      if (!token) {
        setIsLoading(false);
        return;
      }
      try {
        const user = await getMe(token);
        setEmail(user.email);
        localStorage.setItem(EMAIL_KEY, user.email);
      } catch {
        clearAuth();
      } finally {
        setIsLoading(false);
      }
    };
    verifySession();
  }, [token]);

  const login = useCallback(async (userEmail: string, password: string) => {
    const data = await apiLogin(userEmail, password);
    persistAuth(data.access_token, data.user.email);
  }, []);

  const signup = useCallback(async (userEmail: string, password: string) => {
    const data = await apiSignup(userEmail, password);
    persistAuth(data.access_token, data.user.email);
  }, []);

  const logout = useCallback(async () => {
    if (token) {
      try {
        await apiLogout(token);
      } catch {
        // Clear local session even if the API call fails
      }
    }
    clearAuth();
  }, [token]);

  return (
    <AuthContext.Provider
      value={{
        email,
        token,
        isAuthenticated: !!token,
        isLoading,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
