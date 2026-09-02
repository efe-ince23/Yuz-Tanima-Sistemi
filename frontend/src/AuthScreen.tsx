import { Eye, EyeOff, Fingerprint, LoaderCircle, LockKeyhole, LogIn, UserPlus } from "lucide-react";
import { type FormEvent, useState } from "react";

import { login, register } from "./api";
import type { AppUser } from "./types";

interface AuthScreenProps {
  onAuthenticated: (user: AppUser) => void;
}

export default function AuthScreen({ onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData(event.currentTarget);
    try {
      const response = mode === "login"
        ? await login({
            identifier: String(form.get("identifier") ?? ""),
            password: String(form.get("password") ?? ""),
          })
        : await register({
            username: String(form.get("username") ?? ""),
            email: String(form.get("email") ?? ""),
            full_name: String(form.get("fullName") ?? ""),
            password: String(form.get("password") ?? ""),
          });
      onAuthenticated(response.user);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Giriş yapılamadı.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-product" aria-label="Yüz Tanıma Sistemi">
        <div className="auth-brand-mark"><Fingerprint size={38} /></div>
        <p className="eyebrow">Güvenli kimlik eşleştirme</p>
        <h1>Yüz Tanıma Sistemi</h1>
        <p>Fotoğraf, video ve canlı kamera analizlerini tek çalışma alanında yönetin.</p>
        <div className="auth-signal-row">
          <span>YOLOv8-Face</span><span>ArcFace R50</span><span>CUDA</span>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-form-wrap">
          <div className="auth-heading">
            <span className="auth-heading-icon"><LockKeyhole size={22} /></span>
            <div><p>{mode === "login" ? "Oturum aç" : "Yeni hesap"}</p><h2>{mode === "login" ? "Tekrar hoş geldiniz" : "Çalışma alanınızı oluşturun"}</h2></div>
          </div>

          <div className="auth-mode-tabs" role="tablist" aria-label="Hesap işlemleri">
            <button type="button" className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(null); }}>Giriş</button>
            <button type="button" className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(null); }}>Kayıt ol</button>
          </div>

          <form className="auth-form" onSubmit={submit}>
            {mode === "register" && (
              <>
                <label>Ad soyad<input name="fullName" autoComplete="name" required minLength={2} /></label>
                <div className="auth-field-row">
                  <label>Kullanıcı adı<input name="username" autoComplete="username" required minLength={3} pattern="[a-zA-Z0-9_.-]+" /></label>
                  <label>E-posta<input name="email" type="email" autoComplete="email" required /></label>
                </div>
              </>
            )}
            {mode === "login" && (
              <label>Kullanıcı adı veya e-posta<input name="identifier" autoComplete="username" required autoFocus /></label>
            )}
            <label>Parola
              <span className="password-field">
                <input name="password" type={showPassword ? "text" : "password"} autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={mode === "register" ? 10 : 1} />
                <button type="button" onClick={() => setShowPassword((value) => !value)} title={showPassword ? "Parolayı gizle" : "Parolayı göster"}>
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </span>
            </label>
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-submit" type="submit" disabled={busy}>
              {busy ? <LoaderCircle className="spin" size={19} /> : mode === "login" ? <LogIn size={19} /> : <UserPlus size={19} />}
              {busy ? "İşleniyor" : mode === "login" ? "Giriş yap" : "Hesap oluştur"}
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}
