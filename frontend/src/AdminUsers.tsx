import { CheckCircle2, LoaderCircle, Search, ShieldCheck, UserRound, UserX } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { getUsers, setUserActive } from "./api";
import type { AppUser } from "./types";

interface AdminUsersProps {
  currentUserId: string;
}

export default function AdminUsers({ currentUserId }: AdminUsersProps) {
  const [users, setUsers] = useState<AppUser[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [changingId, setChangingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try { setUsers(await getUsers()); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Kullanıcılar yüklenemedi."); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const filtered = useMemo(() => {
    const value = query.trim().toLocaleLowerCase("tr-TR");
    if (!value) return users;
    return users.filter((user) => `${user.full_name} ${user.username} ${user.email}`.toLocaleLowerCase("tr-TR").includes(value));
  }, [query, users]);

  const toggleUser = async (user: AppUser) => {
    setChangingId(user.id);
    setError(null);
    try {
      const updated = await setUserActive(user.id, !user.is_active);
      setUsers((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Kullanıcı güncellenemedi.");
    } finally {
      setChangingId(null);
    }
  };

  return (
    <section className="users-page">
      <header className="page-heading users-heading">
        <div><p className="eyebrow">Yönetim</p><h1>Kullanıcı Hesapları</h1><p>{users.length} hesap · {users.filter((user) => user.is_active).length} aktif</p></div>
      </header>
      <div className="users-toolbar">
        <Search size={19} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ad, kullanıcı adı veya e-posta ara" />
      </div>
      {error && <div className="auth-error" role="alert">{error}</div>}
      <div className="users-list">
        {loading ? <div className="users-empty"><LoaderCircle className="spin" size={26} /> Hesaplar yükleniyor</div> : filtered.map((user) => (
          <article className="user-row" key={user.id}>
            <span className={`user-avatar ${user.role}`}><UserRound size={21} /></span>
            <div className="user-main"><strong>{user.full_name}</strong><span>@{user.username} · {user.email}</span></div>
            <span className={`role-badge ${user.role}`}>{user.role === "admin" ? <ShieldCheck size={15} /> : <UserRound size={15} />}{user.role === "admin" ? "Yönetici" : "Kullanıcı"}</span>
            <span className={`account-state ${user.is_active ? "active" : "inactive"}`}>{user.is_active ? "Aktif" : "Pasif"}</span>
            <button className="icon-button" type="button" disabled={user.id === currentUserId || changingId === user.id} onClick={() => void toggleUser(user)} title={user.is_active ? "Hesabı pasifleştir" : "Hesabı etkinleştir"}>
              {changingId === user.id ? <LoaderCircle className="spin" size={18} /> : user.is_active ? <UserX size={18} /> : <CheckCircle2 size={18} />}
            </button>
          </article>
        ))}
        {!loading && filtered.length === 0 && <div className="users-empty">Eşleşen hesap bulunamadı.</div>}
      </div>
    </section>
  );
}
