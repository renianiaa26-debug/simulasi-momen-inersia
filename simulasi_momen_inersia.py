"""
Simulasi Momen Inersia dan Energi Kinetik Rotasi (Streamlit)
Jalankan:  streamlit run momen_inersia_rotasi.py
Kebutuhan: pip install streamlit numpy pandas matplotlib
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle

st.set_page_config(page_title="Simulasi Momen Inersia", page_icon="🌀", layout="wide")

# ----------------------------------------------------------------------------
# Data dan fungsi dasar
# ----------------------------------------------------------------------------
CINCIN = "Cincin"
SILINDER = "Silinder pejal"
BOLA = "Bola pejal"
BATANG = "Batang (poros di pusat)"
SEMUA = [CINCIN, SILINDER, BOLA, BATANG]

WARNA = {CINCIN: "#f58518", SILINDER: "#4c78a8", BOLA: "#54a24b", BATANG: "#e45756"}
RUMUS = {
    CINCIN: "I = M R²",
    SILINDER: "I = ½ M R²",
    BOLA: "I = 2/5 M R²",
    BATANG: "I = 1/12 M L²",
}


def momen_inersia(bentuk, M, R, L):
    if bentuk == CINCIN:
        return M * R**2
    if bentuk == SILINDER:
        return 0.5 * M * R**2
    if bentuk == BOLA:
        return 0.4 * M * R**2
    return M * L**2 / 12


def gambar_benda(ax, bentuk, R, L, theta, batas):
    """Menggambar benda (tampak depan, poros tegak lurus bidang gambar) yang berputar sebesar theta."""
    c = WARNA[bentuk]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if bentuk == CINCIN:
        ax.add_patch(Circle((0, 0), R, fill=False, ec=c, lw=7))
    elif bentuk == SILINDER:
        ax.add_patch(Circle((0, 0), R, fc=c, ec="black", alpha=0.55))
    elif bentuk == BOLA:
        for f in (1.0, 0.66, 0.33):
            ax.add_patch(Circle((0, 0), R * f, fc=c, ec="none", alpha=0.35))
        ax.add_patch(Circle((0, 0), R, fill=False, ec="black"))

    if bentuk == BATANG:
        ax.plot(
            [-L / 2 * cos_t, L / 2 * cos_t],
            [-L / 2 * sin_t, L / 2 * sin_t],
            color=c, lw=9, solid_capstyle="butt",
        )
        ujung = (L / 2 * cos_t, L / 2 * sin_t)
    else:
        ujung = (R * cos_t, R * sin_t)
        ax.plot([0, ujung[0]], [0, ujung[1]], color="black", lw=2)
    ax.plot(ujung[0], ujung[1], "o", color="red", ms=8)  # penanda untuk melihat putaran
    ax.plot(0, 0, "k.", ms=6)
    ax.set_xlim(-batas, batas)
    ax.set_ylim(-batas, batas)
    ax.set_aspect("equal")
    ax.axis("off")


def I_numerik(bentuk, M, R, L, N):
    """Hampiran I = Σ mᵢ rᵢ² dengan N potongan (pendekatan untuk I = ∫ r² dm)."""
    i = np.arange(N)
    if bentuk == CINCIN:
        return np.sum((M / N) * np.full(N, R) ** 2)
    if bentuk == SILINDER:  # cincin-cincin tipis konsentris
        dr = R / N
        r = (i + 0.5) * dr
        dm = 2 * M * r * dr / R**2
        return np.sum(r**2 * dm)
    if bentuk == BOLA:  # cakram-cakram tipis yang ditumpuk
        dz = 2 * R / N
        z = -R + (i + 0.5) * dz
        rho2 = R**2 - z**2
        dm = 3 * M * rho2 * dz / (4 * R**3)
        return np.sum(0.5 * dm * rho2)
    dx = L / N  # batang: potongan-potongan kecil
    x = -L / 2 + (i + 0.5) * dx
    return np.sum(x**2 * (M / N))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")
bentuk = st.sidebar.selectbox("Bentuk benda", SEMUA)
M = st.sidebar.slider("Massa M (kg)", 0.5, 20.0, 2.0, 0.5)
R = st.sidebar.slider("Jari-jari R (m)", 0.05, 1.0, 0.30, 0.01)
L = st.sidebar.slider("Panjang batang L (m)", 0.1, 3.0, 0.60, 0.05, help="Dipakai untuk benda berbentuk batang.")
omega = st.sidebar.slider("Kecepatan sudut ω (rad/s)", 0.5, 20.0, 5.0, 0.5)

I_sel = momen_inersia(bentuk, M, R, L)
I_all = {b: momen_inersia(b, M, R, L) for b in SEMUA}
batas = 1.25 * max(R, L / 2)

# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.title("🌀 Simulasi Momen Inersia dan Energi Kinetik Rotasi")
st.caption("Bandingkan karakteristik rotasi cincin, silinder pejal, bola pejal, dan batang dalam satu simulasi.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔄 Simulasi Rotasi", "📊 Perbandingan", "⚖️ Distribusi Massa", "📘 Rumus & Integrasi Numerik"]
)

# ------------------------------ TAB 1: simulasi -----------------------------
with tab1:
    kiri, kanan = st.columns([1.1, 1])
    with kanan:
        st.subheader(bentuk)
        st.markdown(f"Rumus: `{RUMUS[bentuk]}`")
        Ek = 0.5 * I_sel * omega**2
        a1, a2 = st.columns(2)
        a1.metric("Momen inersia I", f"{I_sel:.4g} kg·m²")
        a2.metric("Energi kinetik rotasi Ek", f"{Ek:.4g} J")
        a3, a4 = st.columns(2)
        a3.metric("Momentum sudut Iω", f"{I_sel * omega:.4g} kg·m²/s")
        a4.metric("Periode T = 2π/ω", f"{2 * np.pi / omega:.3g} s")
        st.markdown("Energi kinetik rotasi: `Ek = ½ I ω²`")
        durasi = st.slider("Durasi simulasi (s)", 1.0, 10.0, 4.0, 0.5)
        t_now = st.slider("Waktu t (s)", 0.0, float(durasi), 0.0, 0.05)
        jalan = st.button("▶ Putar animasi", key="anim1")
        st.caption(f"Dalam {durasi:g} s benda berputar sebanyak {omega * durasi / (2 * np.pi):.3g} putaran.")

    def frame_tunggal(t):
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        gambar_benda(ax, bentuk, R, L, omega * t, batas)
        ax.set_title(f"t = {t:.2f} s    θ = {np.degrees(omega * t):.0f}°", fontsize=10)
        return fig

    with kiri:
        tempat = st.empty()
        if jalan:
            for tt in np.linspace(0, durasi, 60):
                fig = frame_tunggal(tt)
                tempat.pyplot(fig)
                plt.close(fig)
                time.sleep(0.03)
        else:
            fig = frame_tunggal(t_now)
            tempat.pyplot(fig)
            plt.close(fig)

# --------------------------- TAB 2: perbandingan ----------------------------
with tab2:
    st.subheader("Perbandingan benda dengan massa, jari-jari, dan ω yang sama")
    st.write(f"M = {M:g} kg, R = {R:g} m, L = {L:g} m, ω = {omega:g} rad/s")

    df = pd.DataFrame(
        {
            "Bentuk": SEMUA,
            "Rumus": [RUMUS[b] for b in SEMUA],
            "I (kg·m²)": [round(I_all[b], 5) for b in SEMUA],
            "I / (M R²)": [round(I_all[b] / (M * R**2), 4) for b in SEMUA],
            "Ek (J)": [round(0.5 * I_all[b] * omega**2, 4) for b in SEMUA],
            "Iω (kg·m²/s)": [round(I_all[b] * omega, 4) for b in SEMUA],
        }
    )
    st.dataframe(df, hide_index=True)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    nilai_I = [I_all[b] for b in SEMUA]
    nilai_Ek = [0.5 * I_all[b] * omega**2 for b in SEMUA]
    warna = [WARNA[b] for b in SEMUA]
    axs[0].bar(SEMUA, nilai_I, color=warna)
    axs[0].set_title("Momen inersia I")
    axs[0].set_ylabel("kg·m²")
    axs[1].bar(SEMUA, nilai_Ek, color=warna)
    axs[1].set_title(f"Energi kinetik rotasi pada ω = {omega:g} rad/s")
    axs[1].set_ylabel("J")
    for ax in axs:
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    w = np.linspace(0, 20, 200)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for b in SEMUA:
        ax.plot(w, 0.5 * I_all[b] * w**2, color=WARNA[b], lw=2, label=b)
        ax.plot(omega, 0.5 * I_all[b] * omega**2, "o", color=WARNA[b])
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("Ek (J)")
    ax.set_title("Ek = ½ I ω²: benda dengan I lebih besar menyimpan energi lebih banyak")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    terbesar = max(SEMUA, key=lambda b: I_all[b])
    terkecil = min(SEMUA, key=lambda b: I_all[b])
    st.info(
        f"Pada pengaturan saat ini, momen inersia terbesar dimiliki **{terbesar}** dan terkecil **{terkecil}**. "
        "Untuk M dan R yang sama, urutan benda bundar adalah cincin, silinder pejal, lalu bola pejal."
    )

# ---------------------------- TAB 3: distribusi massa -----------------------
with tab3:
    st.subheader("Pengaruh distribusi massa terhadap gerak rotasi")
    st.write(
        "Semua benda bermassa sama dan diberi torsi yang sama. Benda yang massanya lebih jauh dari poros "
        "memiliki I lebih besar sehingga lebih sulit dipercepat."
    )
    c1, c2 = st.columns(2)
    tau = c1.slider("Torsi τ (N·m)", 0.05, 5.0, 0.5, 0.05)
    T2 = c2.slider("Lama torsi bekerja (s)", 1.0, 10.0, 4.0, 0.5)

    alpha = {b: tau / I_all[b] for b in SEMUA}
    rows = []
    for b in SEMUA:
        w_T = alpha[b] * T2
        th_T = 0.5 * alpha[b] * T2**2
        rows.append(
            {
                "Bentuk": b,
                "I (kg·m²)": round(I_all[b], 5),
                "α = τ/I (rad/s²)": round(alpha[b], 3),
                "ω pada t akhir (rad/s)": round(w_T, 3),
                "Putaran": round(th_T / (2 * np.pi), 3),
                "Ek = τθ (J)": round(tau * th_T, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    t = np.linspace(0, T2, 200)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for b in SEMUA:
        axs[0].plot(t, alpha[b] * t, color=WARNA[b], lw=2, label=b)
        axs[1].plot(t, 0.5 * alpha[b] * t**2, color=WARNA[b], lw=2, label=b)
    axs[0].set_title("Kecepatan sudut ω(t)")
    axs[0].set_ylabel("rad/s")
    axs[1].set_title("Posisi sudut θ(t)")
    axs[1].set_ylabel("rad")
    for ax in axs:
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Animasi: empat benda diberi torsi yang sama")
    t_b = st.slider("Waktu t (s)", 0.0, float(T2), 0.0, 0.05, key="t_banding")
    jalan2 = st.button("▶ Putar animasi", key="anim2")

    def frame_banding(tt):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, b in zip(axs, SEMUA):
            gambar_benda(ax, b, R, L, 0.5 * alpha[b] * tt**2, batas)
            ax.set_title(f"{b}\nω = {alpha[b] * tt:.2f} rad/s", fontsize=9)
        fig.suptitle(f"t = {tt:.2f} s", fontsize=11)
        return fig

    tempat2 = st.empty()
    if jalan2:
        for tt in np.linspace(0, T2, 60):
            fig = frame_banding(tt)
            tempat2.pyplot(fig)
            plt.close(fig)
            time.sleep(0.03)
    else:
        fig = frame_banding(t_b)
        tempat2.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Jika diberi energi kinetik yang sama")
    Ek0 = st.slider("Energi kinetik Ek (J)", 1.0, 100.0, 10.0, 1.0)
    w_sama = {b: np.sqrt(2 * Ek0 / I_all[b]) for b in SEMUA}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(SEMUA, [w_sama[b] for b in SEMUA], color=[WARNA[b] for b in SEMUA])
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(f"Kecepatan sudut untuk Ek = {Ek0:g} J, dari ω = √(2Ek/I)")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Untuk energi yang sama, benda dengan I kecil berputar lebih cepat.")

# ----------------------------- TAB 4: rumus ---------------------------------
with tab4:
    st.subheader("Rumus utama")
    st.markdown(
        """
- **Definisi momen inersia (partikel):** `I = Σ mᵢ rᵢ²`
- **Benda kontinu:** `I = ∫ r² dm`
- **Energi kinetik rotasi:** `Ek = ½ I ω²`
- **Torsi dan percepatan sudut:** `τ = I α`
        """
    )
    st.table(
        pd.DataFrame(
            {
                "Bentuk benda": ["Cincin", "Silinder pejal", "Bola pejal", "Batang (pusat)"],
                "Momen inersia": [RUMUS[b] for b in SEMUA],
            }
        ).set_index("Bentuk benda")
    )

    st.subheader(f"Dari Σ mᵢrᵢ² menuju ∫ r² dm: {bentuk}")
    st.write("Benda dibagi menjadi N potongan kecil, lalu I dihitung dengan menjumlahkan m·r² tiap potongan.")
    N = st.slider("Jumlah potongan N", 2, 500, 10)
    I_num = I_numerik(bentuk, M, R, L, N)
    galat = 100 * abs(I_num - I_sel) / I_sel
    n1, n2, n3 = st.columns(3)
    n1.metric("I numerik (Σ mᵢrᵢ²)", f"{I_num:.5g} kg·m²")
    n2.metric("I rumus (∫ r² dm)", f"{I_sel:.5g} kg·m²")
    n3.metric("Selisih relatif", f"{galat:.3g} %")

    daftar_N = np.unique(np.round(np.logspace(np.log10(2), np.log10(500), 30)).astype(int))
    hasil = [I_numerik(bentuk, M, R, L, int(n)) / I_sel for n in daftar_N]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.semilogx(daftar_N, hasil, "o-", color=WARNA[bentuk])
    ax.axhline(1, ls="--", color="gray")
    ax.axvline(N, ls=":", color="red")
    ax.set_xlabel("jumlah potongan N")
    ax.set_ylabel("I numerik / I rumus")
    ax.set_title("Hampiran jumlah mendekati nilai integral saat N membesar")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    if bentuk == CINCIN:
        st.caption("Pada cincin tipis semua massa berada pada jarak R dari poros, sehingga jumlahnya langsung tepat bernilai MR².")

st.divider()
st.caption("Benda dianggap tegar dan homogen. Poros melalui pusat massa dan tegak lurus bidang gambar.")