# -*- coding: utf-8 -*-
"""
Parte B.
"""
import numpy as np
import pandas as pd

import simulacion_trm as sim    # parte A: trayectorias de la TRM y tabla del crédito

#  PARÁMETROS 
# Cotización de mercado (B.1): Cbonds, USD/COP FX Forward Rate (Mid), 27/10/2026
SPOT_MERCADO = 3132.07          # S/N
FWD_1Y_MERCADO = 3386.30        # 1 año
I_USD = 0.0447                  # Tesoro de EE. UU. a 1 año (Fed H.15, 26/10/2026)
# Tasa en pesos implícita en el forward de mercado (base del modelo)
I_COP = (FWD_1Y_MERCADO / SPOT_MERCADO) * (1 + I_USD) - 1

COBERTURA = 0.75
TRIM_INICIO = 24                # fin del año 6
PACTOS = [24, 28, 32, 36]
VENCIMIENTOS = [28, 32, 36, 40]
UMBRAL_EXTREMO = 0.20           # depreciación anual > 20 % = pérdida cambiaria extrema
TOLERANCIA_PPTO = 0.10          # «fracaso»: costo cubierto > presupuesto + 10 %
TASA_DESCUENTO_EA = 0.13        # para el VPN en pesos (TES a 10 años > 13 %)
NU_STRESS = 4                   # t de Student de estrés (colas mucho más gruesas)
MOSTRAR_GRAFICAS = True         # dibuja las gráficas (en Colab/Jupyter aparecen debajo de la celda)
# 


def prima(i_cop, i_usd):
    return (1 + i_cop) / (1 + i_usd) - 1


def evaluar(paths, cuota, i_cop=I_COP, i_usd=I_USD, variante="rodante"):
    """Resultados por trayectoria de la estrategia de cobertura."""
    n = paths.shape[0]
    p = prima(i_cop, i_usd)
    N = COBERTURA * 4 * cuota                                  # nominal por forward
    S = paths
    F = np.empty((n, 4)); Sv = np.empty((n, 4)); Sp = np.empty((n, 4))
    U = np.empty((n, 4))
    for k, (c, m) in enumerate(zip(PACTOS, VENCIMIENTOS)):
        Sp[:, k] = S[:, c]
        Sv[:, k] = S[:, m]
        if variante == "rodante":
            F[:, k] = S[:, c] * (1 + p)
        else:                                                    # strip fijado en t=24
            F[:, k] = S[:, TRIM_INICIO] * (1 + p) ** (k + 1)
        U[:, k] = cuota * S[:, c + 1:m + 1].sum(axis=1)          # costo del año sin cobertura
    liq = N * (Sv - F)                                           # liquidación NDF (COP)
    H = U - liq                                                  # costo del año con cobertura
    B = 4 * cuota * F                                            # presupuesto al forward
    # flujos trimestrales del crédito (pagos positivos = salidas)
    pagos_sin = cuota * S[:, 1:]
    pagos_con = pagos_sin.copy()
    for k, m in enumerate(VENCIMIENTOS):
        pagos_con[:, m - 1] -= liq[:, k]
    return dict(F=F, Sv=Sv, Sp=Sp, U=U, H=H, B=B, liq=liq, N=N, p=p,
                pagos_sin=pagos_sin, pagos_con=pagos_con)


def tir_ea(desembolso, pagos):
    """TIR trimestral (vectorizada por bisección) expresada como tasa efectiva anual."""
    n = pagos.shape[0]
    lo = np.full(n, -0.2); hi = np.full(n, 0.5)
    t = np.arange(1, pagos.shape[1] + 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        vpn = desembolso - (pagos / (1 + mid[:, None]) ** t).sum(axis=1)
        hi = np.where(vpn > 0, mid, hi)
        lo = np.where(vpn > 0, lo, mid)
    return (1 + mid) ** 4 - 1


def vpn(pagos, tasa_ea=TASA_DESCUENTO_EA):
    iq = (1 + tasa_ea) ** 0.25 - 1
    t = np.arange(1, pagos.shape[1] + 1)
    return (pagos / (1 + iq) ** t).sum(axis=1)


def cvar(x, q=0.95):
    v = np.quantile(x, q)
    return x[x >= v].mean()


def resumen(paths, cuota, etiqueta, desembolso):
    r = evaluar(paths, cuota)
    rs = evaluar(paths, cuota, variante="strip")
    filas = {}
    dep = r["Sv"] / r["Sp"] - 1                                  # depreciación del año del forward
    # --- B.2: proporciones por forward (agrupando los cuatro años) ---
    prot = (r["Sv"] > r["F"]).mean()
    opor = (r["Sv"] < r["F"]).mean()
    ext = dep > UMBRAL_EXTREMO
    filas["Proporción con protección (spot al vencimiento > forward)"] = prot
    filas["Proporción con costo de oportunidad (spot < forward)"] = opor
    filas["Proporción con pérdida cambiaria extrema (depreciación anual > 20 %)"] = ext.mean()
    filas["Proporción con depreciación anual > 30 %"] = (dep > 0.30).mean()
    filas["Proporción con apreciación anual > 20 % (costo de oportunidad extremo)"] = (dep < -0.20).mean()
    filas["En años extremos: sobrecosto sin cobertura vs presupuesto"] = (r["U"][ext] / r["B"][ext] - 1).mean() if ext.any() else np.nan
    filas["En años extremos: sobrecosto con cobertura vs presupuesto"] = (r["H"][ext] / r["B"][ext] - 1).mean() if ext.any() else np.nan
    base_pacto = 4 * cuota * r["Sp"]
    filas["En años extremos: sobrecosto sin cobertura vs spot del pacto"] = (r["U"][ext] / base_pacto[ext] - 1).mean()
    filas["En años extremos: sobrecosto con cobertura vs spot del pacto"] = (r["H"][ext] / base_pacto[ext] - 1).mean()
    filas["En años extremos: pérdida sin cobertura vs spot del pacto (COP)"] = (r["U"][ext] - base_pacto[ext]).mean()
    filas["En años extremos: pérdida con cobertura vs spot del pacto (COP)"] = (r["H"][ext] - base_pacto[ext]).mean()
    perd_der = r["liq"] < 0
    fr_sin = r["U"] > r["B"] * (1 + TOLERANCIA_PPTO)
    fr_con = r["H"] > r["B"] * (1 + TOLERANCIA_PPTO)
    filas["Pérdida del derivado (liquidación negativa)"] = perd_der.mean()
    filas["Pérdida del derivado y cobertura exitosa"] = (perd_der & ~fr_con).mean()
    filas["Ganancia del derivado y cobertura exitosa"] = (~perd_der & ~fr_con).mean()
    filas["Fracaso de la cobertura (costo cubierto > presupuesto + 10 %)"] = fr_con.mean()
    filas["Sin cobertura: costo > presupuesto + 10 %"] = fr_sin.mean()
    # --- B.3: costo total del crédito ---
    for nombre, pagos in [("sin cobertura", r["pagos_sin"]), ("con cobertura rodante", r["pagos_con"]),
                          ("con strip fijado en año 6", rs["pagos_con"])]:
        tot = pagos.sum(axis=1)
        a710 = pagos[:, 24:].sum(axis=1)
        filas[f"Costo total COP – {nombre}: media"] = tot.mean()
        filas[f"Costo total COP – {nombre}: desv. estándar"] = tot.std()
        filas[f"Costo total COP – {nombre}: P5"] = np.quantile(tot, 0.05)
        filas[f"Costo total COP – {nombre}: P95"] = np.quantile(tot, 0.95)
        filas[f"Costo total COP – {nombre}: P99"] = np.quantile(tot, 0.99)
        filas[f"Costo total COP – {nombre}: CVaR 95 %"] = cvar(tot)
        filas[f"Costo años 7-10 – {nombre}: media"] = a710.mean()
        filas[f"Costo años 7-10 – {nombre}: desv. estándar"] = a710.std()
        filas[f"Costo años 7-10 – {nombre}: P95"] = np.quantile(a710, 0.95)
        filas[f"Costo años 7-10 – {nombre}: CVaR 95 %"] = cvar(a710)
        filas[f"VPN pagos al 13 % EA – {nombre}: media"] = vpn(pagos).mean()
        tir = tir_ea(desembolso, pagos)
        filas[f"Costo efectivo anual – {nombre}: media"] = tir.mean()
        filas[f"Costo efectivo anual – {nombre}: P95"] = np.quantile(tir, 0.95)
    # riesgo visto desde el fin del año 6 (cuando se toma la decisión): costo de los años
    # 7-10 relativo al costo de las 16 cuotas a la TRM del trimestre 24
    base24 = 16 * cuota * paths[:, TRIM_INICIO]
    for nombre, pagos in [("sin cobertura", r["pagos_sin"]), ("con cobertura rodante", r["pagos_con"]),
                          ("con strip fijado en año 6", rs["pagos_con"])]:
        rel = pagos[:, 24:].sum(axis=1) / base24 - 1
        filas[f"Desde el año 6 – {nombre}: sobrecosto medio vs TRM del año 6"] = rel.mean()
        filas[f"Desde el año 6 – {nombre}: desv. estándar"] = rel.std()
        filas[f"Desde el año 6 – {nombre}: P95"] = np.quantile(rel, 0.95)
        filas[f"Desde el año 6 – {nombre}: P99"] = np.quantile(rel, 0.99)
    # desviación anual frente al presupuesto (estabilidad presupuestal, por año del forward)
    dev_sin = r["U"] / r["B"] - 1
    dev_con = r["H"] / r["B"] - 1
    filas["Desv. estándar de la desviación anual vs presupuesto – sin cobertura"] = dev_sin.std()
    filas["Desv. estándar de la desviación anual vs presupuesto – con cobertura"] = dev_con.std()
    filas["Resultado acumulado de los 4 forwards: media (COP)"] = r["liq"].sum(axis=1).mean()
    filas["Resultado acumulado de los 4 forwards: P5 (COP)"] = np.quantile(r["liq"].sum(axis=1), 0.05)
    filas["Resultado acumulado de los 4 forwards: P95 (COP)"] = np.quantile(r["liq"].sum(axis=1), 0.95)
    filas["Prob. de que la cobertura rodante cueste más que no cubrir"] = (r["liq"].sum(axis=1) < 0).mean()
    filas["Strip: resultado acumulado de los 4 forwards, media (COP)"] = rs["liq"].sum(axis=1).mean()
    filas["Strip: prob. de que cueste más que no cubrir"] = (rs["liq"].sum(axis=1) < 0).mean()
    return pd.Series(filas, name=etiqueta), r


def por_forward(r):
    filas = []
    for k in range(4):
        filas.append({
            "forward": k + 1, "año": 7 + k,
            "F_mediana": np.median(r["F"][:, k]), "S_venc_mediana": np.median(r["Sv"][:, k]),
            "prob_proteccion": (r["Sv"][:, k] > r["F"][:, k]).mean(),
            "prob_costo_oportunidad": (r["Sv"][:, k] < r["F"][:, k]).mean(),
            "prob_extrema": (r["Sv"][:, k] / r["Sp"][:, k] - 1 > UMBRAL_EXTREMO).mean(),
            "liq_media": r["liq"][:, k].mean(), "liq_p5": np.quantile(r["liq"][:, k], 0.05),
            "liq_p95": np.quantile(r["liq"][:, k], 0.95),
            "costo_sin_media": r["U"][:, k].mean(), "costo_con_media": r["H"][:, k].mean(),
            "costo_sin_sd": r["U"][:, k].std(), "costo_con_sd": r["H"][:, k].std(),
        })
    return pd.DataFrame(filas)


def _num(x, dec=2):
    """Formato colombiano: 3.174,67"""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def graficas(paths_n, cuota, r_n, r_t, r_t4, rs_n, sens, rep, nu):
    """Gráficas de la parte B."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fmt0 = FuncFormatter(lambda v, _: _num(v, 0))
    fmt1 = FuncFormatter(lambda v, _: _num(v, 1))
    pct = FuncFormatter(lambda v, _: _num(v, 0) + " %")
    AZ, GR, VE, RO, NA = "#1f5f9e", "#8a8a8a", "#2e8b57", "#c0392b", "#e67e22"
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.25, top=0.92)

    # (a) Forward pactado vs TRM al vencimiento
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(3)
    j = rng.choice(r_n["F"].size, 4000, replace=False)
    F = r_n["F"].ravel()[j]; Sv = r_n["Sv"].ravel()[j]
    prot = Sv > F
    ax.scatter(F[prot], Sv[prot], s=4, alpha=0.4, color=VE, label=f"Protección ({(r_n['Sv'] > r_n['F']).mean()*100:.1f} %)".replace(".", ","))
    ax.scatter(F[~prot], Sv[~prot], s=4, alpha=0.4, color=RO, label=f"Costo de oportunidad ({(r_n['Sv'] < r_n['F']).mean()*100:.1f} %)".replace(".", ","))
    lim = [min(F.min(), Sv.min()), np.percentile(np.r_[F, Sv], 99.5)]
    ax.plot(lim, lim, color="#333", lw=1, ls="--", label="TRM al vencimiento = forward")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("(a) Forward pactado vs TRM al vencimiento\n(4.000 contratos simulados, normal)", fontsize=11)
    ax.set_xlabel("Forward pactado (COP/USD)"); ax.set_ylabel("TRM al vencimiento (COP/USD)")
    ax.xaxis.set_major_formatter(fmt0); ax.yaxis.set_major_formatter(fmt0); ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (b) TRM al vencimiento / forward − 1, normal vs t
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-45, 60, 85)
    for r, lab, col in [(r_n, "Normal", AZ), (r_t, f"t (ν = {_num(nu, 1)})", NA), (r_t4, "t de estrés (ν = 4)", RO)]:
        ax.hist((r["Sv"] / r["F"] - 1).ravel() * 100, bins=bins, histtype="step", lw=1.5, density=True, color=col, label=lab)
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(1.5, ax.get_ylim()[1] * 0.93, "protección →", fontsize=8, color=VE)
    ax.text(-1.5, ax.get_ylim()[1] * 0.93, "← costo de oportunidad", fontsize=8, color=RO, ha="right")
    ax.set_title("(b) Diferencia entre la TRM al vencimiento y el forward", fontsize=11)
    ax.set_xlabel("TRM al vencimiento / forward − 1"); ax.xaxis.set_major_formatter(pct); ax.set_yticks([])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    # (c) Resultado acumulado de los 4 forwards
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-80, 80, 81)
    for r, lab, col in [(r_n, "Normal", AZ), (r_t, f"t (ν = {_num(nu, 1)})", NA)]:
        liq = r["liq"].sum(axis=1) / 1e6
        ax.hist(liq, bins=bins, histtype="step", lw=1.5, density=True, color=col,
                label=f"{lab}: media {_num(liq.mean(), 1)} millones")
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.set_title("(c) Resultado acumulado de los cuatro forwards", fontsize=11)
    ax.set_xlabel("Millones de COP (positivo = la cobertura ahorra)"); ax.set_yticks([])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.xaxis.set_major_formatter(fmt0); ax.legend(fontsize=8, frameon=False, loc="upper right")

    # (d) Costo años 7-10 visto desde el año 6
    ax = fig.add_subplot(gs[1, 0])
    base24 = 16 * cuota * paths_n[:, TRIM_INICIO]
    bins = np.linspace(-40, 90, 66)
    for pagos, lab, col in [(r_n["pagos_sin"], "Sin cobertura", GR), (r_n["pagos_con"], "Cobertura rodante (enunciado)", AZ),
                            (rs_n["pagos_con"], "Strip fijado en el año 6", VE)]:
        rel = (pagos[:, 24:].sum(axis=1) / base24 - 1) * 100
        ax.hist(rel, bins=bins, histtype="step", lw=1.6, density=True, color=col,
                label=f"{lab}: desv. est. {_num(rel.std(), 1)} %")
    ax.set_title("(d) Costo de los años 7-10 frente a la TRM del año 6", fontsize=11)
    ax.set_xlabel("Sobrecosto"); ax.xaxis.set_major_formatter(pct); ax.set_yticks([])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # (e) Desviación de cada año frente a su presupuesto
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-35, 35, 71)
    d_sin = (r_n["U"] / r_n["B"] - 1).ravel() * 100
    d_con = (r_n["H"] / r_n["B"] - 1).ravel() * 100
    ax.hist(d_sin, bins=bins, color=GR, alpha=0.55, density=True, label=f"Sin cobertura (> +10 %: {_num((d_sin > 10).mean()*100, 2)} %)")
    ax.hist(d_con, bins=bins, color=AZ, alpha=0.6, density=True, label=f"Con cobertura (> +10 %: {_num((d_con > 10).mean()*100, 2)} %)")
    ax.axvline(10, color=RO, ls="--", lw=1.2, label="Presupuesto + 10 % (umbral de fracaso)")
    ax.set_title("(e) Costo de cada año frente a su presupuesto al forward", fontsize=11)
    ax.set_xlabel("Desviación"); ax.xaxis.set_major_formatter(pct); ax.set_yticks([])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (f) Sensibilidad a la prima forward
    ax = fig.add_subplot(gs[1, 2])
    s = sens[np.isclose(sens.i_USD, I_USD)].sort_values("prima_forward")
    ax.plot(s.prima_forward * 100, s.resultado_forwards_media / 1e6, "o-", color=AZ, label="Cobertura rodante")
    ax.plot(s.prima_forward * 100, s.strip_resultado_media / 1e6, "s-", color=VE, label="Strip")
    ax.axhline(0, color="#333", lw=0.8)
    ax.axvline(prima(I_COP, I_USD) * 100, color=RO, ls="--", lw=1.2, label=f"Prima de mercado ({_num(prima(I_COP, I_USD)*100, 1)} %)")
    ax.set_title(f"(f) Resultado esperado de los forwards según la prima\n(i_USD = {_num(I_USD*100, 2)} %)", fontsize=11)
    ax.set_xlabel("Prima forward anual"); ax.set_ylabel("Millones de COP")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(fmt0); ax.legend(fontsize=8, frameon=False, loc="lower left")

    # (g) Escenarios representativos (normal)
    p = prima(I_COP, I_USD)
    q = np.arange(paths_n.shape[1])
    for k, nombre in enumerate(["Favorable", "Central", "Adverso"]):
        ax = fig.add_subplot(gs[2, k])
        S = rep[f"{nombre} – Normal"]
        ax.plot(q[20:], S[20:], color="#333", lw=1.6, label="TRM simulada")
        total = 0
        for i, (c, m) in enumerate(zip(PACTOS, VENCIMIENTOS)):
            Fk = S[c] * (1 + p)
            liq = COBERTURA * 4 * cuota * (S[m] - Fk)
            total += liq
            col = VE if S[m] > Fk else RO
            ax.hlines(Fk, c, m, color=col, lw=2.2, label="Forward pactado" if i == 0 else None)
            ax.plot([m, m], [Fk, S[m]], color=col, lw=1, ls=":")
            ax.annotate(_num(liq / 1e6, 1), (m, max(Fk, S[m])), textcoords="offset points", xytext=(-14, 6),
                        fontsize=8, color=col)
        ax.axvspan(24, 40, color="#f2f2f2", zorder=0)
        ax.set_title(f"(g) Escenario {nombre.lower()}: resultado de los forwards {_num(total / 1e6, 1)} millones", fontsize=10.5)
        ax.set_xlabel("Trimestre (zona gris: años cubiertos)"); ax.yaxis.set_major_formatter(fmt0)
        ax.set_xticks([20, 24, 28, 32, 36, 40])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        if k == 0:
            ax.set_ylabel("COP por USD")
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    for a in fig.axes:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Parte B: cobertura con cuatro forwards anuales (75 % de los pagos de los años 7 a 10)", fontsize=14, y=0.985)
    fig.savefig(sim.ruta("graficas_cobertura_forward.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Ejecuta toda la parte B y devuelve los resultados principales."""
    monto_usd, cuota, amort = sim.tabla_credito()
    desembolso = sim.VALOR_MAQUINARIA_COP * (1 - sim.APORTE)
    p = sim.estimar_parametros(sim.VENTANAS[sim.VENTANA_ELEGIDA])
    nu = sim.estimar_nu(sim.VENTANAS[sim.VENTANA_ELEGIDA])
    print(f"i_COP implícita {I_COP:.4%} | i_USD {I_USD:.2%} | prima forward {prima(I_COP, I_USD):.4%} | nu t {nu:.2f}")

    paths_n, _ = sim.simular(p["mu_q"], p["sig_q"])
    paths_t, _ = sim.simular(p["mu_q"], p["sig_q"], seed=sim.SEED + 1, dist="t", nu=nu)
    paths_t4, _ = sim.simular(p["mu_q"], p["sig_q"], seed=sim.SEED + 2, dist="t", nu=NU_STRESS)

    res_n, r_n = resumen(paths_n, cuota, "Normal", desembolso)
    res_t, r_t = resumen(paths_t, cuota, f"t de Student (nu = {nu:.1f})", desembolso)
    res_t4, r_t4 = resumen(paths_t4, cuota, f"t de Student de estrés (nu = {NU_STRESS})", desembolso)
    tabla = pd.concat([res_n, res_t, res_t4], axis=1)
    tabla.to_csv(sim.ruta("cobertura_resultados.csv"))
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 70,
                           "display.float_format", "{:,.4f}".format):
        t2 = tabla.copy(); t2.columns = ["Normal", "t", "t4"]
        print(t2.to_string())

    pf = pd.concat([por_forward(r_n).assign(dist="Normal"), por_forward(r_t).assign(dist="t")])
    pf.to_csv(sim.ruta("cobertura_por_forward.csv"), index=False)
    print(pf.round(3).to_string(index=False))

    # Trayectorias representativas: entre los caminos con TRM del año 6 cercana a la mediana
    # (percentiles 45-55), se toman los percentiles 10, 50 y 90 de la depreciación durante
    # el período cubierto (TRM trimestre 40 / TRM trimestre 24). Así los tres escenarios
    # parten del mismo nivel y difieren en lo que ocurre durante la cobertura.
    rep = {}
    for etiqueta, paths in [("Normal", paths_n), ("t", paths_t)]:
        s24 = paths[:, TRIM_INICIO]
        lo, hi = np.quantile(s24, [0.45, 0.55])
        cand = np.where((s24 >= lo) & (s24 <= hi))[0]
        dep = paths[cand, 40] / paths[cand, TRIM_INICIO]
        orden = cand[np.argsort(dep)]
        for nombre, q in [("Favorable", 0.10), ("Central", 0.50), ("Adverso", 0.90)]:
            j = orden[int(q * (len(orden) - 1))]
            rep[f"{nombre} – {etiqueta}"] = paths[j]
    pd.DataFrame(rep).to_csv(sim.ruta("trayectorias_representativas.csv"), index_label="trimestre")

    # Sensibilidad a tasas (innovaciones normales, mismas trayectorias)
    filas = []
    for i_cop in [0.06, 0.08, 0.10, 0.1225, I_COP, 0.14]:
        for i_usd in [0.03, I_USD, 0.06]:
            r = evaluar(paths_n, cuota, i_cop, i_usd)
            rs = evaluar(paths_n, cuota, i_cop, i_usd, variante="strip")
            liq = r["liq"].sum(axis=1)
            filas.append({
                "i_COP": i_cop, "i_USD": i_usd, "prima_forward": prima(i_cop, i_usd),
                "prob_proteccion": (r["Sv"] > r["F"]).mean(),
                "resultado_forwards_media": liq.mean(),
                "costo_total_con_media": r["pagos_con"].sum(axis=1).mean(),
                "sd_costo_710_con": r["pagos_con"][:, 24:].sum(axis=1).std(),
                "prob_cobertura_cuesta_mas": (liq < 0).mean(),
                "strip_resultado_media": rs["liq"].sum(axis=1).mean(),
                "fracaso": (r["H"] > r["B"] * (1 + TOLERANCIA_PPTO)).mean(),
            })
    sens = pd.DataFrame(filas)
    sens.to_csv(sim.ruta("sensibilidad_tasas.csv"), index=False)
    print(sens.round(4).to_string(index=False))

    # Distribuciones para la figura de la parte B
    np.savez_compressed(sim.ruta("cobertura_distribuciones.npz"),
                        sin_n=r_n["pagos_sin"][:, 24:].sum(axis=1), con_n=r_n["pagos_con"][:, 24:].sum(axis=1),
                        sin_t=r_t["pagos_sin"][:, 24:].sum(axis=1), con_t=r_t["pagos_con"][:, 24:].sum(axis=1),
                        strip_n=evaluar(paths_n, cuota, variante="strip")["pagos_con"][:, 24:].sum(axis=1),
                        dev_sin=(r_n["U"] / r_n["B"] - 1).ravel(), dev_con=(r_n["H"] / r_n["B"] - 1).ravel())

    if MOSTRAR_GRAFICAS:
        graficas(paths_n, cuota, r_n, r_t, r_t4, evaluar(paths_n, cuota, variante="strip"), sens, rep, nu)
    return {"tabla": tabla, "por_forward": pf, "sensibilidad": sens, "representativas": rep,
            "nu": nu, "i_cop": I_COP}


if __name__ == "__main__":
    main()
