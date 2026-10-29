# -*- coding: utf-8 -*-
"""
Simulación Monte Carlo de la TRM (COP por USD) a 40 trimestres y conversión
de las cuotas del crédito en dólares a pesos (punto A.3).

Año 1 (trimestres 1-4): anclado a los tres escenarios del análisis fundamental.
  - Se sortea un escenario (base / peso fuerte / peso débil) con su probabilidad.
  - La TRM al trimestre 4 sigue una lognormal cuya mediana es el punto medio
    geométrico del rango del escenario y cuya dispersión reproduce ese rango
    como intervalo de 80 %.
  - La trayectoria intermedia es un puente browniano entre la TRM inicial y ese valor.

Años 2-10 (trimestres 5-40): paseo aleatorio geométrico cuya deriva y volatilidad
  YA NO SON SUPUESTOS FIJOS: se estiman con los rendimientos logarítmicos
  trimestrales de la TRM histórica (archivo trm_historica_trimestral.csv).
  La ventana histórica se elige en VENTANA_ELEGIDA; el script también corre
  todas las ventanas de VENTANAS para comparar sus resultados.
  Si el archivo trm_historica_trimestral.csv no está en la carpeta de trabajo,
  el código usa la misma serie, que va embebida más abajo (TRM_HISTORICA_CSV).

Uso: python3 simulacion_trm.py   (desde la carpeta raíz del repositorio)
Salidas (en la carpeta resultados/):
  trm_trayectorias.csv        trayectorias (spot, escenarios, percentiles simulados)
  trm_paths.npy               las N_SIM trayectorias simuladas (ventana elegida)
  parametros_por_ventana.csv  deriva, volatilidad y resultados por ventana histórica
  cuotas_cop_simulacion.csv   percentiles de la cuota en pesos por trimestre
  tabla_trm_40_trimestres.csv     TRM proyectada en cada trimestre (spot, escenarios, simulación)
  tabla_cuotas_cop_40_trimestres.csv  cuota de cada trimestre en USD y en pesos
  graficas_simulacion_trm.png     las gráficas del resultado
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- PARÁMETROS ---------------------------------
S0 = 3174.67            # TRM vigente 17-19 oct 2026 (Superfinanciera)
FECHA_INICIO = "2026-10-18"
N_Q = 40                # trimestres (10 años)
N_SIM = 20000
SEED = 42

# Escenarios al trimestre 4: (nombre, probabilidad, límite inferior, límite superior)
ESCENARIOS = [
    ("Base",        0.50, 3250.0, 3500.0),
    ("Peso fuerte", 0.25, 3000.0, 3250.0),
    ("Peso débil",  0.25, 3500.0, 3900.0),
]
Z80 = 1.2816            # el rango del escenario = intervalo de 80 % de la lognormal
VOL_ANIO1 = 0.10        # volatilidad anual del puente dentro del año 1 (ruido intra-año)
MOSTRAR_TABLAS = True   # imprime los valores proyectados de los 40 trimestres
MOSTRAR_GRAFICAS = True # dibuja las gráficas (en Colab/Jupyter aparecen debajo de la celda)

# Datos históricos: TRM de cierre de cada trimestre (marzo, junio, septiembre, diciembre)
ARCHIVO_HIST = os.path.join("data", "trm_historica_trimestral.csv")
CARPETA_RESULTADOS = "resultados"   # aquí se guardan tablas y gráficas
# Ventanas: los rendimientos usados son los posteriores a la fecha indicada
# y hasta el último dato (septiembre de 2026).
VENTANAS = {
    "3 años (2023-2026)":           "2023-09-30",
    "5 años (2021-2026)":           "2021-09-30",
    "10 años (2016-2026)":          "2016-09-30",
    "20 años (2006-2026)":          "2006-09-30",
    "Flotación libre (1999-2026)":  "1999-09-30",
    "Serie completa (1993-2026)":   "1993-03-31",
}
VENTANA_ELEGIDA = "Flotación libre (1999-2026)"

# Serie histórica embebida (TRM de cierre de cada trimestre). Permite correr el código
# sin el archivo CSV, por ejemplo en Google Colab. Fuentes: Superfinanciera (1993-2022),
# FRED/OCDE COLCCUSSP02STQ (2023), Pound Sterling Live y ValutaFX, cierres de mercado
# (2024-2026), y TRM del 31-dic-2024 (Superfinanciera).
TRM_HISTORICA_CSV = """fecha,trm
1993-03-31,767.10
1993-06-30,786.80
1993-09-30,810.11
1993-12-31,804.04
1994-03-31,821.21
1994-06-30,820.47
1994-09-30,840.90
1994-12-31,829.32
1995-03-31,873.67
1995-06-30,876.89
1995-09-30,977.54
1995-12-31,986.75
1996-03-31,1048.62
1996-06-30,1070.07
1996-09-30,1032.29
1996-12-31,1003.98
1997-03-31,1060.08
1997-06-30,1087.86
1997-09-30,1247.69
1997-12-31,1287.12
1998-03-31,1362.87
1998-06-30,1375.53
1998-09-30,1554.09
1998-12-31,1510.99
1999-03-31,1532.94
1999-06-30,1735.82
1999-09-30,2004.97
1999-12-31,1872.12
2000-03-31,1954.04
2000-06-30,2132.82
2000-09-30,2222.37
2000-12-31,2186.21
2001-03-31,2278.78
2001-06-30,2305.66
2001-09-30,2328.23
2001-12-31,2306.90
2002-03-31,2282.33
2002-06-30,2364.25
2002-09-30,2751.23
2002-12-31,2814.89
2003-03-31,2959.01
2003-06-30,2826.95
2003-09-30,2840.08
2003-12-31,2807.20
2004-03-31,2682.09
2004-06-30,2694.09
2004-09-30,2608.30
2004-12-31,2389.75
2005-03-31,2363.23
2005-06-30,2324.22
2005-09-30,2288.22
2005-12-31,2284.22
2006-03-31,2293.38
2006-06-30,2579.08
2006-09-30,2394.31
2006-12-31,2238.79
2007-03-31,2190.30
2007-06-30,1960.61
2007-09-30,2023.19
2007-12-31,2014.76
2008-03-31,1834.96
2008-06-30,1923.02
2008-09-30,2184.76
2008-12-31,2243.59
2009-03-31,2544.24
2009-06-30,2145.21
2009-09-30,1925.49
2009-12-31,2044.23
2010-03-31,1921.88
2010-06-30,1913.15
2010-09-30,1801.01
2010-12-31,1913.98
2011-03-31,1870.60
2011-06-30,1772.32
2011-09-30,1929.01
2011-12-31,1942.70
2012-03-31,1792.07
2012-06-30,1784.60
2012-09-30,1800.52
2012-12-31,1768.23
2013-03-31,1832.20
2013-06-30,1929.00
2013-09-30,1908.29
2013-12-31,1926.83
2014-03-31,1969.45
2014-06-30,1881.19
2014-09-30,2022.00
2014-12-31,2392.46
2015-03-31,2598.36
2015-06-30,2598.68
2015-09-30,3086.75
2015-12-31,3149.47
2016-03-31,3000.63
2016-06-30,2919.01
2016-09-30,2880.08
2016-12-31,3000.71
2017-03-31,2885.57
2017-06-30,3050.43
2017-09-30,2936.67
2017-12-31,2984.00
2018-03-31,2780.47
2018-06-30,2930.80
2018-09-30,2972.18
2018-12-31,3249.75
2019-03-31,3174.79
2019-06-30,3205.67
2019-09-30,3477.45
2019-12-31,3277.14
2020-03-31,4054.54
2020-06-30,3756.28
2020-09-30,3865.47
2020-12-31,3432.50
2021-03-31,3678.62
2021-06-30,3748.50
2021-09-30,3812.77
2021-12-31,3981.16
2022-03-31,3756.03
2022-06-30,4151.21
2022-09-30,4590.54
2022-12-31,4810.20
2023-03-31,4627.27
2023-06-30,4191.28
2023-09-30,4053.76
2023-12-31,3822.05
2024-03-31,3858.50
2024-06-30,4149.30
2024-09-30,4190.90
2024-12-31,4409.15
2025-03-31,4185.60
2025-06-30,4057.20
2025-09-30,3925.20
2025-12-31,3769.50
2026-03-31,3642.80
2026-06-30,3446.70
2026-09-30,3346.70
"""


def ruta(nombre):
    """Ruta de salida dentro de la carpeta de resultados (la crea si no existe)."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    return os.path.join(CARPETA_RESULTADOS, nombre)


def cargar_historico(archivo=ARCHIVO_HIST):
    """Lee el CSV si está en la carpeta de trabajo; si no, usa la serie embebida."""
    import io
    if archivo and os.path.exists(archivo):
        h = pd.read_csv(archivo, parse_dates=["fecha"])
    else:
        h = pd.read_csv(io.StringIO(TRM_HISTORICA_CSV), parse_dates=["fecha"])
    return h.sort_values("fecha")

# Crédito (punto A.2): SBA 7(a) a tasa fija, sistema francés trimestral
VALOR_MAQUINARIA_COP = 400_000_000
APORTE = 0.10
TASA_NOMINAL = 0.13     # Prime 7,00 % + 6,0 pp (tope SBA 7(a) USD 50.001-250.000)
PAGOS_ANIO = 4
# --------------------------------------------------------------------------


def estimar_parametros(fecha_desde, archivo=ARCHIVO_HIST):
    """Deriva y volatilidad trimestrales (en logaritmos) desde los datos históricos."""
    h = cargar_historico(archivo)
    r = np.log(h["trm"]).diff()
    r.index = h["fecha"]
    r = r.dropna()
    r = r[r.index > pd.Timestamp(fecha_desde)]
    mu_q = r.mean()
    sig_q = r.std(ddof=1)
    return {
        "n_trimestres": len(r),
        "desde": r.index.min().date(), "hasta": r.index.max().date(),
        "mu_q": mu_q, "sig_q": sig_q,
        "deriva_anual": 4 * mu_q,
        "volatilidad_anual": 2 * sig_q,
        "error_est_deriva_anual": 2 * sig_q / np.sqrt(len(r) / 4),
    }


def mediana_y_sigma(lo, hi):
    """Mediana geométrica y sigma log tal que [lo, hi] sea el intervalo de 80 %."""
    med = np.sqrt(lo * hi)
    sig = np.log(hi / lo) / (2 * Z80)
    return med, sig


def trayectoria_escenario_mediana(med, mu_q):
    """Camino mediano: interpolación log-lineal en el año 1; luego deriva histórica."""
    t = np.arange(N_Q + 1)
    logp = np.empty(N_Q + 1)
    logp[:5] = np.log(S0) + (np.log(med) - np.log(S0)) * t[:5] / 4
    for k in range(5, N_Q + 1):
        logp[k] = logp[k - 1] + mu_q
    return np.exp(logp)


def estimar_nu(fecha_desde, archivo=ARCHIVO_HIST):
    """Grados de libertad de una t de Student ajustada por máxima verosimilitud
    a los rendimientos log trimestrales de la ventana histórica."""
    from scipy import stats
    h = cargar_historico(archivo)
    r = np.log(h["trm"]).diff()
    r.index = h["fecha"]
    r = r.dropna()
    x = r[r.index > pd.Timestamp(fecha_desde)].values
    nu, loc, esc = stats.t.fit(x)
    return nu


def simular(mu_q, sig_q, seed=SEED, dist="normal", nu=None):
    """Trayectorias de la TRM.
    dist="normal": innovaciones normales (versión original, mismos sorteos).
    dist="t": innovaciones t de Student con nu grados de libertad, reescaladas para
    tener la misma varianza que la normal; en el año 1 se conserva el rango de cada
    escenario como intervalo de 80 % (usando el cuantil de la t), así que solo
    cambian las colas."""
    rng = np.random.default_rng(seed)
    probs = np.array([e[1] for e in ESCENARIOS])
    idx = rng.choice(len(ESCENARIOS), size=N_SIM, p=probs)
    meds = np.array([mediana_y_sigma(e[2], e[3])[0] for e in ESCENARIOS])
    sigs = np.array([mediana_y_sigma(e[2], e[3])[1] for e in ESCENARIOS])

    if dist == "normal":
        choque = lambda size: rng.standard_normal(size)
        choque_fin1 = lambda size: rng.standard_normal(size)
    elif dist == "t":
        from scipy import stats
        escala = np.sqrt((nu - 2) / nu)                 # varianza unitaria
        choque = lambda size: rng.standard_t(nu, size) * escala
        z80_t = stats.t.ppf(0.90, nu)
        sigs = sigs * Z80 / z80_t                        # mismo intervalo de 80 %
        choque_fin1 = lambda size: rng.standard_t(nu, size)
    else:
        raise ValueError("dist debe ser 'normal' o 't'")

    logS = np.empty((N_SIM, N_Q + 1))
    logS[:, 0] = np.log(S0)
    # Fin del año 1
    end1 = np.log(meds[idx]) + sigs[idx] * choque_fin1(N_SIM)
    # Puente browniano en los trimestres 1..3
    sq1 = VOL_ANIO1 / 2.0
    W = np.cumsum(sq1 * choque((N_SIM, 4)), axis=1)
    t = np.arange(1, 5)
    bridge = W - (t / 4.0) * W[:, [3]]
    for k in range(1, 5):
        logS[:, k] = logS[:, 0] + (end1 - logS[:, 0]) * k / 4 + bridge[:, k - 1]
    # Años 2-10: parámetros históricos
    for k in range(5, N_Q + 1):
        logS[:, k] = logS[:, k - 1] + mu_q + sig_q * choque(N_SIM)
    return np.exp(logS), idx


def tabla_credito():
    monto_usd = VALOR_MAQUINARIA_COP * (1 - APORTE) / S0
    i = TASA_NOMINAL / PAGOS_ANIO
    n = N_Q
    cuota = monto_usd * i / (1 - (1 + i) ** -n)
    saldo = monto_usd
    filas = []
    for k in range(1, n + 1):
        interes = saldo * i
        capital = cuota - interes
        saldo -= capital
        filas.append((k, interes, capital, saldo))
    df = pd.DataFrame(filas, columns=["trimestre", "interes_usd", "capital_usd", "saldo_usd"])
    return monto_usd, cuota, df


def tir_anual(flujos):
    """TIR trimestral por bisección (flujo inicial positivo = desembolso recibido),
    expresada como tasa efectiva anual."""
    lo, hi = -0.5, 1.0
    t = np.arange(len(flujos))
    for _ in range(200):
        mid = (lo + hi) / 2
        vpn = np.sum(flujos / (1 + mid) ** t)
        if vpn > 0:      # la tasa es demasiado alta
            hi = mid
        else:
            lo = mid
    return (1 + mid) ** 4 - 1


def costos_cop(paths, cuota, amort, monto_usd):
    """Costo en pesos por trayectoria: total pagado, efecto cambiario y costo efectivo anual."""
    S = paths[:, 1:]
    total = cuota * S.sum(axis=1)
    base_spot = cuota * N_Q * S0
    efecto_cambiario = total - base_spot
    efecto_interes = (amort["interes_usd"].values * (S - S0)).sum(axis=1)
    efecto_capital = (amort["capital_usd"].values * (S - S0)).sum(axis=1)
    return total, efecto_cambiario, efecto_interes, efecto_capital


def _num(x, dec=2):
    """Formato colombiano: 3.174,67"""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _mostrar(tabla, titulo):
    """Muestra una tabla: en Colab/Jupyter como tabla HTML; en consola como texto."""
    print("\n" + titulo)
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(tabla)
            return
    except ImportError:
        pass
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(tabla.to_string(index=False))


def tablas_40_trimestres(df, paths, cuota, amort):
    """Valores proyectados de los 40 trimestres: TRM y cuota en pesos."""
    # --- TRM ---
    t_trm = df.copy()
    t_trm.to_csv(ruta("tabla_trm_40_trimestres.csv"), index=False)
    cols = {"trimestre": "Trim.", "fecha": "Fecha", "spot_constante": "TRM spot",
            "escenario_base": "Base", "escenario_peso_fuerte": "Peso fuerte",
            "escenario_peso_debil": "Peso débil", "sim_media": "Sim. media", "sim_p05": "Sim. P5",
            "sim_p25": "Sim. P25", "sim_p50": "Sim. P50", "sim_p75": "Sim. P75", "sim_p95": "Sim. P95"}
    f_trm = t_trm.rename(columns=cols)
    for c in list(cols.values())[2:]:
        f_trm[c] = f_trm[c].map(_num)
    _mostrar(f_trm, "TRM proyectada por trimestre (COP por USD)")

    # --- cuotas en pesos ---
    S = paths[:, 1:]
    cuotas = cuota * S
    pc = np.percentile(cuotas, [5, 50, 95], axis=0)
    t_c = pd.DataFrame({
        "trimestre": np.arange(1, N_Q + 1),
        "fecha": df["fecha"].values[1:],
        "cuota_usd": cuota,
        "interes_usd": amort["interes_usd"].values,
        "capital_usd": amort["capital_usd"].values,
        "cuota_cop_spot": cuota * S0,
        "cuota_cop_base": cuota * df["escenario_base"].values[1:],
        "cuota_cop_peso_fuerte": cuota * df["escenario_peso_fuerte"].values[1:],
        "cuota_cop_peso_debil": cuota * df["escenario_peso_debil"].values[1:],
        "cuota_cop_sim_media": cuotas.mean(axis=0),
        "cuota_cop_sim_p05": pc[0], "cuota_cop_sim_p50": pc[1], "cuota_cop_sim_p95": pc[2],
    })
    t_c.to_csv(ruta("tabla_cuotas_cop_40_trimestres.csv"), index=False)
    f_c = pd.DataFrame({
        "Trim.": t_c.trimestre, "Fecha": t_c.fecha,
        "Cuota USD": t_c.cuota_usd.map(_num), "Interés USD": t_c.interes_usd.map(_num),
        "Capital USD": t_c.capital_usd.map(_num),
    })
    for c, nombre in [("cuota_cop_spot", "Spot"), ("cuota_cop_base", "Base"), ("cuota_cop_peso_fuerte", "Peso fuerte"),
                      ("cuota_cop_peso_debil", "Peso débil"), ("cuota_cop_sim_media", "Sim. media"),
                      ("cuota_cop_sim_p05", "Sim. P5"), ("cuota_cop_sim_p50", "Sim. P50"), ("cuota_cop_sim_p95", "Sim. P95")]:
        f_c[nombre] = (t_c[c] / 1e6).map(_num)
    total = {"Trim.": "Total", "Fecha": "", "Cuota USD": _num(t_c.cuota_usd.sum()),
             "Interés USD": _num(t_c.interes_usd.sum()), "Capital USD": _num(t_c.capital_usd.sum())}
    for c, nombre in [("cuota_cop_spot", "Spot"), ("cuota_cop_base", "Base"), ("cuota_cop_peso_fuerte", "Peso fuerte"),
                      ("cuota_cop_peso_debil", "Peso débil"), ("cuota_cop_sim_media", "Sim. media")]:
        total[nombre] = _num(t_c[c].sum() / 1e6)
    for nombre in ["Sim. P5", "Sim. P50", "Sim. P95"]:
        total[nombre] = "–"          # los percentiles por trimestre no se suman
    f_c = pd.concat([f_c.astype(str), pd.DataFrame([total])], ignore_index=True)
    _mostrar(f_c, "Cuota de cada trimestre: en USD y en millones de pesos")
    return t_c


def graficas(df, paths, cuota, amort, paths_por_ventana, total):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: _num(v, 0))
    fmt1 = FuncFormatter(lambda v, _: _num(v, 1))
    x = df["trimestre"].values
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    # 1) TRM
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(7)
    for j in rng.choice(paths.shape[0], 25, replace=False):
        ax.plot(x, paths[j], color="#9db9d8", lw=0.5, alpha=0.6)
    ax.fill_between(x, df.sim_p05, df.sim_p95, color="#9db9d8", alpha=0.35, lw=0, label="Simulación P5–P95")
    ax.fill_between(x, df.sim_p25, df.sim_p75, color="#5b8cc0", alpha=0.45, lw=0, label="Simulación P25–P75")
    ax.plot(x, df.sim_p50, color="#1f3d63", lw=2, label="Simulación mediana")
    ax.plot(x, df.sim_media, color="#1f3d63", lw=1.2, ls=":", label="Simulación media")
    ax.plot(x, df.spot_constante, color="#555", ls="--", lw=1.2, label="TRM spot constante")
    ax.plot(x, df.escenario_peso_fuerte, color="#2e8b57", lw=1.3, label="Escenario peso fuerte")
    ax.plot(x, df.escenario_base, color="#e67e22", lw=1.3, label="Escenario base")
    ax.plot(x, df.escenario_peso_debil, color="#c0392b", lw=1.3, label="Escenario peso débil")
    ax.set_title("TRM proyectada (COP por USD)"); ax.set_xlabel("Trimestre"); ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # 2) Cuota en pesos
    ax = fig.add_subplot(gs[0, 1])
    k = x[1:]
    c = cuota * paths[:, 1:] / 1e6
    p5, p25, p50, p75, p95 = np.percentile(c, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(k, p5, p95, color="#9db9d8", alpha=0.35, lw=0, label="Simulación P5–P95")
    ax.fill_between(k, p25, p75, color="#5b8cc0", alpha=0.45, lw=0, label="Simulación P25–P75")
    ax.plot(k, p50, color="#1f3d63", lw=2, label="Simulación mediana")
    ax.plot(k, cuota * df.spot_constante.values[1:] / 1e6, color="#555", ls="--", lw=1.2, label="Con TRM spot")
    ax.plot(k, cuota * df.escenario_base.values[1:] / 1e6, color="#e67e22", lw=1.3, label="Escenario base")
    ax.set_title(f"Cuota trimestral en pesos (USD {_num(cuota)} constantes)"); ax.set_xlabel("Trimestre")
    ax.set_ylabel("Millones de COP"); ax.yaxis.set_major_formatter(fmt1); ax.legend(fontsize=8, frameon=False, loc="upper left")

    # 3) Ventanas históricas
    ax = fig.add_subplot(gs[1, 0])
    colores = ["#c0392b", "#e67e22", "#8e44ad", "#16a085", "#1f3d63", "#7f8c8d"]
    for (nombre, (pth, _, prm)), col in zip(paths_por_ventana.items(), colores):
        lw = 2.4 if nombre == VENTANA_ELEGIDA else 1.2
        ax.plot(x, np.percentile(pth, 50, axis=0), color=col, lw=lw,
                label=f"{nombre}: deriva {_num(prm['deriva_anual']*100, 1)} %, vol. {_num(prm['volatilidad_anual']*100, 1)} %")
    ax.axhline(S0, color="#555", ls="--", lw=1)
    ax.set_title("TRM mediana según la ventana histórica usada"); ax.set_xlabel("Trimestre")
    ax.yaxis.set_major_formatter(fmt); ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # 4) Histograma del costo total
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(total / 1e6, bins=80, color="#5b8cc0", alpha=0.8)
    spot_total = cuota * N_Q * S0 / 1e6
    for v, lab, col, ls in [(spot_total, "TRM spot", "#555", "--"), (total.mean() / 1e6, "Media", "#1f3d63", "-"),
                            (np.percentile(total, 5) / 1e6, "P5", "#2e8b57", ":"), (np.percentile(total, 95) / 1e6, "P95", "#c0392b", ":")]:
        ax.axvline(v, color=col, ls=ls, lw=1.5, label=f"{lab}: {_num(v, 1)}")
    ax.set_title("Costo total del crédito en pesos (20.000 simulaciones)"); ax.set_xlabel("Millones de COP")
    ax.set_yticks([]); ax.xaxis.set_major_formatter(fmt); ax.legend(fontsize=8, frameon=False)

    # 5) Intereses vs tasa de cambio (escenario base)
    ax = fig.add_subplot(gs[2, :])
    Sb = df.escenario_base.values[1:]
    i_spot = amort["interes_usd"].values * S0 / 1e6
    c_spot = amort["capital_usd"].values * S0 / 1e6
    ef_tc = cuota * (Sb - S0) / 1e6
    ax.bar(k, c_spot, color="#5b8cc0", label="Capital valorado a TRM spot")
    ax.bar(k, i_spot, bottom=c_spot, color="#e67e22", label="Intereses valorados a TRM spot")
    ax.bar(k, ef_tc, bottom=c_spot + i_spot, color="#c0392b", label="Efecto de la tasa de cambio")
    ax.set_title("Escenario base: cuota en pesos separada entre capital, intereses y efecto cambiario")
    ax.set_xlabel("Trimestre"); ax.set_ylabel("Millones de COP"); ax.yaxis.set_major_formatter(fmt1)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)

    for a in fig.axes:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.savefig(ruta("graficas_simulacion_trm.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Ejecuta toda la parte A y devuelve los resultados principales."""
    monto_usd, cuota, amort = tabla_credito()
    print(f"Monto USD {monto_usd:,.2f} | cuota trimestral USD {cuota:,.2f}\n")

    # 1) Parámetros y resultados para cada ventana histórica
    filas = []
    paths_por_ventana = {}
    for nombre, desde in VENTANAS.items():
        p = estimar_parametros(desde)
        paths, idx = simular(p["mu_q"], p["sig_q"])
        paths_por_ventana[nombre] = (paths, idx, p)
        total, ef_tc, _, _ = costos_cop(paths, cuota, amort, monto_usd)
        filas.append({
            "ventana": nombre, "n_trimestres": p["n_trimestres"],
            "deriva_anual": p["deriva_anual"], "volatilidad_anual": p["volatilidad_anual"],
            "error_est_deriva": p["error_est_deriva_anual"],
            "TRM_p50_anio5": np.percentile(paths[:, 20], 50),
            "TRM_p05_anio10": np.percentile(paths[:, 40], 5),
            "TRM_p50_anio10": np.percentile(paths[:, 40], 50),
            "TRM_p95_anio10": np.percentile(paths[:, 40], 95),
            "costo_total_cop_media": total.mean(),
            "costo_total_cop_p95": np.percentile(total, 95),
        })
    tab = pd.DataFrame(filas)
    tab.to_csv(ruta("parametros_por_ventana.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
        print(tab.to_string(index=False))

    # 2) Ventana elegida: trayectorias, escenarios y cuotas en pesos
    paths, idx, p = paths_por_ventana[VENTANA_ELEGIDA]
    mu_q = p["mu_q"]
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fechas = pd.date_range(FECHA_INICIO, periods=N_Q + 1, freq=pd.DateOffset(months=3))
    df = pd.DataFrame({
        "trimestre": np.arange(N_Q + 1),
        "fecha": fechas.date,
        "spot_constante": S0,
        "escenario_base": trayectoria_escenario_mediana(mediana_y_sigma(*ESCENARIOS[0][2:])[0], mu_q),
        "escenario_peso_fuerte": trayectoria_escenario_mediana(mediana_y_sigma(*ESCENARIOS[1][2:])[0], mu_q),
        "escenario_peso_debil": trayectoria_escenario_mediana(mediana_y_sigma(*ESCENARIOS[2][2:])[0], mu_q),
        "sim_media": paths.mean(axis=0),
        "sim_p05": pct[0], "sim_p25": pct[1], "sim_p50": pct[2], "sim_p75": pct[3], "sim_p95": pct[4],
    })
    df.to_csv(ruta("trm_trayectorias.csv"), index=False)
    np.save(ruta("trm_paths.npy"), paths.astype("float32"))

    cuotas_cop = cuota * paths[:, 1:]
    pc = np.percentile(cuotas_cop, [5, 50, 95], axis=0)
    pd.DataFrame({
        "trimestre": np.arange(1, N_Q + 1),
        "cuota_cop_spot": cuota * S0,
        "cuota_cop_media": cuotas_cop.mean(axis=0),
        "cuota_cop_p05": pc[0], "cuota_cop_p50": pc[1], "cuota_cop_p95": pc[2],
    }).to_csv(ruta("cuotas_cop_simulacion.csv"), index=False)

    total, ef_tc, ef_int, ef_cap = costos_cop(paths, cuota, amort, monto_usd)
    monto_cop = VALOR_MAQUINARIA_COP * (1 - APORTE)
    tir = np.array([tir_anual(np.concatenate([[monto_cop], -cuota * paths[j, 1:]]))
                    for j in range(0, N_SIM, 20)])   # muestra de 1.000 trayectorias
    print(f"\nVentana elegida: {VENTANA_ELEGIDA} | deriva {p['deriva_anual']:.4f} | vol {p['volatilidad_anual']:.4f}")
    print(df.iloc[[0, 1, 2, 4, 8, 12, 20, 40]].round(1).to_string(index=False))
    print("\nProporción de sorteos por escenario:",
          {ESCENARIOS[i][0]: round(float((idx == i).mean()), 3) for i in range(3)})
    print("Costo total en COP: media %.0f | P5 %.0f | P50 %.0f | P95 %.0f" % (
        total.mean(), *np.percentile(total, [5, 50, 95])))
    print("Efecto cambiario: media %.0f (sobre intereses %.0f, sobre capital %.0f)" % (
        ef_tc.mean(), ef_int.mean(), ef_cap.mean()))
    print("Probabilidad de pagar más que con la TRM spot: %.3f" % (ef_tc > 0).mean())
    print("Costo efectivo anual en COP (muestra): media %.4f | P5 %.4f | P50 %.4f | P95 %.4f" % (
        tir.mean(), *np.percentile(tir, [5, 50, 95])))

    # 3) Valores proyectados de los 40 trimestres y gráficas
    if MOSTRAR_TABLAS:
        tablas_40_trimestres(df, paths, cuota, amort)
    if MOSTRAR_GRAFICAS:
        graficas(df, paths, cuota, amort, paths_por_ventana, total)
    return {"monto_usd": monto_usd, "cuota": cuota, "amort": amort, "ventanas": tab,
            "trayectorias": df, "paths": paths, "total": total, "tir": tir}


if __name__ == "__main__":
    main()
