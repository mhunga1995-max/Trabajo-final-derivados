# Financiación de maquinaria amarilla con un crédito en dólares y cobertura con forwards

Trabajo final de Derivados · Universidad EAFIT · Octubre de 2026

Una empresa colombiana compra maquinaria amarilla por COP 400 millones: aporta el 10 % y financia el 90 % con un crédito SBA 7(a) de U.S. Bank en dólares (13,00 % nominal anual fijo, 10 años, cuotas trimestrales por sistema francés). El trabajo proyecta la TRM, convierte las cuotas a pesos con una simulación Monte Carlo de 20.000 trayectorias y evalúa una cobertura con cuatro forwards anuales sobre el 75 % de los pagos de los años 7 a 10.

## 📄 Informe

**➡️ [Abrir el informe completo (`informe.ipynb`)](informe.ipynb)**

El cuaderno contiene todo el texto del análisis junto al código en Python que produce cada tabla y gráfica. GitHub lo muestra ya ejecutado, con los resultados visibles.

## Resultados principales

| Indicador | Resultado |
|---|---|
| Cuota trimestral del crédito | USD 5.106,06 (monto USD 113.397,61 a TRM 3.174,67) |
| Costo del crédito en dólares | 13,65 % efectivo anual |
| Costo total en pesos con TRM spot | COP 648,4 millones |
| Costo total en pesos, simulación (media) | COP 778,3 millones; P5–P95: 529,2 a 1.115,9 |
| Costo efectivo esperado en pesos | ≈ 17,4 % anual |
| Probabilidad de pagar más que con la TRM spot | 74,8 % |
| Forwards: proporción de años con protección | 32,5 % (costo de oportunidad en el 67,5 %) |
| Costo esperado de la cobertura | COP 13,3 millones (1,7 % del costo total) |
| Años con costo > presupuesto + 10 % | 3,81 % sin cobertura → 0,07 % con cobertura |

**Conclusión:** la cobertura hace casi predecible el costo de cada año y reduce entre 85 % y 90 % las pérdidas en los años de devaluación fuerte, pero con el diferencial de tasas actual tiene un costo esperado positivo. Es conveniente si la empresa valora más la estabilidad presupuestal que el ahorro esperado.

![Simulación de la TRM y cuota en pesos](resultados/graficas_simulacion_trm.png)

![Evaluación de la cobertura forward](resultados/graficas_cobertura_forward.png)

## Estructura del repositorio

```
├── informe.ipynb                  ← informe completo (texto + código + resultados)
├── simulacion_trm.py              ← parte A: amortización, simulación de la TRM, conversión a pesos
├── cobertura_forward.py           ← parte B: cobertura con cuatro forwards (NDF)
├── requirements.txt
├── data/
│   └── trm_historica_trimestral.csv   ← TRM de cierre trimestral 1993–2026
├── resultados/                    ← tablas (CSV) y gráficas (PNG) que generan los scripts
└── documentos/
    ├── Informe_derivados_APA.docx           ← versión en Word del informe (normas APA)
    └── amortizacion_credito_maquinaria.xlsx ← amortización, conversión a pesos y cobertura en Excel
```

## Cómo reproducir los resultados

Requiere Python 3.10 o superior.

```bash
git clone <url-de-este-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt

python simulacion_trm.py       # parte A  (≈ 2 segundos)
python cobertura_forward.py    # parte B  (≈ 5 segundos)
```

Los resultados quedan en `resultados/`. Para volver a ejecutar el informe: `jupyter notebook informe.ipynb` → *Run All*. La semilla es fija (`SEED = 42`), así que las cifras coinciden con las del informe.

### Supuestos que se pueden cambiar

| Parámetro | Archivo | Valor |
|---|---|---|
| TRM inicial `S0` | `simulacion_trm.py` | 3.174,67 |
| Escenarios a 12 meses `ESCENARIOS` | `simulacion_trm.py` | base 50 %, peso fuerte 25 %, peso débil 25 % |
| Ventana histórica `VENTANA_ELEGIDA` | `simulacion_trm.py` | Flotación libre (1999–2026) |
| Tasa del crédito `TASA_NOMINAL` | `simulacion_trm.py` | 13,00 % |
| Porcentaje cubierto `COBERTURA` | `cobertura_forward.py` | 75 % |
| Forward de mercado `FWD_1Y_MERCADO` e `I_USD` | `cobertura_forward.py` | 3.386,30 y 4,47 % |

## Fuentes de datos

TRM histórica: Superintendencia Financiera (1993–2022), FRED/OCDE `COLCCUSSP02STQ` (2023), Pound Sterling Live y ValutaFX (2024–2026). Curva forward USD/COP: Cbonds (27/10/2026). Tasas: Banco de la República y Reserva Federal (H.15). Prime y crédito: U.S. Bank y SBA. La lista completa de referencias (APA) está al final del informe.
