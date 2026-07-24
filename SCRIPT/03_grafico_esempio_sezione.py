# -*- coding: utf-8 -*-
"""
Step 3 del progetto parallelo (vector tiles) - grafico di anteprima.

Per una sezione di esempio (SEZ_ESEMPIO, da impostare qui sotto), genera un
grafico a due pannelli a partire da traffico_tile_serie_storica_milano.csv:

  1. Mappa dei segmenti stradali monitorati nella sezione, colorati per
     congestione massima osservata. Per evitare che una singola lettura
     isolata (rumore) distorca la lettura, il colore e' calcolato solo sui
     segmenti "robusti" (almeno MIN_LETTURE osservazioni): i segmenti sotto
     soglia restano in grigio neutro, a segnalare "dato insufficiente" senza
     suggerire un valore affidabile.
  2. Pattern orario del segmento piu' critico (il robusto con congestione
     massima piu' alta): barra = massimo nell'ora, linea = media nell'ora,
     etichetta = numero di letture a supporto di ciascun punto.

E' pensato come anteprima esplorativa (non analisi definitiva): utile per
farsi un'idea dell'output finale mentre la raccolta dati e' ancora in corso.

Output: preview_esempio_sezione_<comune>.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_CSV = CARTELLA_SCRIPT / "traffico_tile_serie_storica_milano.csv"
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"

SEZ_ESEMPIO = 151460002842  # cambiare qui per generare il grafico di un'altra sezione
MIN_LETTURE = 5  # soglia minima di osservazioni per considerare un segmento "robusto"

INK = "#2b2b2b"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"
ACCENT = "#c0392b"


def main():
    df = pd.read_csv(IN_CSV)
    sezioni = gpd.read_file(IN_GEOJSON)

    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)

    sez = df[df["SEZ2011"] == SEZ_ESEMPIO].copy()
    if sez.empty:
        raise ValueError(f"Nessun dato per SEZ2011={SEZ_ESEMPIO} in {IN_CSV}")
    poligono = sezioni[sezioni["SEZ2011"].astype(int) == SEZ_ESEMPIO].iloc[0]
    comune = poligono["COMUNE"]

    per_segmento = sez.groupby(["lat_r", "lon_r", "road_type"]).agg(
        congestione_max=("congestione", "max"), n_letture=("congestione", "size")
    ).reset_index()

    robusti = per_segmento[per_segmento["n_letture"] >= MIN_LETTURE].sort_values(
        "congestione_max", ascending=False)
    non_robusti = per_segmento[per_segmento["n_letture"] < MIN_LETTURE]
    if robusti.empty:
        raise ValueError(
            f"Nessun segmento con almeno {MIN_LETTURE} letture per SEZ2011={SEZ_ESEMPIO}: "
            "serve piu' tempo di raccolta prima di poter generare questo grafico.")
    top = robusti.iloc[0]

    letture_top = sez[(sez["lat_r"] == top["lat_r"]) & (sez["lon_r"] == top["lon_r"])].copy()
    letture_top["ora_locale"] = (letture_top["timestamp_utc"].str[11:13].astype(int) + 2) % 24
    pattern_orario = letture_top.groupby("ora_locale")["congestione"].agg(
        ["mean", "max", "count"]).reset_index()

    # ------------------------------------------------------------ palette
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=0, vmax=robusti["congestione_max"].max())

    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=130,
                                    gridspec_kw={"wspace": 0.32})

    # ------------------------------------------------------------ pannello 1: mappa
    poligono_wgs = gpd.GeoSeries([poligono.geometry], crs="EPSG:4326")
    poligono_buff_wgs = gpd.GeoSeries([poligono.geometry], crs="EPSG:4326").to_crs("EPSG:32632") \
        .buffer(50).to_crs("EPSG:4326")
    poligono_wgs.plot(ax=ax1, facecolor="none", edgecolor=MUTED, linewidth=1.1, zorder=1)
    poligono_buff_wgs.plot(ax=ax1, facecolor="none", edgecolor=MUTED, linewidth=1.0,
                            linestyle=(0, (4, 3)), zorder=1)

    # segmenti non robusti (< MIN_LETTURE): puntini piccoli, grigio neutro, per
    # segnalare "dato insufficiente" senza suggerire un valore di congestione
    # affidabile (colorarli sfalserebbe la lettura con possibili osservazioni
    # isolate/rumorose)
    ax1.scatter(non_robusti["lon_r"], non_robusti["lat_r"], s=14, facecolor="#d9d9d9",
                edgecolor="none", zorder=2, label=f"< {MIN_LETTURE} letture (dato insufficiente)")

    # segmenti robusti: colore = congestione massima, dimensione = n. letture
    sizes = 18 + robusti["n_letture"].clip(upper=60)
    ax1.scatter(robusti["lon_r"], robusti["lat_r"], c=robusti["congestione_max"],
                cmap=cmap, norm=norm, s=sizes, edgecolor="white", linewidth=0.4, zorder=3)

    ax1.scatter([top["lon_r"]], [top["lat_r"]], s=220, facecolor="none",
                edgecolor=ACCENT, linewidth=2, zorder=4)
    ax1.annotate(f"segmento piu' critico\n(robusto, {int(top['n_letture'])} letture)",
                 (top["lon_r"], top["lat_r"]), xytext=(-14, -30), textcoords="offset points",
                 ha="right", fontsize=8.5, color=ACCENT,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax1.set_title(f"Congestione massima per segmento — {comune}",
                  fontsize=11, color=INK, loc="left", pad=10)
    ax1.set_xlabel("longitudine", fontsize=8.5, color=MUTED)
    ax1.set_ylabel("latitudine", fontsize=8.5, color=MUTED)
    ax1.tick_params(labelsize=7.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.85, fontsize=7.5, handletextpad=0.3)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.045, pad=0.03)
    cbar.set_label(f"congestione massima\n(min. {MIN_LETTURE} letture)", fontsize=7.5, color=MUTED)
    cbar.ax.tick_params(labelsize=7.5, color=MUTED)

    # ------------------------------------------------------------ pannello 2: pattern orario
    ax2.bar(pattern_orario["ora_locale"], pattern_orario["max"], color="#fde3cf",
            width=0.7, label="massimo nell'ora", zorder=1)
    ax2.plot(pattern_orario["ora_locale"], pattern_orario["mean"], color=ACCENT,
             marker="o", markersize=5, linewidth=2, label="media nell'ora", zorder=2)

    for _, r in pattern_orario.iterrows():
        ax2.annotate(str(int(r["count"])), (r["ora_locale"], r["max"]), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)

    ax2.set_title(f"Pattern orario del segmento più critico ({top['road_type']})",
                  fontsize=11, color=INK, loc="left", pad=10)
    ax2.set_xlabel("ora del giorno (locale)", fontsize=8.5, color=MUTED)
    ax2.set_ylabel("congestione", fontsize=8.5, color=MUTED)
    ax2.set_xticks(range(5, 24, 2))
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(GRID)

    ultima_data = pd.to_datetime(sez["timestamp_utc"]).max().strftime("%d/%m %H:%M UTC")
    fig.suptitle(f"Preview — dati raccolti fino al {ultima_data}, da confermare con più giorni di raccolta",
                 fontsize=9.5, color=MUTED, x=0.01, ha="left", y=1.04)

    fig.text(0.01, 0.005,
              f"Sinistra: SEZ2011 {SEZ_ESEMPIO} ({comune}) · {len(per_segmento)} segmenti monitorati totali "
              f"({len(robusti)} robusti, colorati · {len(non_robusti)} non robusti, in grigio) · "
              "dimensione del punto = n. letture · contorno tratteggiato = buffer di 50m per l'assegnazione. "
              "Destra: l'etichetta sopra ogni barra = n. letture in quell'ora.",
              fontsize=7.3, color=MUTED)

    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    nome_comune_file = comune.lower().replace(" ", "_").replace("'", "")
    out_path = CARTELLA_SCRIPT / f"preview_esempio_sezione_{nome_comune_file}.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
