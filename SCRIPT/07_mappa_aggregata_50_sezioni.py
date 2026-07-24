# -*- coding: utf-8 -*-
"""
Step 7 del progetto parallelo (vector tiles) - mappe d'insieme delle 50 sezioni.

Genera DUE mappe separate, una per domanda:

  1. mappa_aggregata_50_sezioni_gap_score_milano.png
     "Quali sezioni" - le 50 sezioni colorate per gap_score, il punteggio di
     bisogno stimato (EV charge desert) che le ha selezionate a monte di
     tutto il progetto. E' la slide che giustifica perche' proprio queste 50.

  2. mappa_aggregata_50_sezioni_congestione_milano.png
     "Cosa abbiamo trovato" - le stesse 50 sezioni colorate per la congestione
     massima rilevata dal monitoraggio del traffico (vector tile). E' la
     slide che introduce il dettaglio per sezione di 04_mappa_reale_contextily.py.

Le due mappe sono tenute separate (non un'unica mappa a doppia codifica
colore+contorno) perche' rispondono a due domande diverse: gap_score dice
QUALI sezioni prioritizzare, la congestione dice DOVE dentro una sezione
scelta. Attenzione: la sezione con congestione piu' alta NON e' detto sia
anche la prima per gap_score (nei dati raccolti finora non lo e': la piu'
congestionata e' 13esima su 50 per gap_score) - le due mappe evidenziano
percio' due sezioni diverse, ciascuna la "prima" per la propria metrica, e
NON vanno presentate come se fossero la stessa sezione vista da due angoli.

Richiede una connessione internet (contextily scarica le tile della mappa
al volo).
"""

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_CSV = CARTELLA_SCRIPT / "traffico_tile_serie_storica_milano.csv"
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"

MIN_LETTURE = 5  # soglia minima di osservazioni per considerare un segmento di traffico "robusto"
N_ETICHETTE = 5  # quante sezioni in cima alla classifica etichettare direttamente
BOZZA = True  # metti a False quando la raccolta dati di traffico e' conclusa

INK = "#2b2b2b"
MUTED = "#5a5a5a"
ACCENT = "#c0392b"

# alla scala di queste mappe (tutte le 50 sezioni, ~80km di estensione) 500m
# corrispondono a solo ~11px: due punti "vicini sullo schermo" possono essere
# a 1-1.5km di distanza reale, quindi la soglia va tenuta piu' larga di quanto
# sembri necessario guardando le coordinate
SOGLIA_VICINANZA_M = 2000


def calcola_congestione_per_sezione():
    """Congestione massima per sezione, solo sui segmenti "robusti" (>= MIN_LETTURE
    osservazioni) - stessa definizione di "punto piu' critico" usata in 03/04."""
    df = pd.read_csv(IN_CSV)
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)

    per_segmento = df.groupby(["SEZ2011", "lat_r", "lon_r"]).agg(
        congestione_max=("congestione", "max"), n_letture=("congestione", "size")
    ).reset_index()
    robusti = per_segmento[per_segmento["n_letture"] >= MIN_LETTURE]

    per_sezione = robusti.groupby("SEZ2011").agg(
        congestione_max=("congestione_max", "max"),
        n_segmenti_robusti=("congestione_max", "size"),
    ).reset_index()

    periodo = (pd.to_datetime(df["timestamp_utc"]).min(), pd.to_datetime(df["timestamp_utc"]).max())
    return per_sezione, periodo


def disegna_mappa_aggregata(gdf, colonna, cmap_name, vmin, vmax, etichetta_colorbar,
                             titolo, sottotitolo_top1, formato_valore, nota_footer, out_path,
                             colonna_dati_mancanti=None, etichetta_dati_mancanti=None,
                             colonna_tie_break=None):
    """Disegna una mappa d'insieme delle sezioni in gdf (EPSG:3857, con colonna
    'centroid_lon'/'centroid_lat' gia' proiettate in geometry), colorate per
    'colonna', con la prima in classifica evidenziata e le prossime
    N_ETICHETTE - 1 etichettate direttamente (con anti-sovrapposizione)."""
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colonna_check = colonna if colonna_dati_mancanti is None else colonna_dati_mancanti
    mancanti = gdf[colonna_check].isna()
    gdf_ok = gdf[~mancanti]
    gdf_ko = gdf[mancanti]

    # a parita' di valore il tie-break e' colonna_tie_break (tipicamente il numero
    # di segmenti robusti a supporto): niente scelte arbitrarie legate all'ordine
    # dei dati, vedi anche 03/04/05/06
    if colonna_tie_break:
        classifica = gdf[~mancanti].sort_values([colonna, colonna_tie_break], ascending=[False, False])
    else:
        classifica = gdf[~mancanti].sort_values(colonna, ascending=False)
    top_n = classifica.head(N_ETICHETTE)
    riga_top1 = top_n.iloc[0]

    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)

    if len(gdf_ko):
        gdf_ko.plot(ax=ax, color="#bdbdbd", markersize=70, edgecolor="white", linewidth=0.8,
                    alpha=0.85, zorder=3, label=etichetta_dati_mancanti)

    gdf_ok.plot(ax=ax, column=colonna, cmap=cmap, vmin=vmin, vmax=vmax,
                markersize=90, edgecolor="white", linewidth=0.8, zorder=4)

    ax.scatter([riga_top1.geometry.x], [riga_top1.geometry.y], s=550, facecolor="none",
               edgecolor=ACCENT, linewidth=2.4, zorder=5)
    ax.annotate(f"{riga_top1['COMUNE']} — {sottotitolo_top1}",
                (riga_top1.geometry.x, riga_top1.geometry.y),
                xytext=(24, 24), textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, color=ACCENT, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.3,
                                 shrinkA=0, shrinkB=9, connectionstyle="arc3,rad=0.12"))

    # etichette dirette per le prossime N_ETICHETTE-1 (il primo posto e' gia' etichettato
    # sopra). Due sezioni vicine (es. stesso comune) possono avere centroidi a pochi
    # metri l'uno dall'altro: se un punto gia' etichettato e' troppo vicino, l'offset
    # dell'etichetta viene allontanato progressivamente per evitare che si sovrappongano.
    # Il punto del primo in classifica e' un caso speciale: la sua stessa etichetta e'
    # un blocco di testo grande piazzato in alto a destra (vedi sopra), quindi un vicino
    # va spedito in basso a sinistra invece che semplicemente "piu' in alto"
    # la mappa si autoscala all'estensione dei 50 punti: un'etichetta vicino al
    # bordo destro con ha="left" (cresce verso destra) esce dall'area della
    # mappa e finisce sopra la colorbar - va specchiata (ha="right")
    minx, _, maxx, _ = gdf.total_bounds
    largh = maxx - minx

    punto_top1 = (riga_top1.geometry.x, riga_top1.geometry.y)
    punti_etichettati = [punto_top1]
    for _, riga in top_n.iloc[1:].iterrows():
        x, y = riga.geometry.x, riga.geometry.y
        vicino_al_top1 = ((x - punto_top1[0]) ** 2 + (y - punto_top1[1]) ** 2) ** 0.5 < SOGLIA_VICINANZA_M
        vicini = sum(1 for px, py in punti_etichettati if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < SOGLIA_VICINANZA_M)
        vicino_bordo_destro = x > minx + 0.85 * largh
        if vicino_al_top1:
            # in basso a destra (non a sinistra): a sinistra rischia di finire sul
            # credito OpenStreetMap/CARTO se il punto e' vicino al bordo della mappa
            xytext, ha, va = (9, -9 - (vicini - 1) * 15), "left", "top"
        elif vicino_bordo_destro:
            xytext, ha, va = (-9, 9 + vicini * 15), "right", "bottom"
        else:
            xytext, ha, va = (9, 9 + vicini * 15), "left", "bottom"
        ax.annotate(f"{riga['COMUNE']} ({riga[colonna]:{formato_valore}})",
                    (x, y), xytext=xytext, textcoords="offset points", ha=ha, va=va,
                    fontsize=8.5, color=INK, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8,
                                     shrinkA=0, shrinkB=6) if vicini else None)
        punti_etichettati.append((x, y))

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10)

    ax.set_axis_off()
    ax.set_title(titolo, fontsize=15, color=INK, loc="left", pad=12, fontweight="bold")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(etichetta_colorbar, fontsize=10, color=MUTED)

    if len(gdf_ko):
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9,
                  edgecolor="none", fontsize=9)

    fig.text(0.01, 0.014, nota_footer, fontsize=10, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.055)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Salvato: {out_path}")
    return classifica


def main():
    sezioni = gpd.read_file(IN_GEOJSON)
    sezioni["SEZ2011"] = sezioni["SEZ2011"].astype(int)

    congestione_per_sezione, (inizio, fine) = calcola_congestione_per_sezione()

    riepilogo = sezioni[["SEZ2011", "COMUNE", "gap_score", "centroid_lat", "centroid_lon"]].merge(
        congestione_per_sezione, on="SEZ2011", how="left")

    n_insufficienti = int(riepilogo["congestione_max"].isna().sum())
    if n_insufficienti:
        print(f"Attenzione: {n_insufficienti} sezioni senza alcun segmento di traffico robusto "
              f"(>= {MIN_LETTURE} letture): {riepilogo.loc[riepilogo['congestione_max'].isna(), 'SEZ2011'].tolist()}")

    gdf = gpd.GeoDataFrame(
        riepilogo,
        geometry=gpd.points_from_xy(riepilogo["centroid_lon"], riepilogo["centroid_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    nota_bozza = " · preview, da confermare" if BOZZA else ""

    # ------------------------------------------------------------ mappa 1: gap_score
    # scala sul range effettivo di QUESTE 50 sezioni (gia' tutte sopra la soglia
    # del gomito usata a monte, 0.429): un 0-1 fisso le schiaccerebbe tutte
    # nella stessa fascia di colore, illeggibile
    vmin_gap = (gdf["gap_score"].min() // 0.05) * 0.05
    vmax_gap = -(-gdf["gap_score"].max() // 0.05) * 0.05  # arrotonda per eccesso al 0.05
    classifica_gap = disegna_mappa_aggregata(
        gdf, colonna="gap_score", cmap_name="YlOrRd", vmin=vmin_gap, vmax=vmax_gap,
        etichetta_colorbar="gap_score\n(bisogno stimato di colonnine)",
        titolo="Perché queste 50 sezioni — gap_score più alto dell'area di Milano",
        sottotitolo_top1="la più critica per gap_score",
        formato_valore=".3f",
        nota_footer=f"50 sezioni · gap_score tra {vmin_gap:.2f} e {vmax_gap:.2f} "
                     "(tutte sopra la soglia del gomito 0,429 calcolata a livello nazionale)",
        out_path=CARTELLA_SCRIPT / "mappa_aggregata_50_sezioni_gap_score_milano.png",
    )

    # ------------------------------------------------------------ mappa 2: congestione
    # scala fissa 0-1, stessa convenzione di 04_mappa_reale_contextily.py: qui la
    # confrontabilita' tra slide diverse conta piu' che sfruttare tutto il range
    classifica_cong = disegna_mappa_aggregata(
        gdf, colonna="congestione_max", cmap_name="YlOrRd", vmin=0, vmax=1,
        etichetta_colorbar=f"congestione massima per sezione\n(min. {MIN_LETTURE} letture per segmento)",
        titolo="Cosa dice il traffico — congestione massima rilevata nelle 50 sezioni",
        sottotitolo_top1="congestione più alta rilevata\n(vedi dettaglio slide successiva)",
        formato_valore=".2f",
        nota_footer=f"50 sezioni · dati dal {inizio.strftime('%d/%m %H:%M')} "
                     f"al {fine.strftime('%d/%m %H:%M UTC')}{nota_bozza}",
        out_path=CARTELLA_SCRIPT / "mappa_aggregata_50_sezioni_congestione_milano.png",
        colonna_tie_break="n_segmenti_robusti",
        colonna_dati_mancanti="congestione_max",
        etichetta_dati_mancanti="dati insufficienti (nessun segmento robusto)",
    )

    posizione_congestione_top1_in_gap = int(
        (gdf["gap_score"] > classifica_cong.iloc[0]["gap_score"]).sum()) + 1
    print(f"\nNota: la sezione con congestione piu' alta ({classifica_cong.iloc[0]['COMUNE']}) "
          f"e' {posizione_congestione_top1_in_gap}a su 50 per gap_score, non la prima: "
          "le due mappe evidenziano deliberatamente due sezioni diverse.")

    print("\nClassifica gap_score (prime 5):")
    print(classifica_gap[["SEZ2011", "COMUNE", "gap_score"]].head(5).to_string(index=False))
    print("\nClassifica congestione massima (prime 5):")
    print(classifica_cong[["SEZ2011", "COMUNE", "gap_score", "congestione_max", "n_segmenti_robusti"]]
          .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
