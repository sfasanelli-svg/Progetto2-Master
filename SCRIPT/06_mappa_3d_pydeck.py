# -*- coding: utf-8 -*-
"""
Step 6 del progetto parallelo (vector tiles) - mappa 3D.

Stessa logica dei grafici precedenti (congestione massima per segmento,
soglia di robustezza a MIN_LETTURE), ma come mappa 3D interattiva
(pydeck/deck.gl): ogni segmento robusto diventa una colonna estrusa sopra
una mappa reale (basemap CARTO, nessun token Mapbox richiesto), con
altezza proporzionale alla congestione massima osservata — un effetto
visivo molto d'impatto per evidenziare i punti più critici.

Come la mappa interattiva Folium (05), va aperta in un browser (richiede
internet per le tile della mappa) — non produce direttamente
un'immagine statica.

Output: mappa_3d_<comune>.html
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_CSV = CARTELLA_SCRIPT / "traffico_tile_serie_storica_milano.csv"
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"

SEZ_ESEMPIO = 151460002842  # cambiare qui per un'altra sezione
MIN_LETTURE = 5
ALTEZZA_MASSIMA_METRI = 250  # altezza della colonna piu' congestionata


def congestione_a_colore(valore, vmax):
    """Converte un valore di congestione in RGBA (0-255) con la stessa
    palette (YlOrRd) usata nei grafici statici, per coerenza visiva."""
    cmap = plt.get_cmap("YlOrRd")
    r, g, b, a = cmap(valore / vmax if vmax else 0)
    return [int(r * 255), int(g * 255), int(b * 255), 200]


def main():
    df = pd.read_csv(IN_CSV)
    sezioni = gpd.read_file(IN_GEOJSON)

    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)

    sez = df[df["SEZ2011"] == SEZ_ESEMPIO].copy()
    if sez.empty:
        raise ValueError(f"Nessun dato per SEZ2011={SEZ_ESEMPIO}")
    poligono = sezioni[sezioni["SEZ2011"].astype(int) == SEZ_ESEMPIO].iloc[0]
    comune = poligono["COMUNE"]

    per_segmento = sez.groupby(["lat_r", "lon_r", "road_type"]).agg(
        congestione_max=("congestione", "max"), n_letture=("congestione", "size")
    ).reset_index()
    robusti = per_segmento[per_segmento["n_letture"] >= MIN_LETTURE].sort_values(
        "congestione_max", ascending=False).copy()
    if robusti.empty:
        raise ValueError(f"Nessun segmento robusto per SEZ2011={SEZ_ESEMPIO}, serve piu' tempo di raccolta")

    vmax = robusti["congestione_max"].max()
    robusti["altezza"] = robusti["congestione_max"] / vmax * ALTEZZA_MASSIMA_METRI
    robusti["colore"] = robusti["congestione_max"].apply(lambda v: congestione_a_colore(v, vmax))
    robusti["congestione_txt"] = robusti["congestione_max"].round(2).astype(str)

    layer = pdk.Layer(
        "ColumnLayer",
        data=robusti,
        get_position=["lon_r", "lat_r"],
        get_elevation="altezza",
        elevation_scale=1,
        radius=12,
        get_fill_color="colore",
        pickable=True,
        auto_highlight=True,
    )

    centro = poligono.geometry.centroid
    view_state = pdk.ViewState(
        longitude=centro.x, latitude=centro.y, zoom=16, pitch=55, bearing=20,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"text": "{road_type}\ncongestione massima: {congestione_txt}\nletture: {n_letture}"},
    )

    nome_comune_file = comune.lower().replace(" ", "_").replace("'", "")
    out_path = CARTELLA_SCRIPT / f"mappa_3d_{nome_comune_file}.html"
    deck.to_html(str(out_path), notebook_display=False)
    print(f"Salvato: {out_path}")
    print(f"Colonne disegnate: {len(robusti)} (segmenti robusti, >= {MIN_LETTURE} letture)")


if __name__ == "__main__":
    main()
