# -*- coding: utf-8 -*-
"""
Step 5 del progetto parallelo (vector tiles) - mappa interattiva.

Stessa logica dei grafici precedenti (congestione massima per segmento,
soglia di robustezza a MIN_LETTURE), ma come mappa interattiva HTML
(Folium/Leaflet): zoomabile, con popup al passaggio del mouse su ogni
segmento. Pensata per una demo dal vivo durante una presentazione (si
apre in un browser), non per essere incollata in una slide statica.

Output: mappa_interattiva_<comune>.html
"""

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_CSV = CARTELLA_SCRIPT / "traffico_tile_serie_storica_milano.csv"
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"

SEZ_ESEMPIO = 151460002842  # cambiare qui per un'altra sezione
MIN_LETTURE = 5


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
    # a parita' di congestione massima il tie-break e' il numero di letture (il
    # segmento con piu' osservazioni a supporto e' il dato piu' affidabile), non
    # l'ordine arbitrario del groupby - vedi anche 03/04/06
    robusti = per_segmento[per_segmento["n_letture"] >= MIN_LETTURE].sort_values(
        ["congestione_max", "n_letture"], ascending=[False, False])
    non_robusti = per_segmento[per_segmento["n_letture"] < MIN_LETTURE]
    if robusti.empty:
        raise ValueError(f"Nessun segmento robusto per SEZ2011={SEZ_ESEMPIO}, serve piu' tempo di raccolta")
    top = robusti.iloc[0]

    centro = [poligono.geometry.centroid.y, poligono.geometry.centroid.x]
    m = folium.Map(location=centro, zoom_start=16, tiles="CartoDB positron")

    # perimetro sezione + buffer 50m
    folium.GeoJson(
        poligono.geometry.__geo_interface__,
        style_function=lambda f: {"color": "#2b2b2b", "weight": 2.5, "fillOpacity": 0},
        name="Perimetro sezione",
    ).add_to(m)
    buffer_50m = gpd.GeoSeries([poligono.geometry], crs="EPSG:4326").to_crs("EPSG:32632") \
        .buffer(50).to_crs("EPSG:4326").iloc[0]
    folium.GeoJson(
        buffer_50m.__geo_interface__,
        style_function=lambda f: {"color": "#2b2b2b", "weight": 1.5, "dashArray": "5,5", "fillOpacity": 0},
        name="Buffer 50m (assegnazione)",
    ).add_to(m)

    colormap = LinearColormap(
        colors=["#ffffcc", "#fd8d3c", "#800026"], vmin=0, vmax=robusti["congestione_max"].max(),
        caption=f"Congestione massima (min. {MIN_LETTURE} letture)"
    )
    colormap.add_to(m)

    # segmenti non robusti: grigio, popup che segnala il dato insufficiente
    for _, r in non_robusti.iterrows():
        folium.CircleMarker(
            location=[r["lat_r"], r["lon_r"]], radius=4, color="#bdbdbd", fill=True,
            fill_color="#bdbdbd", fill_opacity=0.7, weight=0,
            popup=folium.Popup(
                f"<b>{r['road_type']}</b><br>{int(r['n_letture'])} letture — "
                f"dato insufficiente (soglia minima {MIN_LETTURE})", max_width=220),
        ).add_to(m)

    # segmenti robusti: colore = congestione massima, raggio = n. letture
    for _, r in robusti.iterrows():
        e_top = (r["lat_r"] == top["lat_r"]) and (r["lon_r"] == top["lon_r"])
        folium.CircleMarker(
            location=[r["lat_r"], r["lon_r"]],
            radius=6 + min(r["n_letture"], 60) / 8,
            color="#c0392b" if e_top else "white",
            weight=3 if e_top else 1,
            fill=True, fill_color=colormap(r["congestione_max"]), fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{r['road_type']}</b><br>Congestione massima: {r['congestione_max']:.2f}<br>"
                f"Letture: {int(r['n_letture'])}"
                + ("<br><b>Segmento più critico</b>" if e_top else ""), max_width=220),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    nome_comune_file = comune.lower().replace(" ", "_").replace("'", "")
    out_path = CARTELLA_SCRIPT / f"mappa_interattiva_{nome_comune_file}.html"
    m.save(str(out_path))
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
