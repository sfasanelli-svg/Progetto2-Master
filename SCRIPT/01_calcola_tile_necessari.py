"""
Step 1 del progetto parallelo (vector tiles).

Calcola, per le 50 sezioni critiche di Milano (stesso input del progetto
principale: top50_sezioni_critiche_milano.geojson), l'insieme MINIMO e
DEDUPLICATO di tile TomTom (zoom 15) necessari a coprire tutte le sezioni.
Un singolo tile a zoom 15 copre circa 1 km di lato a questa latitudine:
risoluzione piu' fine di zoom 13, con segmenti stradali piu' precisi.

A differenza dell'approccio a candidati puntuali (progetto principale,
3 chiamate/sezione), qui si scarica un tile per area: 50 sezioni sparse
richiedono solo ~63 tile totali (deduplicati) a zoom 15 - comunque MENO
chiamate del progetto a candidati (150), con copertura COMPLETA (tutti i
segmenti stradali nel tile) invece che campionata su pochi punti. La
quota mensile di questa API (200.000/mese) lascia ampio margine per
usare uno zoom piu' alto (piu' dettaglio) senza avvicinarsi al limite,
a differenza del progetto a candidati (20.000/mese) dove la cadenza
oraria va tenuta identica per garantire un confronto equo tra i due
metodi.

Output: tile_necessari.csv (colonne: tile_x, tile_y, zoom) e
        sezione_tile.csv (SEZ2011 -> tile_x, tile_y, zoom; una sezione
        puo' comparire su piu' righe se coperta da piu' tile)
"""

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"
OUT_TILE = CARTELLA_SCRIPT / "tile_necessari.csv"
OUT_MAPPA = CARTELLA_SCRIPT / "sezione_tile.csv"

ZOOM = 15


def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def main():
    gdf = gpd.read_file(IN_GEOJSON)

    righe_mappa = []
    tutti_i_tile = set()

    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        x1, y1 = lonlat_to_tile(minx, maxy, ZOOM)  # angolo NW
        x2, y2 = lonlat_to_tile(maxx, miny, ZOOM)  # angolo SE

        tile_sezione = set()
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                tile_sezione.add((x, y))

        for x, y in tile_sezione:
            righe_mappa.append({"SEZ2011": row["SEZ2011"], "tile_x": x, "tile_y": y, "zoom": ZOOM})
        tutti_i_tile |= tile_sezione

        print(f"{row['SEZ2011']} ({row['COMUNE']}): {len(tile_sezione)} tile")

    pd.DataFrame(righe_mappa).to_csv(OUT_MAPPA, index=False)
    pd.DataFrame(sorted(tutti_i_tile), columns=["tile_x", "tile_y"]).assign(zoom=ZOOM).to_csv(OUT_TILE, index=False)

    print(f"\nTile totali unici necessari: {len(tutti_i_tile)}")
    print(f"Salvati: {OUT_TILE}, {OUT_MAPPA}")


if __name__ == "__main__":
    main()
