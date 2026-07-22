"""
Step 2 del progetto parallelo (vector tiles) - script principale.

Scarica i 27 tile TomTom (Traffic Flow Vector Tile, stile "relative",
zoom 13) necessari a coprire le 50 sezioni critiche di Milano, decodifica
ciascun tile (formato Mapbox Vector Tile / protobuf) e ne estrae TUTTI i
segmenti stradali con il relativo rapporto di congestione. Per ogni
segmento si verifica poi a quale sezione appartiene (il segmento deve
intersecare il poligono della sezione, bufferizzato di 50m) e si tengono
solo i segmenti effettivamente dentro una delle 50 sezioni.

A differenza dell'endpoint puntuale "Flow Segment Data" (usato nel
progetto principale, che restituisce un solo segmento vicino a un punto
scelto a priori), qui si ottiene la copertura COMPLETA della rete
stradale della sezione in un'unica chiamata per tile: nessuna selezione
a priori di "candidati", il confronto tra segmenti puo' avvenire su
tutta la rete stradale realmente presente.

Nel campo "relative" di TomTom, traffic_level e' gia' il rapporto
currentSpeed/freeFlowSpeed (verificato incrociando con l'endpoint
puntuale): congestione = 1 - traffic_level.

Stessa logica di resilienza del progetto principale: se l'ora UTC
corrente e' gia' coperta da una lettura precedente, lo script esce senza
fare alcuna chiamata (pensato per essere lanciato ogni 15 minuti da
GitHub Actions, con margine molto piu' ampio: 27 chiamate/esecuzione
invece di 150).

API key TomTom:
  1. variabile d'ambiente TOMTOM_API_KEY (GitHub Actions, via secret);
  2. altrimenti SCRIPT/tomtom_key.txt (uso locale).

Output: traffico_tile_serie_storica_milano.csv, in append. Colonne:
        timestamp_utc, SEZ2011, COMUNE, gap_score, road_type,
        traffic_road_coverage, congestione, lat, lon
"""

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString

CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_GEOJSON = CARTELLA_SCRIPT / "top50_sezioni_critiche_milano.geojson"
IN_TILE = CARTELLA_SCRIPT / "tile_necessari.csv"
OUT_CSV = CARTELLA_SCRIPT / "traffico_tile_serie_storica_milano.csv"
KEY_PATH = CARTELLA_SCRIPT / "tomtom_key.txt"

TOMTOM_TILE_URL = "https://api.tomtom.com/traffic/map/4/tile/flow/relative/{zoom}/{x}/{y}.pbf"
PAUSA_TRA_RICHIESTE_S = 0.3
BUFFER_SEZIONE_METRI = 50

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32632"


def leggi_api_key():
    da_env = os.environ.get("TOMTOM_API_KEY")
    if da_env:
        return da_env.strip()
    return KEY_PATH.read_text(encoding="utf-8").strip()


def ora_gia_coperta(out_path, ora_utc_corrente):
    if not out_path.exists():
        return False
    prefisso_ora = ora_utc_corrente.strftime("%Y-%m-%dT%H")
    with open(out_path, encoding="utf-8") as f:
        next(f, None)
        for riga in f:
            if riga.startswith(prefisso_ora):
                return True
    return False


def tile_px_to_lonlat(x_tile, y_tile, zoom, px, py, extent):
    n = 2 ** zoom
    lon = (x_tile + px / extent) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y_tile + py / extent) / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def scarica_e_decodifica_tile(x, y, zoom, api_key, tentativi=3):
    for tentativo in range(1, tentativi + 1):
        r = requests.get(TOMTOM_TILE_URL.format(zoom=zoom, x=x, y=y),
                          params={"key": api_key}, timeout=20)
        if r.status_code == 200:
            if not r.content:
                return []
            return mapbox_vector_tile.decode(r.content)
        if r.status_code == 429:
            print("    rate limit (429), attendo 10s e riprovo...")
            time.sleep(10)
            continue
        print(f"    tentativo {tentativo}: HTTP {r.status_code} - {r.text[:200]}")
        time.sleep(3)
    return None


def estrai_segmenti_lonlat(tile_decodificato, x, y, zoom):
    """Ritorna una lista di dict {geometry (shapely, lon/lat), road_type,
    traffic_road_coverage, congestione} per tutti i segmenti del tile."""
    segmenti = []
    if not tile_decodificato:
        return segmenti

    for layer in tile_decodificato.values():
        extent = layer.get("extent", 4096)
        for feat in layer["features"]:
            props = feat["properties"]
            traffic_level = props.get("traffic_level")
            if traffic_level is None:
                continue
            geom = feat["geometry"]

            def converti_linea(coords_px):
                return [tile_px_to_lonlat(x, y, zoom, px, py, extent) for px, py in coords_px]

            if geom["type"] == "LineString":
                linea = LineString(converti_linea(geom["coordinates"]))
            elif geom["type"] == "MultiLineString":
                parti = [LineString(converti_linea(c)) for c in geom["coordinates"] if len(c) >= 2]
                if not parti:
                    continue
                linea = MultiLineString(parti) if len(parti) > 1 else parti[0]
            else:
                continue

            segmenti.append({
                "geometry": linea,
                "road_type": props.get("road_type"),
                "traffic_road_coverage": props.get("traffic_road_coverage"),
                "congestione": 1 - traffic_level,
            })
    return segmenti


def main(forza=False):
    ora_corrente = datetime.now(timezone.utc)

    if not forza and ora_gia_coperta(OUT_CSV, ora_corrente):
        print(f"Ora UTC {ora_corrente.strftime('%Y-%m-%dT%H')} gia' coperta da "
              f"un'esecuzione precedente: nessuna chiamata TomTom, esco.")
        return

    api_key = leggi_api_key()
    timestamp_utc = ora_corrente.isoformat(timespec="seconds")

    sezioni = gpd.read_file(IN_GEOJSON)[["SEZ2011", "COMUNE", "gap_score", "geometry"]]
    sezioni_utm = sezioni.to_crs(CRS_UTM)
    sezioni_buff_utm = sezioni_utm.copy()
    sezioni_buff_utm["geometry"] = sezioni_utm.geometry.buffer(BUFFER_SEZIONE_METRI)
    sezioni_buff = gpd.GeoDataFrame(
        sezioni_buff_utm[["SEZ2011", "COMUNE", "gap_score"]],
        geometry=sezioni_buff_utm.geometry, crs=CRS_UTM
    ).to_crs(CRS_WGS84)

    tile_df = pd.read_csv(IN_TILE)

    tutti_i_segmenti = []
    for i, row in tile_df.iterrows():
        x, y, zoom = int(row["tile_x"]), int(row["tile_y"]), int(row["zoom"])
        print(f"[{i+1}/{len(tile_df)}] tile ({x},{y},z{zoom})...")
        tile_decodificato = scarica_e_decodifica_tile(x, y, zoom, api_key)
        segmenti = estrai_segmenti_lonlat(tile_decodificato, x, y, zoom)
        print(f"    {len(segmenti)} segmenti nel tile")
        tutti_i_segmenti.extend(segmenti)
        time.sleep(PAUSA_TRA_RICHIESTE_S)

    if not tutti_i_segmenti:
        print("Nessun segmento scaricato, esco senza scrivere output.")
        return

    gdf_segmenti = gpd.GeoDataFrame(tutti_i_segmenti, geometry="geometry", crs=CRS_WGS84)
    gdf_segmenti_utm = gdf_segmenti.to_crs(CRS_UTM)

    # assegna ciascun segmento alla/e sezione/i con cui interseca (poligono bufferizzato)
    join = gpd.sjoin(gdf_segmenti, sezioni_buff, how="inner", predicate="intersects")

    righe = []
    for _, r in join.iterrows():
        centro = r["geometry"].centroid
        righe.append({
            "timestamp_utc": timestamp_utc,
            "SEZ2011": r["SEZ2011"],
            "COMUNE": r["COMUNE"],
            "gap_score": r["gap_score"],
            "road_type": r["road_type"],
            "traffic_road_coverage": r["traffic_road_coverage"],
            "congestione": r["congestione"],
            "lat": centro.y,
            "lon": centro.x,
            "assegnazione": "dentro_sezione",
            "distanza_m": 0.0,
        })

    # fallback "distance-aware" (stessa logica del progetto principale per
    # l'offerta di colonnine): per le sezioni senza alcun segmento dentro il
    # buffer, si assegna il segmento scaricato piu' vicino, segnalando la
    # distanza. Non e' un dato "interno" alla sezione ma la migliore proxy
    # disponibile fra i tile scaricati in questa esecuzione.
    sezioni_coperte = set(r["SEZ2011"] for r in righe)
    sezioni_mancanti = sezioni_utm[~sezioni_utm["SEZ2011"].isin(sezioni_coperte)]

    for sez, riga_sez in sezioni_mancanti.iterrows():
        distanze = gdf_segmenti_utm.geometry.distance(riga_sez.geometry)
        idx_min = distanze.idxmin()
        seg = gdf_segmenti.loc[idx_min]
        centro = seg["geometry"].centroid
        righe.append({
            "timestamp_utc": timestamp_utc,
            "SEZ2011": riga_sez["SEZ2011"],
            "COMUNE": riga_sez["COMUNE"],
            "gap_score": riga_sez["gap_score"],
            "road_type": seg["road_type"],
            "traffic_road_coverage": seg["traffic_road_coverage"],
            "congestione": seg["congestione"],
            "lat": centro.y,
            "lon": centro.x,
            "assegnazione": "piu_vicino_esterno",
            "distanza_m": round(distanze.loc[idx_min], 1),
        })

    out = pd.DataFrame(righe)
    n_sezioni_coperte = out["SEZ2011"].nunique() if len(out) else 0
    print(f"\nSegmenti totali scaricati: {len(gdf_segmenti)}")
    print(f"Segmenti assegnati a una sezione: {len(out)}")
    print(f"Sezioni con almeno un segmento: {n_sezioni_coperte} / {len(sezioni)}")

    file_esiste = OUT_CSV.exists()
    out.to_csv(OUT_CSV, mode="a", header=not file_esiste, index=False)
    print(f"{'Aggiunte' if file_esiste else 'Salvate'} {len(out)} righe in: {OUT_CSV}")


if __name__ == "__main__":
    import sys
    forza = "--forza" in sys.argv
    main(forza=forza)
