import pandas as pd
import os

print("--- Iniciando extracción y filtrado del dataset ---")

# Rutas de entrada y salida
raw_path = os.path.join("data", "raw", "Matches.csv")
processed_dir = os.path.join("data", "processed")
output_path = os.path.join(processed_dir, "premier_league_2014_2024.csv")

# Comprobar que existe el archivo original
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"No se ha encontrado el archivo en {raw_path}")

# Cargar el dataset completo
print("Cargando Matches.csv (esto puede tardar unos segundos)...")
df = pd.read_csv(raw_path, low_memory=False)

# Convertir la columna de fecha a tipo datetime para filtrar los años
df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
df["Year"] = df["MatchDate"].dt.year

# Filtrar: Premier League inglesa (Division == 'E0') entre 2014 y 2024
print("Filtrando Premier League (E0) entre 2014 y 2024...")
df_pl = df[(df["Division"] == "E0") & (df["Year"] >= 2014) & (df["Year"] <= 2024)].copy()

# Crear la variable objetivo binaria: HomeWin (1 si gana el local 'H', 0 si es empate 'D' o visitante 'A')
df_pl["HomeWin"] = (df_pl["FTResult"] == "H").astype(int)

# Guardar el dataset limpio
os.makedirs(processed_dir, exist_ok=True)
df_pl.to_csv(output_path, index=False)

print("\n--- ¡Dataset preparado con éxito! ---")
print(f"Total de partidos seleccionados: {len(df_pl)}")
print(f"Archivo guardado en: {output_path}")
print("\nDistribución del resultado (FTResult):")
print(df_pl["FTResult"].value_counts())
print("\nDistribución de la variable objetivo (HomeWin):")
print(df_pl["HomeWin"].value_counts(normalize=True).round(3))