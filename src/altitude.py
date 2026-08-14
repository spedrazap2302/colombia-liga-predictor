# Elevation in meters for each team's home city.
# Only cities with a real, well-documented effect need to be precise --
# sea-level teams can all safely share ~0-50m without changing results.
TEAM_ALTITUDE = {
    "Millonarios": 2640,
    "Santa Fe": 2640,
    "Boyaca C": 2790,     # Tunja
    "Bucaramanga": 959,
    "Nacional": 1495,     # Medellin
    "Medellin": 1495,
    "Envigado": 1495,
    "America": 1000,      # Cali
    "Cali": 1000,
    "Once Caldas": 2170,  # Manizales
    "Pereira": 1411,
    "Tolima": 431,        # Ibague
    "Huila": 442,         # Neiva
    "Pasto": 2527,
    "Jaguares": 30,       # Monteria
    "Junior": 20,         # Barranquilla (if this ever reappears in your data)
    "Inter Bogota": 2640,
    "Alianza": 1720,      # Valledupar area teams vary; adjust if needed
    "Llaneros": 442,      # Villavicencio
    "Fortaleza": 2640,    # Bogota-based
    "Patriotas": 2782,    # Tunja
    "Cucuta": 320,
    "Magdalena": 15,      # Santa Marta
}

DEFAULT_ALTITUDE = 1000  # fallback for any team not listed above


def get_altitude(team):
    return TEAM_ALTITUDE.get(team, DEFAULT_ALTITUDE)


def altitude_boost(home_team, away_team, points_per_1000m=25):
    """Extra Elo-style rating points added to the home team's effective
    rating, based on how much higher their city sits than the away
    team's city. Sea-level-vs-sea-level or same-altitude matchups get ~0."""
    home_alt = get_altitude(home_team)
    away_alt = get_altitude(away_team)
    altitude_gap = max(home_alt - away_alt, 0)  # only helps the HOME team if THEY'RE higher
    return (altitude_gap / 1000) * points_per_1000m