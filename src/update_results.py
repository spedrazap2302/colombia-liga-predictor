"""Adds new Liga BetPlay results to data/matches.csv automatically.

Where the data comes from: ESPN's website loads its Colombian Primera A
scoreboard (league code "col.1") from a public JSON address. Asking that
address for one day at a time gives every game played that day: teams,
score, stage and group. This script does the same thing the ESPN page does,
then fills the results into our CSV. It replaces typing results in by hand.

Run it from the project folder:
    python src/update_results.py              # catch up from the last result in the file
    python src/update_results.py --since 2026-08-01
    python src/update_results.py --dry-run    # show what would change, write nothing

A GitHub Action runs it every day (see .github/workflows/update-results.yml).
"""
import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

DATA_PATH = "data/matches.csv"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/col.1/scoreboard"
COLOMBIA_TZ = timezone(timedelta(hours=-5))  # Colombia is UTC-5 all year (no daylight saving)

# ESPN's team names -> the short names used in matches.csv.
# A team ESPN uses that isn't here (for example a newly promoted club) stops the
# script with a clear message, so nothing gets saved under the wrong name.
TEAM_MAP = {
    "Águilas Doradas": "Aguilas",
    "Alianza FC": "Alianza",
    "América de Cali": "America",
    "Atlético Bucaramanga": "Bucaramanga",
    "Atlético Junior": "Junior",
    "Atlético Nacional": "Nacional",
    "Boyacá Chicó FC": "Boyaca C",
    "Cúcuta Deportivo": "Cucuta",
    "Deportes Tolima": "Tolima",
    "Deportivo Cali": "Cali",
    "Deportivo Pasto": "Pasto",
    "Deportivo Pereira": "Pereira",
    "Envigado": "Envigado",
    "Fortaleza CEIF": "Fortaleza",
    "Independiente Medellín": "Medellin",
    "Independiente Santa Fe": "Santa Fe",
    "Internacional de Bogotá": "Inter Bogota",
    "Jaguares de Córdoba": "Jaguares",
    "Llaneros FC": "Llaneros",
    "Millonarios": "Millonarios",
    "Once Caldas": "Once Caldas",
}

COLUMNS = ["date", "season", "phase_type", "stage", "home_team", "away_team",
           "home_goals", "away_goals", "penalty_winner", "espn_id"]


# ---------------------------------------------------------------- downloading

def get_json(url, tries=3):
    """Downloads one ESPN page of JSON, retrying a couple of times if the
    connection hiccups."""
    import requests  # imported here so the parsing functions can be tested without it

    for attempt in range(tries):
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "colombia-liga-predictor (personal project)"})
            resp.raise_for_status()
            return resp.json()
        except Exception as err:  # network error or bad status
            if attempt == tries - 1:
                raise
            print(f"  retrying after error: {err}")
            time.sleep(3)


def match_days(since, until):
    """ESPN's scoreboard lists every date that has games (the 'calendar').
    Only those days are downloaded, one request each (ESPN refuses date ranges)."""
    calendar = get_json(SCOREBOARD)["leagues"][0]["calendar"]
    days = sorted({c[:10] for c in calendar})
    return [d for d in days if since.isoformat() <= d <= until.isoformat()]


def download_events(since, until):
    events = []
    for day in match_days(since, until):
        data = get_json(f"{SCOREBOARD}?dates={day.replace('-', '')}")
        events.extend(data.get("events", []))
        time.sleep(0.5)  # be polite: one request every half second
    return events


# ---------------------------------------------------------------- parsing

def local_date(utc_text):
    """ESPN gives kickoff in UTC ("2026-08-02T01:15Z"). A night game in
    Colombia is already the next day in UTC, so convert to Colombian time."""
    kickoff = datetime.strptime(utc_text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return kickoff.astimezone(COLOMBIA_TZ).date()


def season_label(year, slug, day):
    """'apertura' = first half (2026-I), 'clausura' = second half (2026-II).
    Playoff games don't say which half, so use the month."""
    if slug == "apertura":
        half = "I"
    elif slug == "clausura":
        half = "II"
    else:
        half = "I" if day.month <= 7 else "II"
    return f"{year}-{half}"


def phase_and_stage(slug, group):
    """Turns ESPN's stage name into our phase_type / stage columns."""
    if slug in ("apertura", "clausura"):
        return "regular", "regular"
    if "semifinal" in slug and group:            # cuadrangulares: "Group A" / "Group B"
        return "cuadrangular", "group_" + group.split()[-1]
    if "semifinal" in slug:                       # knockout format (used in 2026-I)
        return "knockout_top8", "semifinal"
    if "quarterfinal" in slug:
        return "knockout_top8", "quarterfinal"
    if "final" in slug:
        return "final", "final"                   # phase fixed later to match the season's format
    raise ValueError(f"Unknown ESPN stage '{slug}'. Add it to phase_and_stage() in update_results.py.")


def team_name(espn_name):
    if espn_name not in TEAM_MAP:
        raise SystemExit(f"New team name from ESPN: '{espn_name}'. "
                         f"Add it to TEAM_MAP in src/update_results.py (ESPN name -> our short name).")
    return TEAM_MAP[espn_name]


def parse_events(events):
    """Turns ESPN events into rows shaped like matches.csv. Only finished
    games are kept (ESPN lists games, not future fixtures, for this league)."""
    rows = []
    for e in events:
        if not e["status"]["type"].get("completed"):
            continue
        comp = e["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        day = local_date(e["date"])
        group = (comp.get("group") or {}).get("abbreviation")
        phase, stage = phase_and_stage(e["season"]["slug"], group)

        penalty_winner = None
        hs, as_ = home.get("shootoutScore"), away.get("shootoutScore")
        if hs is not None and as_ is not None and hs != as_:
            penalty_winner = team_name(home["team"]["displayName"] if hs > as_ else away["team"]["displayName"])

        rows.append({
            "date": day.isoformat(),
            "season": season_label(e["season"]["year"], e["season"]["slug"], day),
            "phase_type": phase,
            "stage": stage,
            "home_team": team_name(home["team"]["displayName"]),
            "away_team": team_name(away["team"]["displayName"]),
            "home_goals": int(home["score"]),
            "away_goals": int(away["score"]),
            "penalty_winner": penalty_winner,
            "espn_id": str(e["id"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------- merging

def merge_results(matches, new):
    """Puts ESPN results into the matches table without creating duplicates.

    For each ESPN game, in order:
      1. Already in the file (same espn_id): fix the score if it changed.
      2. A game we already entered by hand (same teams, season and stage, no
         espn_id yet): if the score agrees, just tag it with the espn_id;
         if it disagrees, keep ours and report it so a person can check.
      3. A fixture with no score yet (same teams, season and stage): fill it in.
      4. Otherwise it's a new game: add it.
    Returns the updated table and a log of what changed.
    """
    m = matches.copy()
    if "espn_id" not in m.columns:
        m["espn_id"] = pd.NA
    m["espn_id"] = m["espn_id"].astype("string")
    log = {"filled": [], "added": [], "tagged": [], "corrected": [], "conflicts": []}

    for _, g in new.iterrows():
        label = f"{g['date']} {g['home_team']} {g['home_goals']}-{g['away_goals']} {g['away_team']} ({g['season']} {g['stage']})"

        # the final's phase follows the season's format (cuadrangular or knockout)
        if g["phase_type"] == "final":
            season_phases = set(m.loc[m["season"] == g["season"], "phase_type"]) | set(new.loc[new["season"] == g["season"], "phase_type"])
            g["phase_type"] = "knockout_top8" if "knockout_top8" in season_phases else "cuadrangular"

        same_id = m.index[m["espn_id"] == g["espn_id"]]
        if len(same_id):
            i = same_id[0]
            if (m.at[i, "home_goals"], m.at[i, "away_goals"]) != (g["home_goals"], g["away_goals"]):
                m.loc[i, ["home_goals", "away_goals", "penalty_winner"]] = [g["home_goals"], g["away_goals"], g["penalty_winner"]]
                log["corrected"].append(label)
            continue

        same_game = ((m["season"] == g["season"]) & (m["home_team"] == g["home_team"])
                     & (m["away_team"] == g["away_team"]) & (m["phase_type"] == g["phase_type"])
                     & (m["stage"] == g["stage"]) & m["espn_id"].isna())
        days_apart = (pd.to_datetime(m["date"]) - pd.Timestamp(g["date"])).dt.days.abs()

        entered = m.index[same_game & m["home_goals"].notna() & (days_apart <= 3)]
        if len(entered):
            i = entered[0]
            m.at[i, "espn_id"] = g["espn_id"]
            if (m.at[i, "home_goals"], m.at[i, "away_goals"]) == (g["home_goals"], g["away_goals"]):
                log["tagged"].append(label)
            else:
                log["conflicts"].append(f"{label}  <- file has {m.at[i, 'home_goals']:.0f}-{m.at[i, 'away_goals']:.0f}")
            continue

        fixture = m.index[same_game & m["home_goals"].isna()]
        if len(fixture):
            # if the same pairing is listed twice, take the fixture closest in date
            i = days_apart[fixture].idxmin()
            m.loc[i, ["date", "home_goals", "away_goals", "penalty_winner", "espn_id"]] = [
                g["date"], g["home_goals"], g["away_goals"], g["penalty_winner"], g["espn_id"]]
            log["filled"].append(label)
            continue

        m = pd.concat([m, g.to_frame().T], ignore_index=True)
        log["added"].append(label)

    # Once a season's playoffs have started, any regular-season fixture still
    # without a score was never played as listed (postponed and played under
    # another date is caught above). Drop it so the simulator doesn't play it.
    log["dropped"] = []
    for season in new.loc[new["phase_type"] != "regular", "season"].unique():
        stale = m.index[(m["season"] == season) & (m["phase_type"] == "regular") & m["home_goals"].isna()]
        for i in stale:
            log["dropped"].append(f"{m.at[i, 'date']} {m.at[i, 'home_team']} vs {m.at[i, 'away_team']} ({season})")
        m = m.drop(index=stale)

    m["home_goals"] = pd.to_numeric(m["home_goals"]).astype("Int64")
    m["away_goals"] = pd.to_numeric(m["away_goals"]).astype("Int64")
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    return m[COLUMNS], log


def print_log(log):
    for key, title in [("filled", "Filled in fixtures"), ("added", "Added new games"),
                       ("corrected", "Scores corrected by ESPN"), ("tagged", "Already in the file (now linked to ESPN)"),
                       ("conflicts", "CHECK THESE: score in file differs from ESPN (file kept)"),
                       ("dropped", "Removed unplayed regular-season fixtures")]:
        if log.get(key):
            print(f"\n{title} ({len(log[key])}):")
            for line in log[key]:
                print("  " + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="first date to check, YYYY-MM-DD (default: a week before the last result in the file)")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without saving")
    parser.add_argument("--from-json", help="use a saved ESPN response file instead of downloading (for testing)")
    args = parser.parse_args()

    matches = pd.read_csv(DATA_PATH, dtype={"season": str, "espn_id": str})

    if args.from_json:
        with open(args.from_json) as f:
            saved = json.load(f)
        events = [e for day in saved.values() for e in day["events"]]
    else:
        last_result = pd.to_datetime(matches.dropna(subset=["home_goals"])["date"]).max().date()
        since = date.fromisoformat(args.since) if args.since else last_result - timedelta(days=7)
        today = datetime.now(COLOMBIA_TZ).date()
        print(f"Checking ESPN for games from {since} to {today}...")
        events = download_events(since, today)

    new = parse_events(events)
    print(f"ESPN finished games found: {len(new)}")
    updated, log = merge_results(matches, new)
    print_log(log)

    changed = any(log[k] for k in ("filled", "added", "corrected", "tagged", "dropped"))
    if not changed:
        print("\nNothing new. matches.csv left unchanged.")
    elif args.dry_run:
        print("\nDry run: nothing saved.")
    else:
        updated.to_csv(DATA_PATH, index=False)
        print(f"\nSaved {DATA_PATH} ({updated['home_goals'].notna().sum()} played, {updated['home_goals'].isna().sum()} upcoming).")


if __name__ == "__main__":
    main()
