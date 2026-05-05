import base64
import json
import os
import sqlite3
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from google import genai

# =========================================================
# 0. 基础设置
# =========================================================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="HCMC Grid Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "diagnosis_cache.db"
MODEL_NAME = "gemini-3-flash-preview"
PROMPT_VERSION = "v2_with_shared_mobility"
CONGESTION_FILE = BASE_DIR / "grid_500m_daytype_3hour_top1_3_with_congestion_reason.geojson"
LANDUSE_FILE    = BASE_DIR / "HCMC_grid_500_scaled.geojson"

# =========================================================
# 0.5 数据库缓存
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grid_diagnosis_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        grid_id TEXT UNIQUE NOT NULL,
        model_name TEXT,
        prompt_version TEXT,
        grid_profile_json TEXT NOT NULL,
        ai_result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def get_cached_diagnosis(grid_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM grid_diagnosis_cache
        WHERE grid_id = ?
    """, (grid_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "grid_id": row["grid_id"],
        "model_name": row["model_name"],
        "prompt_version": row["prompt_version"],
        "grid_profile": json.loads(row["grid_profile_json"]),
        "ai_result": json.loads(row["ai_result_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_cached_diagnosis(
    grid_id: str,
    grid_profile: Dict[str, Any],
    ai_result: Dict[str, Any]
):
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id FROM grid_diagnosis_cache WHERE grid_id = ?", (grid_id,))
    row = cur.fetchone()

    if row:
        cur.execute("""
            UPDATE grid_diagnosis_cache
            SET model_name = ?,
                prompt_version = ?,
                grid_profile_json = ?,
                ai_result_json = ?,
                updated_at = ?
            WHERE grid_id = ?
        """, (
            MODEL_NAME,
            PROMPT_VERSION,
            json.dumps(grid_profile, ensure_ascii=False),
            json.dumps(ai_result, ensure_ascii=False),
            now,
            grid_id
        ))
    else:
        cur.execute("""
            INSERT INTO grid_diagnosis_cache (
                grid_id, model_name, prompt_version,
                grid_profile_json, ai_result_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            grid_id,
            MODEL_NAME,
            PROMPT_VERSION,
            json.dumps(grid_profile, ensure_ascii=False),
            json.dumps(ai_result, ensure_ascii=False),
            now,
            now
        ))

    conn.commit()
    conn.close()
# =========================================================
# 1. 基础字典
# =========================================================
CLUSTER_NAME_MAP = {
    0: "Walkable Low-Density Areas",
    1: "Motorcycle-Dependent Transitional Areas",
    2: "Low-Activity Peripheral Areas",
    3: "High-Accessibility Urban Core Areas",
    4: "Hyper-Dense Underserved Areas",
}

LANDUSE_NAME_MAP = {
    "education":      "Education",
    "green":          "Green",
    "industrial":     "Industrial",
    "infrastructure": "Infrastructure",
    "mix_use":        "Mixed Use",
    "others":         "Others",
    "public_use":     "Public Use",
    "residential":    "Residential",
    "transportation": "Transportation",
    "water":          "Water",
}

LANDUSE_RATIO_FIELDS = [
    ("lu_education",      "Education"),
    ("lu_green",          "Green"),
    ("lu_industrial",     "Industrial"),
    ("lu_infrastructure", "Infrastructure"),
    ("lu_mix_use",        "Mixed Use"),
    ("lu_others",         "Others"),
    ("lu_public_use",     "Public Use"),
    ("lu_residential",    "Residential"),
    ("lu_transportation", "Transportation"),
    ("lu_water",          "Water"),
]

ZSCORE_FIELDS = [
    ("pop_est_grid_scaled",         "Population Intensity"),
    ("CI_final_mean_scaled",        "Congestion Level"),
    ("n_motorcycle_scaled",         "Motorcycle Usage"),
    ("share_motorcycle_all_scaled", "Motorcycle Dependence"),
    ("acc30_500m_mean_scaled",      "Transit Accessibility (Network Connectivity)"),
    ("walk_mean_500m_scaled",       "Local Walkability"),
    ("seat_sum_avg_scaled",         "Transit Supply (Seats per Unit Time)"),
]

# =========================================================
# 指标含义说明
# =========================================================
INDICATOR_DEFINITIONS = """
INDICATOR DEFINITIONS
─────────────────────
All z-score indicators below are standardized relative to the CITY-WIDE distribution
of all grid cells across Ho Chi Minh City. A z-score of 0 means exactly at the city
average; positive values mean above average; negative values mean below average.

Standardization was performed using Z-score normalization (mean=0, std=1) computed
across all valid grid cells. Grids with missing values for a given indicator were
excluded from that indicator's normalization but were NOT excluded from other indicators.
This means a grid may have a valid z-score for some indicators and null for others —
null values should be noted but must not distort the interpretation of available indicators.

- Population Intensity
    Estimated residential population density in this grid cell relative to the city average.
    High value → this grid is more densely populated than most grids in the city.
    Low value  → this grid is sparsely populated compared to the city average.

- Congestion Level
    Average congestion index across time periods, relative to the city average.
    High value → roads in this grid are more frequently or severely congested than
                 the typical grid in Ho Chi Minh City.
    Low value  → roads are relatively free-flowing compared to the city average.

- Motorcycle Usage
    Absolute count of motorcycles observed in this grid, relative to the city average.
    High value → more motorcycles physically present than the typical city grid.
    Low value  → fewer motorcycles than average.

- Motorcycle Dependence
    Share of motorcycles relative to all vehicles, compared to the city-wide average share.
    High value → motorcycles make up a larger proportion of traffic than is typical
                 across the city, indicating structural modal dependence on motorcycles.
    Low value  → more balanced modal split than average, or lower motorcycle share.

- Transit Accessibility (Network Connectivity)
    Measures how many destinations across the city can be reached via the PUBLIC TRANSPORT
    NETWORK within 30 minutes from this grid, relative to the city average.
    High value → this grid is better connected within the transit network than most grids;
                 residents can reach more places by public transport than the city average.
    Low value  → this grid is poorly connected to the transit network compared to the city
                 average; residents can reach fewer destinations even if buses pass nearby.
    IMPORTANT: This measures NETWORK REACH AND CONNECTIVITY, not the physical presence
    or frequency of buses. A grid can have buses running through it but still score low
    if those buses do not connect to many destinations across the city.

- Local Walkability
    Quality and continuity of the pedestrian environment within 500 m, relative to the
    city-wide average walkability score.
    High value → better sidewalks, safer crossings, more pleasant walking conditions
                 than most grids in the city.
    Low value  → pedestrian infrastructure is worse than the city average; walking is
                 more difficult, unsafe, or uncomfortable than typical.

- Transit Supply (Seats per Unit Time)
    Total public transport seat capacity available per unit time in this grid, relative
    to the city-wide average transit supply.
    This directly measures HOW MUCH public transport service is physically provided:
    more seats per unit time = more buses/routes operating with higher frequency.
    High value → this grid receives more transit service than the city average.
    Low value  → this grid is underserved by public transport relative to the city average,
                 regardless of how well the network connects to other destinations.
    CRITICAL DISTINCTION: Transit Accessibility and Transit Supply measure different things.
    · High Transit Accessibility + Low Transit Supply = the network theoretically connects
      this grid to many destinations, but actual buses are too infrequent or sparse to
      make that connectivity usable in practice.
    · Low Transit Accessibility + High Transit Supply = many buses run locally but they
      do not effectively connect to the broader city network.
    · Both low = severely underserved in both network reach and actual service provision.
    · Both high = well-served in both dimensions (uncommon outside the urban core).

VARIABLE NAME RULE
──────────────────
When writing any output text (diagnosis, weekday_pattern, weekend_pattern, problem_logic,
strategy reasons, summary), ALWAYS use the human-readable indicator names above.
NEVER use raw variable names such as:
  seat_sum_avg_scaled, acc30_500m_mean_scaled, walk_mean_500m_scaled,
  share_motorcycle_all_scaled, n_motorcycle_scaled, CI_final_mean_scaled,
  pop_est_grid_scaled, lu_residential, lu_mix_use, lu_public_use, lu_education,
  lu_industrial, lu_transportation, lu_infrastructure, lu_green, lu_water, lu_others,
  or any other underscore_style field name.

Always substitute:
  seat_sum_avg_scaled          → Transit Supply
  acc30_500m_mean_scaled       → Transit Accessibility
  walk_mean_500m_scaled        → Local Walkability
  share_motorcycle_all_scaled  → Motorcycle Dependence
  n_motorcycle_scaled          → Motorcycle Usage
  CI_final_mean_scaled         → Congestion Level
  pop_est_grid_scaled          → Population Intensity
  lu_residential               → residential land use ratio
  lu_mix_use                   → mixed-use land use ratio
  lu_public_use                → public-use land use ratio
  lu_education                 → education land use ratio
  (and similarly for all other lu_ fields)
"""

# =========================================================
# 指标 → 策略触发映射
# =========================================================
INDICATOR_TO_STRATEGY_MAPPING = """
INDICATOR-TO-STRATEGY TRIGGER MAPPING
──────────────────────────────────────
All thresholds below refer to city-wide z-scores. A value of 0 = city average.
Use these as explicit evidence when deciding which strategies to recommend.
You are NOT required to select from every family — only recommend what the data supports.

── Transit Supply ─────────────────────────────────────────────────────────
  < -0.5   → This grid receives less transit service than the city average.
             Consider "Bus/Metro Operation" family:
             · "Increase service frequency"  — especially if Population Intensity > +0.5
               (high demand relative to city average but insufficient supply)
             · "Bus priority infrastructure" — if Congestion Level > +0.5
               (buses are delayed by congestion, compounding the supply gap)
             · "Dynamic bus/metro dispatch"  — if congestion varies across time periods
             · "Bus–Metro coordination"      — if Transit Accessibility > 0
               (network reach exists but supply does not match its potential)

  < -1.5 AND Population Intensity > +1.0
             → Severe supply-demand mismatch relative to city norms.
               "Increase service frequency" is strongly indicated.

── Transit Accessibility (Network Connectivity) ────────────────────────────
  < -0.5   → This grid is more poorly connected to the transit network than
             the city average; residents can reach fewer destinations by public transport.
             Consider:
             · "Bus–Metro coordination" to extend network reach via feeder integration
             · "Integrated mobility systems" to compensate for weak network coverage

  > +1.0 AND Local Walkability < -0.5
             → Network connectivity is above city average but pedestrian access to it
               is below city average — a usability gap between reach and access.
               Strongly consider:
               · "Station-area pedestrian design"
               · "Pedestrian street improvement"
               · "Safe street crossings"

  > +1.0 AND Transit Supply < -0.5
             → Network is well-connected relative to the city, but actual service
               provision is below average — the network's potential is underutilized.
               Consider "Increase service frequency" or "Bus priority infrastructure".

── Local Walkability ───────────────────────────────────────────────────────
  < -0.5   → Pedestrian environment is worse than the city average.
             Consider "Walkability and Station Access" family:
             · "Pedestrian street improvement"   — general improvement below city norm
             · "Safe street crossings"           — if Congestion Level or Motorcycle Usage
               is also above average (dangerous crossing conditions)
             · "Station-area pedestrian design"  — if Transit Accessibility > 0
               (stations are reachable via network but hard to access on foot)
             · "Accessibility improvements"      — if land use includes
               public-use ratio > 0.1 or education ratio > 0.1

  < -1.5   → Walkability is severely below city average; recommend at least
             2 sub-strategies from "Walkability and Station Access".

── Motorcycle Dependence ───────────────────────────────────────────────────
  > +0.5   → Motorcycles make up a higher share of traffic than the city average,
             indicating structural modal dependence beyond the city norm.
             Consider "First/Last Mile Connection":
             · "Shared electric motorcycles"   — general motorcycle dependence
             · "Micro-mobility hubs"           — if Transit Accessibility > 0
               (stations exist in the network but access is motorcycle-dependent)
             · "Electrification of mobility"   — if Motorcycle Usage is also above average
             · "Integrated mobility systems"   — if Transit Accessibility and/or
               Local Walkability are also below average (fragmented modal environment)

  > +1.5   → Strong structural motorcycle dependence, well above city norm.
             At least 1-2 sub-strategies from "First/Last Mile Connection" are likely appropriate.

── Motorcycle Usage ────────────────────────────────────────────────────────
  > +0.5   → More motorcycles physically present than the city average.
             If Congestion Level is also above average:
             · "Motorcycle lane organization" from "Traffic Management"
             · "Intersection management" for time-concentrated congestion

── Congestion Level ────────────────────────────────────────────────────────
  > +0.5   → More congested than the city average. Consider "Traffic Management":
             · "Motorcycle lane organization"    — if Motorcycle Usage also above average
             · "Intersection management"         — if congestion concentrated in specific
               time periods rather than all-day
             · "One-way street systems"          — if residential + mixed-use ratio > 0.4
               (dense local street network)
             · "Real-time monitoring & control"  — if congestion spans 3+ time periods
               or shows significant weekday/weekend differences

  Congestion type = "severe mismatch"
             → Both demand and supply are problematic relative to city norms.
               Consider strategies from BOTH "Bus/Metro Operation" AND "Traffic Management".

  Congestion type = "supply shortage"
             → Infrastructure is the bottleneck. Prioritize "Traffic Management"
               and "Bus priority infrastructure".

  Congestion type = "demand pressure"
             → Travel demand exceeds capacity. Prioritize "Increase service frequency"
               and demand-side strategies.

── Real-time monitoring & control (special trigger) ───────────────────────
  Appropriate when ANY of:
  · Congestion occurs across 3 or more distinct time periods, OR
  · Congestion patterns differ significantly between weekday and weekend, OR
  · Congestion Level > +1.0 AND Motorcycle Usage > +1.0 (both well above city average)

── Accessibility improvements (special trigger) ────────────────────────────
  Appropriate when:
  · Local Walkability < -0.5 (below city average) AND
  · public-use land use ratio > 0.1 OR education land use ratio > 0.1
"""

CLUSTER_DEFINITIONS = {
    "Walkable Low-Density Areas": {
        "profile": {
            "transit_accessibility": "-0.43 (below city average)",
            "walkability":           "+1.01 (much above city average)",
            "motorcycle_usage":      "-0.62 (moderately below city average)",
            "population_density":    "-0.40 (below city average)",
        },
        "interpretation": (
            "Lower-density areas with strong local walkability and relatively limited need for "
            "long-distance travel compared to the city norm. A significant share of daily activities "
            "can be satisfied locally, resulting in lower demand for network-based transport."
        ),
        "planning_meaning": (
            "These are low-density areas with limited travel demand relative to the city. "
            "Large-scale transport investment is not a priority. Planning should focus on maintaining "
            "local walkability and providing selective and demand-responsive public transport "
            "connections where necessary."
        ),
    },
    "Low-Activity Peripheral Areas": {
        "profile": {
            "transit_accessibility": "-0.37 (below city average)",
            "walkability":           "-0.46 (below city average)",
            "motorcycle_usage":      "-0.61 (moderately below city average)",
            "population_density":    "-0.41 (below city average)",
        },
        "interpretation": (
            "Peripheral areas with below-average density and generally weak mobility systems "
            "compared to the city norm, in terms of both local walkability and public transport "
            "network reach. Travel demand is limited, and accessibility constraints reflect both "
            "spatial isolation and underdeveloped infrastructure relative to the city average."
        ),
        "planning_meaning": (
            "These are low-density peripheral areas with generally low mobility demand compared "
            "to the city. Rather than intensive infrastructure expansion, planning should prioritize "
            "basic connectivity and ensure minimum public transport provision, with targeted and "
            "flexible services introduced where accessibility gaps are evident."
        ),
    },
    "Motorcycle-Dependent Transitional Areas": {
        "profile": {
            "transit_accessibility": "+0.15 (around city average)",
            "walkability":           "-0.36 (moderately below city average)",
            "motorcycle_usage":      "+1.16 (much above city average)",
            "population_density":    "-0.04 (around city average)",
        },
        "interpretation": (
            "Areas with around-average public transport network reach but below-average local "
            "walkability, leading to strong reliance on motorcycles as a flexible connector — "
            "motorcycle usage is well above the city norm. Public transport provides some regional "
            "connectivity, but weak walking conditions limit its effective use."
        ),
        "planning_meaning": (
            "These areas are suitable for targeted integration strategies. Planning should improve "
            "first- and last-mile connections to transit, strengthen feeder systems, and enhance "
            "walkability around key nodes so that existing network accessibility can be used "
            "more effectively."
        ),
    },
    "High-Accessibility Urban Core Areas": {
        "profile": {
            "transit_accessibility": "+3.42 (far above city average)",
            "walkability":           "-1.08 (much below city average)",
            "motorcycle_usage":      "+0.80 (above city average)",
            "population_density":    "+1.61 (above city average)",
        },
        "interpretation": (
            "Areas with extremely high public transport network reach — far above the city average — "
            "enabling access to a wide range of destinations. However, Local Walkability is well below "
            "the city average, indicating that local movement and micro-level accessibility are "
            "constrained, creating friction despite strong network connectivity."
        ),
        "planning_meaning": (
            "The main issue here is not insufficient network accessibility, but the imbalance between "
            "exceptional network reach and below-average local mobility conditions. Planning should "
            "improve pedestrian environments, reduce mode conflicts, and enhance internal movement "
            "efficiency within already well-connected areas."
        ),
    },
    "Hyper-Dense Underserved Areas": {
        "profile": {
            "transit_accessibility": "+0.63 (moderately above city average)",
            "walkability":           "-0.92 (below city average)",
            "motorcycle_usage":      "+0.90 (above city average)",
            "population_density":    "+2.33 (far above city average)",
        },
        "interpretation": (
            "Very high-density areas — far above the city average — with moderately above-average "
            "public transport network reach. Although these areas are connected to the broader urban "
            "network, the level of accessibility and transit supply does not match the intensity of "
            "demand, leading to structural pressure and system overload relative to city norms."
        ),
        "planning_meaning": (
            "These are critical mismatch zones where mobility demand far exceeds the effective level "
            "of transport provision compared to the city average. Planning should focus on expanding "
            "effective network capacity, improving both access to transit and internal circulation, "
            "and reducing the imbalance between population concentration and mobility supply."
        ),
    },
}

CONGESTION_DEFINITIONS = {
    "demand pressure": (
        "Congestion caused mainly by high travel demand. Infrastructure may exist, but the volume "
        "of trips temporarily exceeds normal operating capacity."
    ),
    "supply shortage": (
        "Congestion caused mainly by insufficient road capacity, weak traffic organization, "
        "bottlenecks, or inadequate transport supply."
    ),
    "severe mismatch": (
        "Congestion caused by both high demand and insufficient supply simultaneously. "
        "This indicates a structural mismatch rather than a single-factor problem."
    ),
    "moderate pressure": (
        "Mild to moderate congestion that does not necessarily indicate a severe structural issue. "
        "It may still require localized improvement."
    ),
}

STRATEGY_LIBRARY = {
    "Bus/Metro Operation": {
        "goal": (
            "Improve public transport reliability, capacity, and coordination in areas with "
            "high demand, strong transit potential, or overloaded core corridors."
        ),
        "typical_use_cases": [
            "Transit Supply below city average (< -0.5)",
            "high Population Intensity with insufficient service",
            "Transit Accessibility above average but Transit Supply below average",
            "peak-hour overload",
            "urban core network saturation",
            "congestion type = demand pressure or severe mismatch",
        ],
        "sub_strategies": {
            "Increase service frequency": {
                "what_it_means": (
                    "Increase bus or metro frequency, especially during peak periods, "
                    "to reduce waiting time and absorb concentrated demand."
                ),
                "best_for": [
                    "Transit Supply below city average",
                    "Population Intensity above city average",
                    "demand pressure congestion type",
                    "peak-hour congestion",
                ],
            },
            "Bus priority infrastructure": {
                "what_it_means": (
                    "Use dedicated bus lanes, bus-priority signals, and stop optimization "
                    "to improve bus reliability and travel speed."
                ),
                "best_for": [
                    "supply shortage or severe mismatch congestion type",
                    "Congestion Level above city average — buses delayed by mixed traffic",
                    "corridors where buses exist but are slowed by road conditions",
                ],
            },
            "Dynamic bus/metro dispatch": {
                "what_it_means": (
                    "Adjust dispatch and scheduling in real time using demand and delay data "
                    "to better match service supply to actual conditions."
                ),
                "best_for": [
                    "congestion that varies across time periods",
                    "significant weekday vs weekend pattern differences",
                    "recurrent peak-hour overload with off-peak underuse",
                ],
            },
            "Bus–Metro coordination": {
                "what_it_means": (
                    "Improve feeder connections, transfers, and timetable coordination between "
                    "buses and metro lines to reduce friction within the public transport network."
                ),
                "best_for": [
                    "Transit Accessibility above average but Transit Supply below average",
                    "Transit Accessibility below city average — network extension via feeders needed",
                    "areas where network connectivity exists but actual service integration is poor",
                ],
            },
        },
    },
    "Walkability and Station Access": {
        "goal": (
            "Improve pedestrian access, safety, comfort, and station reachability so that "
            "public transport and local trips are more usable and attractive."
        ),
        "typical_use_cases": [
            "Local Walkability below city average (< -0.5)",
            "Transit Accessibility above average but Local Walkability below average — "
            "network reachable but hard to access on foot",
            "unsafe walking environment due to mixed traffic above city norm",
            "land use with public-use or education ratio > 0.1 (vulnerable users)",
        ],
        "sub_strategies": {
            "Pedestrian street improvement": {
                "what_it_means": (
                    "Improve sidewalks, remove obstacles, and provide shading and better "
                    "walking conditions."
                ),
                "best_for": [
                    "Local Walkability below city average",
                    "mixed-use and residential areas",
                    "general local access improvement",
                ],
            },
            "Safe street crossings": {
                "what_it_means": (
                    "Introduce or improve crossings through signals, shorter crossing distances, "
                    "and traffic calming measures."
                ),
                "best_for": [
                    "Local Walkability below average combined with above-average Motorcycle Usage or Congestion Level",
                    "busy corridors with dangerous crossing conditions",
                    "areas where motorcycles and pedestrians conflict",
                ],
            },
            "Station-area pedestrian design": {
                "what_it_means": (
                    "Create direct, legible, and pedestrian-priority walking routes to stations or stops."
                ),
                "best_for": [
                    "Transit Accessibility above city average but Local Walkability below average",
                    "stations physically present in the network but difficult to reach on foot",
                    "public-use, education, and mixed-use areas near transit nodes",
                ],
            },
            "Accessibility improvements": {
                "what_it_means": (
                    "Provide ramps, barrier-free access, and inclusive design for elderly "
                    "and disabled users."
                ),
                "best_for": [
                    "Local Walkability below city average AND public-use ratio > 0.1 or education ratio > 0.1",
                    "station access issues affecting non-able-bodied users",
                    "areas with schools, hospitals, or civic facilities",
                ],
            },
        },
    },
    "First/Last Mile Connection": {
        "goal": (
            "Strengthen the connection between local origins/destinations and public transport "
            "corridors, especially where motorcycles currently fill the access gap."
        ),
        "typical_use_cases": [
            "Motorcycle Dependence above city average (> +0.5)",
            "weak feeder services",
            "Transit Accessibility around average but Local Walkability below average",
            "transitional areas where motorcycles replace missing first/last-mile options",
        ],
        "sub_strategies": {
            "Shared electric motorcycles": {
                "what_it_means": (
                    "Provide shared e-motorcycles and designated pick-up/drop-off zones "
                    "to reduce inefficient private motorcycle dependence."
                ),
                "best_for": [
                    "Motorcycle Dependence above city average",
                    "short feeder trip distances to transit",
                    "transitional areas needing flexible low-cost access",
                ],
            },
            "Micro-mobility hubs": {
                "what_it_means": (
                    "Create organized hubs for docking, parking, charging, or battery swap "
                    "near transit stations and key nodes."
                ),
                "best_for": [
                    "Motorcycle Dependence above average with Transit Accessibility also above average",
                    "station-area access disorganization",
                    "motorcycle-heavy feeder patterns needing structure",
                ],
            },
            "Electrification of mobility": {
                "what_it_means": (
                    "Encourage electric mobility use through incentives, charging/swap systems, "
                    "and low-emission station-area policies."
                ),
                "best_for": [
                    "Motorcycle Usage above city average AND Motorcycle Dependence above average",
                    "corridors aiming to reduce emissions while preserving feeder flexibility",
                ],
            },
            "Integrated mobility systems": {
                "what_it_means": (
                    "Integrate feeder services, shared mobility, and public transport through "
                    "planning apps, payment systems, and coordination."
                ),
                "best_for": [
                    "Motorcycle Dependence above average AND Transit Accessibility below average — fragmented modal environment",
                    "areas where Transit Supply, Local Walkability, and Motorcycle Dependence are all problematic",
                    "transitional and mixed-use areas with complex trip patterns",
                ],
            },
        },
    },
    "Traffic Management": {
        "goal": (
            "Improve road-space efficiency, reduce mixed-traffic conflicts, and manage "
            "bottlenecks through design and operational control."
        ),
        "typical_use_cases": [
            "Congestion Level above city average (> +0.5)",
            "supply shortage or severe mismatch congestion type",
            "Motorcycle Usage above average causing mixed-traffic conflict",
            "intersection or corridor bottlenecks",
        ],
        "sub_strategies": {
            "One-way street systems": {
                "what_it_means": (
                    "Convert selected narrow or conflict-heavy roads into one-way systems "
                    "to improve circulation and reduce friction."
                ),
                "best_for": [
                    "supply shortage congestion in dense residential or mixed-use areas",
                    "narrow street networks with bidirectional conflict",
                    "disorganized local circulation",
                ],
            },
            "Motorcycle lane organization": {
                "what_it_means": (
                    "Use dedicated motorcycle lanes, markings, and separators to better "
                    "organize mixed traffic."
                ),
                "best_for": [
                    "Motorcycle Usage above city average AND Congestion Level above average",
                    "mixed traffic conflict between motorcycles and other vehicles",
                    "transitional areas with dominant motorcycle presence",
                ],
            },
            "Intersection management": {
                "what_it_means": (
                    "Improve intersection operations through motorcycle waiting boxes, "
                    "signal optimization, and conflict reduction."
                ),
                "best_for": [
                    "congestion concentrated in specific time periods rather than all-day",
                    "Motorcycle Usage above average at intersections",
                    "localized severe mismatch at key nodes",
                ],
            },
            "Real-time monitoring & control": {
                "what_it_means": (
                    "Use traffic sensors, cameras, and adaptive control to monitor and "
                    "adjust traffic conditions in real time."
                ),
                "best_for": [
                    "congestion across 3 or more distinct time periods",
                    "significant weekday vs weekend pattern differences",
                    "Congestion Level well above average AND Motorcycle Usage well above average",
                    "operationally complex, variable demand conditions",
                ],
            },
        },
    },
}

# =========================================================
# 3. Pydantic Models
# =========================================================
class RecommendedStrategyModel(BaseModel):
    family: str
    strategy: str
    reason: str
    priority: str


class ImagePromptRequest(BaseModel):
    grid_id: str = Field(..., description="Selected grid id")
    cluster_name: str = Field(..., description="Cluster name of selected grid")
    dominant_lu: str = Field(..., description="Dominant land use code or name")
    diagnosis: str = Field(..., description="Diagnosis text already generated for the selected grid")
    problem_logic: str = Field(default="", description="Problem logic text from previous diagnosis")
    recommended_strategies: List[RecommendedStrategyModel] = Field(default_factory=list)
    street_image: Dict[str, Any] = Field(..., description="Selected street image info")


# =========================================================
# 4. 数据读取和索引
# =========================================================
def load_geojson(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_time_key(time_str: str) -> int:
    try:
        hh, mm = time_str.split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 999999


def _zscore_label(value: Optional[float]) -> str:
    if value is None:
        return "N/A (missing — this grid was excluded from this indicator's city-wide normalization)"
    if value >= 2.5:
        return f"{value:+.2f} (far above city average)"
    if value >= 1.0:
        return f"{value:+.2f} (above city average)"
    if value >= 0.3:
        return f"{value:+.2f} (slightly above city average)"
    if value > -0.3:
        return f"{value:+.2f} (around city average)"
    if value > -1.0:
        return f"{value:+.2f} (slightly below city average)"
    if value > -2.5:
        return f"{value:+.2f} (below city average)"
    return f"{value:+.2f} (far below city average)"


def build_indexes():
    congestion_geojson = load_geojson(CONGESTION_FILE)
    landuse_geojson    = load_geojson(LANDUSE_FILE)

    landuse_by_id: Dict[str, Dict[str, Any]]       = {}
    congestion_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for feature in landuse_geojson.get("features", []):
        props   = feature.get("properties", {})
        grid_id = str(props.get("id"))
        if not grid_id or grid_id == "None":
            continue

        cluster_code     = props.get("cluster_30min")
        cluster_name     = CLUSTER_NAME_MAP.get(cluster_code, f"Cluster {cluster_code}")
        dominant_lu_code = props.get("dominant_lu", "N/A")

        landuse_ratios: Dict[str, float] = {}
        for field, _ in LANDUSE_RATIO_FIELDS:
            landuse_ratios[field] = float(props.get(field, 0) or 0)

        zscore_raw: Dict[str, Optional[float]] = {}
        for field, _ in ZSCORE_FIELDS:
            raw = props.get(field)
            zscore_raw[field] = float(raw) if raw is not None else None

        landuse_by_id[grid_id] = {
            "id":               grid_id,
            "cluster_code":     cluster_code,
            "cluster_name":     cluster_name,
            "dominant_lu_code": dominant_lu_code,
            "dominant_lu_name": LANDUSE_NAME_MAP.get(dominant_lu_code, dominant_lu_code),
            **landuse_ratios,
            **zscore_raw,
        }

    for feature in congestion_geojson.get("features", []):
        props   = feature.get("properties", {})
        grid_id = str(props.get("id"))
        if not grid_id or grid_id == "None":
            continue
        congestion_by_id[grid_id].append({
            "day_type":        props.get("day_type",        "Unknown"),
            "time_3hour_str":  props.get("time_3hour_str",  "Unknown"),
            "congestion_type": props.get("congestion_type", "Unknown"),
        })

    for grid_id in congestion_by_id:
        congestion_by_id[grid_id] = sorted(
            congestion_by_id[grid_id],
            key=lambda x: (x["day_type"], sort_time_key(x["time_3hour_str"]))
        )

    return landuse_by_id, congestion_by_id


LANDUSE_BY_ID, CONGESTION_BY_ID = build_indexes()
init_db()

# =========================================================
# 5. Grid profile 生成
# =========================================================
def summarize_landuse_mix(base: Dict[str, Any]) -> Dict[str, Any]:
    ratios = []
    for field, label in LANDUSE_RATIO_FIELDS:
        value = float(base.get(field, 0) or 0)
        ratios.append({"field": field, "label": label, "value": round(value, 6)})
    ratios_sorted = sorted(ratios, key=lambda x: x["value"], reverse=True)
    return {
        "landuse_ratio_breakdown":     ratios_sorted,
        "top_landuse_components":      ratios_sorted[:3],
        "non_zero_landuse_components": [r for r in ratios_sorted if r["value"] > 0],
    }


def build_zscore_profile(base: Dict[str, Any]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for field, label in ZSCORE_FIELDS:
        raw   = base.get(field)
        value = float(raw) if raw is not None else None
        profile[label] = {
            "zscore":         round(value, 3) if value is not None else None,
            "interpretation": _zscore_label(value),
        }
    return profile


def derive_triggered_strategies(
    base: Dict[str, Any],
    congestion_events: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Pre-compute which (family → strategy) pairs are supported by the z-score
    values and congestion event data. All reason strings use human-readable
    indicator names and city-average framing — no raw variable names.
    """

    def get(field: str) -> Optional[float]:
        v = base.get(field)
        return float(v) if v is not None else None

    seat     = get("seat_sum_avg_scaled")
    acc      = get("acc30_500m_mean_scaled")
    walk     = get("walk_mean_500m_scaled")
    moto_dep = get("share_motorcycle_all_scaled")
    moto_use = get("n_motorcycle_scaled")
    ci       = get("CI_final_mean_scaled")
    pop      = get("pop_est_grid_scaled")
    lu_pub   = float(base.get("lu_public_use", 0) or 0)
    lu_edu   = float(base.get("lu_education",  0) or 0)
    lu_res   = float(base.get("lu_residential", 0) or 0)
    lu_mix   = float(base.get("lu_mix_use",     0) or 0)

    congestion_types = {e["congestion_type"].lower() for e in congestion_events}
    n_time_periods   = len(congestion_events)
    weekday_events   = [e for e in congestion_events if str(e["day_type"]).lower() == "weekday"]
    weekend_events   = [e for e in congestion_events if str(e["day_type"]).lower() == "weekend"]
    has_day_diff     = bool(weekday_events) != bool(weekend_events) or (
        weekday_events and weekend_events and
        {e["congestion_type"] for e in weekday_events} != {e["congestion_type"] for e in weekend_events}
    )

    triggered: Dict[str, List[str]] = {}

    def add(family: str, strategy: str, reason: str) -> None:
        key = f"{family} → {strategy}"
        if key not in triggered:
            triggered[key] = []
        triggered[key].append(reason)

    # ── Bus/Metro Operation ───────────────────────────────────────────────
    if seat is not None and seat < -0.5:
        add("Bus/Metro Operation", "Increase service frequency",
            f"Transit Supply={seat:+.2f} (below city average): this grid receives less "
            f"transit service than the city norm")
    if seat is not None and seat < -0.5 and pop is not None and pop > 0.5:
        add("Bus/Metro Operation", "Increase service frequency",
            f"Population Intensity={pop:+.2f} (above city average) combined with "
            f"Transit Supply={seat:+.2f} (below average): high demand, insufficient supply")

    if ci is not None and ci > 0.5 and seat is not None and seat < -0.5:
        add("Bus/Metro Operation", "Bus priority infrastructure",
            f"Congestion Level={ci:+.2f} (above city average) delays buses; "
            f"Transit Supply={seat:+.2f} (below average) compounds the problem")
    if "supply shortage" in congestion_types or "severe mismatch" in congestion_types:
        add("Bus/Metro Operation", "Bus priority infrastructure",
            f"Congestion type includes supply shortage / severe mismatch — "
            f"infrastructure bottleneck relative to city norm")

    if has_day_diff or n_time_periods >= 3:
        add("Bus/Metro Operation", "Dynamic bus/metro dispatch",
            f"Congestion spans {n_time_periods} time period(s) "
            + ("with weekday/weekend pattern differences" if has_day_diff else "")
            + " — variable demand requires adaptive scheduling")

    if acc is not None and acc > 0 and seat is not None and seat < -0.5:
        add("Bus/Metro Operation", "Bus–Metro coordination",
            f"Transit Accessibility={acc:+.2f} (above city average, network reach is good) "
            f"but Transit Supply={seat:+.2f} (below average) — actual service does not match "
            f"the network's potential")
    if acc is not None and acc < -0.5:
        add("Bus/Metro Operation", "Bus–Metro coordination",
            f"Transit Accessibility={acc:+.2f} (below city average): this grid is poorly "
            f"connected to the transit network; feeder integration needed to extend reach")

    # ── Walkability and Station Access ────────────────────────────────────
    if walk is not None and walk < -0.5:
        add("Walkability and Station Access", "Pedestrian street improvement",
            f"Local Walkability={walk:+.2f} (below city average): pedestrian environment "
            f"is worse than the city norm")

    if walk is not None and walk < -0.5 and (
        (moto_use is not None and moto_use > 0.5) or (ci is not None and ci > 0.5)
    ):
        extras = []
        if moto_use is not None and moto_use > 0.5:
            extras.append(f"Motorcycle Usage={moto_use:+.2f} (above city average)")
        if ci is not None and ci > 0.5:
            extras.append(f"Congestion Level={ci:+.2f} (above city average)")
        add("Walkability and Station Access", "Safe street crossings",
            f"Local Walkability={walk:+.2f} (below city average) combined with "
            + " and ".join(extras) + " → dangerous crossing conditions above city norm")

    if walk is not None and walk < -0.5 and acc is not None and acc > 0:
        add("Walkability and Station Access", "Station-area pedestrian design",
            f"Transit Accessibility={acc:+.2f} (above city average — network is reachable) "
            f"but Local Walkability={walk:+.2f} (below city average — hard to reach on foot): "
            f"usability gap between network reach and pedestrian access")

    if walk is not None and walk < -0.5 and (lu_pub > 0.1 or lu_edu > 0.1):
        add("Walkability and Station Access", "Accessibility improvements",
            f"Local Walkability={walk:+.2f} (below city average) with "
            f"public-use land use ratio={lu_pub:.3f} and/or education land use ratio={lu_edu:.3f}: "
            f"vulnerable users (elderly, disabled, students) need barrier-free access")

    # ── First/Last Mile Connection ─────────────────────────────────────────
    if moto_dep is not None and moto_dep > 0.5:
        add("First/Last Mile Connection", "Shared electric motorcycles",
            f"Motorcycle Dependence={moto_dep:+.2f} (above city average): motorcycles make up "
            f"a higher modal share than the city norm, indicating structural dependence")

    if moto_dep is not None and moto_dep > 0.5 and acc is not None and acc > 0:
        add("First/Last Mile Connection", "Micro-mobility hubs",
            f"Motorcycle Dependence={moto_dep:+.2f} (above city average) with "
            f"Transit Accessibility={acc:+.2f} (above average — stations exist in the network): "
            f"motorcycle-dependent access to above-average network nodes needs organization")

    if moto_dep is not None and moto_dep > 0.5 and moto_use is not None and moto_use > 0.5:
        add("First/Last Mile Connection", "Electrification of mobility",
            f"Motorcycle Dependence={moto_dep:+.2f} and Motorcycle Usage={moto_use:+.2f} "
            f"(both above city average): high volume and share make electrification viable "
            f"and impactful")

    if moto_dep is not None and moto_dep > 0.5 and (
        (acc is not None and acc < -0.5) or (walk is not None and walk < -0.5)
    ):
        details = []
        if acc is not None and acc < -0.5:
            details.append(f"Transit Accessibility={acc:+.2f} (below city average)")
        if walk is not None and walk < -0.5:
            details.append(f"Local Walkability={walk:+.2f} (below city average)")
        add("First/Last Mile Connection", "Integrated mobility systems",
            f"Motorcycle Dependence={moto_dep:+.2f} (above city average) with "
            + " and ".join(details)
            + " — fragmented modal environment requires integrated coordination")

    # ── Traffic Management ────────────────────────────────────────────────
    if ci is not None and ci > 0.5:
        if lu_res + lu_mix > 0.4:
            add("Traffic Management", "One-way street systems",
                f"Congestion Level={ci:+.2f} (above city average) in a grid with "
                f"residential + mixed-use land use ratio={lu_res+lu_mix:.2f} (dense local streets "
                f"likely contributing to bidirectional conflict)")

    if ci is not None and ci > 0.5 and moto_use is not None and moto_use > 0.5:
        add("Traffic Management", "Motorcycle lane organization",
            f"Congestion Level={ci:+.2f} and Motorcycle Usage={moto_use:+.2f} "
            f"(both above city average): mixed-traffic conflict above city norm")

    if ci is not None and ci > 0.5 and n_time_periods <= 3:
        add("Traffic Management", "Intersection management",
            f"Congestion Level={ci:+.2f} (above city average) concentrated in "
            f"{n_time_periods} time period(s) — suggests localized intersection bottleneck "
            f"rather than network-wide problem")

    if (n_time_periods >= 3 or has_day_diff) and ci is not None and ci > 1.0 and moto_use is not None and moto_use > 1.0:
        add("Traffic Management", "Real-time monitoring & control",
            f"Congestion Level={ci:+.2f} and Motorcycle Usage={moto_use:+.2f} "
            f"(both well above city average) across {n_time_periods} time period(s)"
            + (" with weekday/weekend differences" if has_day_diff else "")
            + " — operationally complex conditions require adaptive monitoring")
    elif n_time_periods >= 3 and has_day_diff:
        add("Traffic Management", "Real-time monitoring & control",
            f"Congestion spans {n_time_periods} time periods with weekday/weekend pattern "
            f"differences — variable conditions above city norm benefit from adaptive control")

    return triggered

def build_shared_mobility_assessment(
    base: Dict[str, Any],
    congestion_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    def get(field: str) -> Optional[float]:
        v = base.get(field)
        return float(v) if v is not None else None

    seat = get("seat_sum_avg_scaled")
    acc = get("acc30_500m_mean_scaled")
    walk = get("walk_mean_500m_scaled")
    moto_dep = get("share_motorcycle_all_scaled")
    moto_use = get("n_motorcycle_scaled")
    ci = get("CI_final_mean_scaled")
    pop = get("pop_est_grid_scaled")

    score = 0
    reasons = []
    trigger_signals = []
    recommended_actions = []

    # 1. Motorcycle Dependence：最重要
    if moto_dep is not None:
        if moto_dep > 1.5:
            score += 30
            reasons.append(f"Motorcycle Dependence is {moto_dep:+.2f}, far above the city average, indicating strong structural reliance on motorcycles.")
            trigger_signals.append("Motorcycle Dependence far above city average")
        elif moto_dep > 0.5:
            score += 20
            reasons.append(f"Motorcycle Dependence is {moto_dep:+.2f}, above the city average, suggesting above-average need for first/last-mile alternatives.")
            trigger_signals.append("Motorcycle Dependence above city average")

    # 2. Transit Supply：公交供给不足时，共享出行更可能有需求
    if seat is not None:
        if seat < -1.0:
            score += 20
            reasons.append(f"Transit Supply is {seat:+.2f}, clearly below the city average, meaning this grid is underserved by public transport.")
            trigger_signals.append("Transit Supply clearly below city average")
        elif seat < -0.5:
            score += 12
            reasons.append(f"Transit Supply is {seat:+.2f}, below the city average, indicating limited transport service provision.")
            trigger_signals.append("Transit Supply below city average")

    # 3. Walkability：步行差，shared mobility更需要
    if walk is not None:
        if walk < -1.0:
            score += 15
            reasons.append(f"Local Walkability is {walk:+.2f}, clearly below the city average, weakening non-motorized access.")
            trigger_signals.append("Local Walkability clearly below city average")
        elif walk < -0.5:
            score += 10
            reasons.append(f"Local Walkability is {walk:+.2f}, below the city average, making local access more difficult.")
            trigger_signals.append("Local Walkability below city average")

    # 4. Transit Accessibility：高 accessibility + 高 motorcycle dependence 很适合 first/last-mile
    if acc is not None:
        if acc > 0.5 and moto_dep is not None and moto_dep > 0.5:
            score += 15
            reasons.append(f"Transit Accessibility is {acc:+.2f}, above the city average, while Motorcycle Dependence is also above average, suggesting strong potential for shared mobility as a feeder to transit.")
            trigger_signals.append("Transit Accessibility above average + Motorcycle Dependence above average")
        elif acc < -0.5 and moto_dep is not None and moto_dep > 0.5:
            score += 8
            reasons.append(f"Transit Accessibility is {acc:+.2f}, below the city average, and Motorcycle Dependence is above average, indicating fragmented mobility conditions where shared mobility may still help local access.")
            trigger_signals.append("Transit Accessibility below average + Motorcycle Dependence above average")

    # 5. Population Intensity：有人才有实际需求
    if pop is not None:
        if pop > 1.0:
            score += 12
            reasons.append(f"Population Intensity is {pop:+.2f}, above the city average, so any mobility gap affects a relatively large number of people.")
            trigger_signals.append("Population Intensity above city average")
        elif pop > 0.3:
            score += 6
            reasons.append(f"Population Intensity is {pop:+.2f}, slightly above the city average, indicating a moderate user base for shared mobility.")
            trigger_signals.append("Population Intensity slightly above city average")

    # 6. Motorcycle Usage：大量摩托存在时，共享替代更有现实基础
    if moto_use is not None:
        if moto_use > 1.0:
            score += 10
            reasons.append(f"Motorcycle Usage is {moto_use:+.2f}, well above the city average, so structured shared mobility could help organize and partially replace excessive motorcycle activity.")
            trigger_signals.append("Motorcycle Usage well above city average")
        elif moto_use > 0.5:
            score += 6
            reasons.append(f"Motorcycle Usage is {moto_use:+.2f}, above the city average, indicating strong two-wheeler presence.")
            trigger_signals.append("Motorcycle Usage above city average")

    # 7. Congestion：有助于判断是否需要组织化接驳
    if ci is not None and ci > 0.5:
        score += 5
        reasons.append(f"Congestion Level is {ci:+.2f}, above the city average, which strengthens the case for structured first/last-mile solutions.")
        trigger_signals.append("Congestion Level above city average")

    # 8. 基于事件数量的小修正
    if len(congestion_events) >= 3:
        score += 3
        trigger_signals.append("Congestion occurs across multiple time periods")

    # 上限
    score = min(score, 100)

    # 分类
    if score >= 70:
        level = "High"
        need_shared_mobility = True
    elif score >= 40:
        level = "Medium"
        need_shared_mobility = True
    else:
        level = "Low"
        need_shared_mobility = False

    # primary role
    if need_shared_mobility:
        if moto_dep is not None and moto_dep > 0.5 and acc is not None and acc > 0.5:
            primary_role = "First/last-mile feeder to transit"
        elif walk is not None and walk < -0.5:
            primary_role = "Local access support"
        elif moto_dep is not None and moto_dep > 0.5 and moto_use is not None and moto_use > 0.5:
            primary_role = "Motorcycle replacement and management"
        else:
            primary_role = "Supplementary local mobility"
    else:
        primary_role = "Not a current priority"

    # recommended actions
    if level == "High":
        recommended_actions = [
            "Shared electric motorcycles",
            "Micro-mobility hubs",
            "Integrated mobility systems"
        ]
    elif level == "Medium":
        recommended_actions = [
            "Shared electric motorcycles",
            "Micro-mobility hubs"
        ]
    else:
        recommended_actions = []

    # 简要结论
    if need_shared_mobility:
        summary = f"This grid has a {level.lower()} need for shared mobility."
    else:
        summary = "Shared mobility is not a current priority for this grid."

    return {
        "score": score,
        "level": level,
        "need_shared_mobility": need_shared_mobility,
        "primary_role": primary_role,
        "summary": summary,
        "reasons": reasons,
        "recommended_actions": recommended_actions,
        "trigger_signals": trigger_signals,
    }

def build_grid_profile(grid_id: str) -> Dict[str, Any]:
    if grid_id not in LANDUSE_BY_ID:
        raise HTTPException(status_code=404, detail=f"Grid id {grid_id} not found.")

    base   = LANDUSE_BY_ID[grid_id].copy()
    events = CONGESTION_BY_ID.get(grid_id, [])

    weekday_events = [e for e in events if str(e["day_type"]).lower() == "weekday"]
    weekend_events = [e for e in events if str(e["day_type"]).lower() == "weekend"]

    shared_mobility_assessment = build_shared_mobility_assessment(base, events)

    return {
        **base,
        **summarize_landuse_mix(base),
        "zscore_profile": build_zscore_profile(base),
        "triggered_strategy_signals": derive_triggered_strategies(base, events),
        "shared_mobility_assessment": shared_mobility_assessment,
        "congestion_events": events,
        "weekday_congestion_events": weekday_events,
        "weekend_congestion_events": weekend_events,
    }


# =========================================================
# 6. Diagnosis Prompt 构建
# =========================================================
def build_diagnosis_prompt(grid_profile: Dict[str, Any]) -> str:
    zscore_block    = json.dumps(grid_profile.get("zscore_profile", {}),             ensure_ascii=False, indent=2)
    triggered_block = json.dumps(grid_profile.get("triggered_strategy_signals", {}), ensure_ascii=False, indent=2)

    return f"""
You are an urban mobility strategy assistant for Ho Chi Minh City.
Your task is to diagnose one grid cell and recommend strategies only from the provided strategy library.

====================
{INDICATOR_DEFINITIONS}

====================
CLUSTER DEFINITIONS
====================
{json.dumps(CLUSTER_DEFINITIONS, ensure_ascii=False, indent=2)}

====================
CONGESTION DEFINITIONS
====================
{json.dumps(CONGESTION_DEFINITIONS, ensure_ascii=False, indent=2)}

====================
STRATEGY LIBRARY
====================
{json.dumps(STRATEGY_LIBRARY, ensure_ascii=False, indent=2)}

====================
{INDICATOR_TO_STRATEGY_MAPPING}

====================
GRID PROFILE
====================
{json.dumps(grid_profile, ensure_ascii=False, indent=2)}

====================
Z-SCORE INDICATOR PROFILE
(all values are relative to the city-wide average across all Ho Chi Minh City grids)
====================
{zscore_block}

Z-score reference scale (city-wide):
  > +2.5        : far above city average
  +1.0 to +2.5  : above city average
  +0.3 to +1.0  : slightly above city average
  -0.3 to +0.3  : around city average
  -1.0 to -0.3  : slightly below city average
  -2.5 to -1.0  : below city average
  < -2.5        : far below city average
  null / N/A    : this grid had missing data for this indicator and was excluded
                  from its city-wide normalization; do not infer a value from null.

====================
PRE-COMPUTED STRATEGY TRIGGER SIGNALS
(derived from z-score values and congestion events; all descriptions use human-readable names)
====================
{triggered_block}

====================
TASK
====================
1. Diagnose the mobility issue of this grid using:
   - the cluster meaning and planning context
   - the z-score profile (all values are relative to the city-wide average; reference them explicitly)
   - the full land use composition (ratios, not just dominant category)
   - congestion events, types, and temporal patterns
   - the pre-computed trigger signals above as your primary evidence base

2. Select strategies ONLY from the strategy library.
   - Use the trigger signals as your primary guide.
   - Recommend between 3 and 5 strategies total.
   - You are NOT required to select from every family — only what the data supports.
   - If multiple triggers point to the same family, you may select more than one sub-strategy.
   - If no triggers point to a family, do not force a selection from it.
   - For each strategy, state which z-score value or congestion signal justifies it.

3. If weekday and weekend patterns differ, explain them separately.

4. If a z-score is null/missing, note it briefly and do not infer a value from it.
5. Do NOT independently invent a different shared mobility priority logic.
   Instead, use the provided "shared_mobility_assessment" in the grid profile as the source of truth.
   You may explain it in natural language, but do not contradict its level, score, primary role,
   or recommended actions.
====================
OUTPUT FORMAT
====================
Return valid JSON only — no markdown, no text outside the JSON:
{{
  "diagnosis": "2-4 sentences. Reference specific z-score values relative to city average. Use human-readable indicator names only.",
  "weekday_pattern": "1-3 sentences.",
  "weekend_pattern": "1-3 sentences.",
  "problem_logic": "1-3 sentences explaining the mobility problem with z-score evidence relative to city average.",
  "recommended_strategies": [
    {{
      "family": "exact family name from strategy library",
      "strategy": "exact strategy name from strategy library",
      "reason": "specific reason referencing z-score value relative to city average, using human-readable indicator names only",
      "priority": "short-term or medium-term"
    }}
  ],
  "summary": "1 concise sentence.",
  "shared_mobility": {{
    "ai_interpretation": "2-3 sentences explaining the already-computed shared mobility assessment in plain language, without contradicting the provided level/score/role.",
  "implementation_note": "1 concise sentence on whether immediate deployment is justified."
  }}
}}

Critical output rules:
- demand_score must be an integer from 0 to 1.
- shared_mobility.reasoning must explicitly explain why this grid does or does not need shared mobility.
- If demand is low, recommended_actions can be an empty list.
- NEVER use raw variable names.
- Always use human-readable indicator names.
- Always frame z-score values relative to the city average.
- Use exact family and strategy names from the strategy library.
- Output JSON only.
"""


# =========================================================
# 7. Image Prompt 构建
# =========================================================
def build_image_prompt_prompt(request_data: Dict[str, Any]) -> str:
    return f"""
You are an urban design visualization assistant.

Your task is NOT to diagnose the grid again.
The diagnosis and recommended strategies have already been produced.

Your job is to generate a visualization-oriented image prompt for a generative image model,
based on the selected grid diagnosis, the already-selected strategies, and one user-selected
street-view image.

====================
SELECTED GRID AND DESIGN CONTEXT
====================
{json.dumps(request_data, ensure_ascii=False, indent=2)}

====================
TASK
====================
1. Read the provided diagnosis and recommended strategies.
2. Do NOT re-analyze land use composition or z-scores.
3. Select all visually representable strategies from the recommended strategies.
4. Convert them into a street-level design visualization concept for the selected street image.
5. Keep the prompt grounded in realistic urban street improvement, not fantasy design.

====================
VISUALIZATION RULES
====================
- Preserve the original street-view perspective and camera viewpoint.
- Keep street geometry and building massing consistent unless a strategy implies redesign.
- Focus on visible spatial changes: sidewalks, crossings, bus lanes, bus stop access,
  motorcycle lane organization, street trees, curb management, feeder mobility hubs.
- Avoid abstract operational strategies unless representable through visible street elements.
- The output should read like a prompt for a realistic "after improvement" urban street scene.
- Do not mention land use ratios, z-score values, or raw variable names.
- Do not repeat the diagnosis in narrative form.

====================
OUTPUT FORMAT
====================
Return valid JSON only — no markdown:
{{
  "selected_visual_strategies": ["strategy 1", "strategy 2"],
  "scene_summary": "1-2 sentences summarizing the intended improved scene",
  "design_changes": ["visible change 1", "visible change 2", "visible change 3"],
  "image_generation_prompt": "a single detailed prompt for an image generation model",
  "negative_prompt": "things to avoid in the generated image"
}}
"""


# =========================================================
# 8. Gemini 调用
# =========================================================
def _parse_gemini_json_response(response_text: str) -> Dict[str, Any]:
    text = (response_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)

        # Only inject shared_mobility defaults for diagnosis responses
        # P06 / image-prompt responses have different keys and must not be polluted
        if "diagnosis" in parsed or "recommended_strategies" in parsed:
            if "shared_mobility" not in parsed or not isinstance(parsed["shared_mobility"], dict):
                parsed["shared_mobility"] = {
                    "ai_interpretation": "No AI interpretation was returned for shared mobility.",
                    "implementation_note": ""
                }
            else:
                parsed["shared_mobility"].setdefault("ai_interpretation",
                                                     "No AI interpretation was returned for shared mobility.")
                parsed["shared_mobility"].setdefault("implementation_note", "")

        return parsed
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail={"message": "Gemini did not return valid JSON.", "raw_output": text},
        )


def call_gemini_for_diagnosis(grid_profile: Dict[str, Any]) -> Dict[str, Any]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_diagnosis_prompt(grid_profile),
    )
    return _parse_gemini_json_response(response.text or "")


def call_gemini_for_image_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_image_prompt_prompt(payload),
    )
    return _parse_gemini_json_response(response.text or "")


# =========================================================
# 9. API 路由
# =========================================================
@app.get("/")
def root():
    return {"message": "HCMC Grid Diagnosis API is running."}


@app.get("/api/grid/{grid_id}")
def get_grid_profile_endpoint(grid_id: str):
    return build_grid_profile(grid_id)


@app.get("/api/diagnosis/{grid_id}")
def get_grid_diagnosis(grid_id: str):
    cached = get_cached_diagnosis(grid_id)

    if cached and cached["model_name"] == MODEL_NAME and cached["prompt_version"] == PROMPT_VERSION:
        return {
            "source": "database",
            "grid_profile": cached["grid_profile"],
            "ai_result": cached["ai_result"],
            "model_name": cached["model_name"],
            "prompt_version": cached["prompt_version"],
        }

    grid_profile = build_grid_profile(grid_id)
    ai_result = call_gemini_for_diagnosis(grid_profile)

    save_cached_diagnosis(
        grid_id=grid_id,
        grid_profile=grid_profile,
        ai_result=ai_result
    )

    return {
        "source": "generated" if not cached else "regenerated",
        "grid_profile": grid_profile,
        "ai_result": ai_result,
        "model_name": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
    }

@app.post("/api/diagnosis/{grid_id}/reanalyze")
def reanalyze_grid_diagnosis(grid_id: str):
    grid_profile = build_grid_profile(grid_id)
    ai_result = call_gemini_for_diagnosis(grid_profile)

    save_cached_diagnosis(
        grid_id=grid_id,
        grid_profile=grid_profile,
        ai_result=ai_result
    )

    return {
        "source": "reanalysed",
        "grid_profile": grid_profile,
        "ai_result": ai_result,
        "model_name": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
    }

@app.post("/api/image-prompt")
def generate_image_prompt(request: ImagePromptRequest):
    payload      = request.model_dump()
    grid_profile = build_grid_profile(request.grid_id)

    merged_payload = {
        "grid_id":                request.grid_id,
        "cluster_name":           request.cluster_name,
        "dominant_lu":            request.dominant_lu,
        "diagnosis":              request.diagnosis,
        "problem_logic":          request.problem_logic,
        "recommended_strategies": payload["recommended_strategies"],
        "street_image":           request.street_image,
        "grid_context_for_reference": {
            "cluster_name":           grid_profile.get("cluster_name"),
            "dominant_lu_name":       grid_profile.get("dominant_lu_name"),
            "top_landuse_components": grid_profile.get("top_landuse_components", []),
            "zscore_profile":         grid_profile.get("zscore_profile", {}),
        },
    }

    ai_result = call_gemini_for_image_prompt(merged_payload)
    return {"input": merged_payload, "ai_result": ai_result}


def download_image(image_url: str) -> bytes:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {str(e)}")
def analyze_image_with_gemini(image_bytes: bytes) -> Dict[str, Any]:
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": """
You are an urban street analysis expert.

Analyze this street image and extract:

1. Road structure (lanes, width, organization)
2. Pedestrian condition (sidewalk, crossings, safety)
3. Traffic composition (cars, motorcycles, buses)
4. Street environment (trees, shading, density)
5. Key problems visible in the scene

Return JSON only, no markdown:
{
  "road_structure": "",
  "pedestrian_condition": "",
  "traffic_composition": "",
  "environment": "",
  "problems": ["", ""]
}
"""},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }
        ]
    )
    return _parse_gemini_json_response(response.text or "")

def build_p06_prompt(data: Dict[str, Any]) -> str:
    return f"""
You are an urban design visualization expert writing an image editing prompt for Ho Chi Minh City street improvement.

====================
CURRENT STREET CONDITIONS
====================
{json.dumps(data["image_analysis"], indent=2)}

====================
GRID MOBILITY DIAGNOSIS
====================
{data["diagnosis"]}

====================
RECOMMENDED STRATEGIES TO VISUALIZE
====================
{json.dumps(data["recommended_strategies"], indent=2)}

====================
YOUR TASK
====================
Write a structured image editing prompt using the EXACT format below.
The prompt must be driven strictly by the recommended strategies above.
ONLY include sections that are directly justified by the strategies.
Do NOT add bus stops, metro entrances, bike lanes, or any element unless strategies explicitly call for it.

====================
MANDATORY RULES
====================
RULE 1 CAMERA LOCK: identical camera angle, position, height, focal length, direction as original. Never change viewpoint.
RULE 2 MINIMAL INTERVENTION: 60% of scene unchanged. Only modify what strategies require.
RULE 3 LIGHTING LOCK: same time of day, shadow direction, sky colour, light temperature as Before photo.
RULE 4 HUMAN SCALE: pedestrian height 1.6m as reference. Bus shelter 3m, display pole 2.5m, scooter dock 1.1m, metro canopy 3.2m.
RULE 5 ROAD PRESERVATION: no vehicle lanes converted to pedestrian space. Road changes are paint and markings only within existing width.
RULE 6 BEHAVIORAL COHERENCE: show people and vehicles actively using each improvement. All motorcycle riders wear helmets.
RULE 7 VIETNAMESE SPECIFICITY: Vietnamese signage (Tram xe buyt, route numbers), white lane lines, yellow centre line, high motorbike density.
RULE 8 FOREGROUND CLEAR: new elements must not block more than 15% of image width. Key improvement zone must be clearly visible.
RULE 9 PHOTOREALISM: real photograph look, real materials with wear, no CGI, no renders, no concept art.
RULE 10 CLEAN ENVIRONMENT: new infrastructure newly installed and well-maintained. No rust, cracks, litter, or vandalism.

====================
PROMPT FORMAT
====================
Write the final_prompt using this structured section format.
Only include sections justified by the strategies. Do not invent sections.

START WITH (always):
Transform this street photograph into a highly realistic improved urban scene in Ho Chi Minh City,
maintaining the exact same camera angle, position, and lighting as the original photo.

REMOVE (always include):
- all floating UI icons, overlay symbols, and interface elements
- any unrealistic graphic indicators or map markers

THEN add ONLY the sections justified by the recommended strategies, using the exact mapping below.
Match each recommended strategy to its corresponding section template:

─── Bus/Metro Operation ───

[IF recommended_strategies includes "Increase service frequency"]:
IMPROVE BUS FREQUENCY:
- one or two buses visible approaching or stopped at the bus stop in the scene
- bus stop has a small digital real-time arrival display (LCD screen on ~2.5m pole)
- Vietnamese signage: "Trạm xe buýt" with route numbers visible on shelter
- 3–5 people waiting naturally at the stop, one person boarding

[IF recommended_strategies includes "Bus priority infrastructure"]:
BUS PRIORITY LANE:
- one existing lane repainted as dedicated bus lane: red surface with white "XE BUÝT" text
- white lane boundary line separating bus lane from mixed traffic (Vietnamese standard)
- a bus moving in the dedicated lane, other vehicles in adjacent lanes
- helmeted motorcyclists respecting the lane boundary

[IF recommended_strategies includes "Bus–Metro coordination"]:
BUS–METRO INTERCHANGE SIGNAGE:
- bilingual wayfinding sign connecting bus stop to metro: "Metro / Ga tàu điện ngầm" in HCMC green accent colour
- painted ground arrows guiding passengers between bus stop and metro direction
- one person transferring naturally, looking at their phone for directions

[IF recommended_strategies includes "Dynamic bus/metro dispatch"]:
SMART DISPATCH DISPLAY:
- small digital information panel at bus stop showing real-time bus positions (LCD, not holographic)
- one discreet IoT sensor box mounted on shelter pole (~15cm box, industrial design)
- one person glancing at the display, another checking phone

─── First/Last Mile Connection ───

[IF recommended_strategies includes "Micro-mobility hubs"]:
MICRO-MOBILITY HUB:
- compact docking rack at kerb: 4–6 electric scooters parked neatly (~1.1m rack height)
- ground markings: yellow dashed box with "KHU ĐỖ XE ĐIỆN" text
- one person scanning QR code to unlock a scooter, another docking one
- battery swap cabinet nearby (~1.2m high, powder-coated steel)

[IF recommended_strategies includes "Shared electric motorcycles"]:
SHARED ELECTRIC MOTORCYCLE ZONE:
- 3–4 shared electric motorcycles parked in a clearly marked kerb bay
- subtle charging indicator lights on handlebars
- painted ground zone with "XE MÁY ĐIỆN DÙNG CHUNG" text
- one helmeted rider mounting a shared motorcycle

[IF recommended_strategies includes "Integrated mobility systems"]:
INTEGRATED MOBILITY SIGNAGE:
- unified wayfinding pole (~2.5m) showing walking times to bus stop, metro, and bike share
- QR code panel for app-based journey planning
- clean aluminium panel design, no neon or animation

[IF recommended_strategies includes "Electrification of mobility"]:
EV CHARGING POINT:
- one compact EV charging post at kerb (~1.3m high, similar to Singapore/Hanoi style)
- one electric scooter or motorcycle plugged in
- subtle ground marking indicating the charging bay

─── Traffic Management ───

[IF recommended_strategies includes "Motorcycle lane organization"]:
MOTORCYCLE LANE ORGANISATION:
- painted lane boundary separating motorcycle lane from car lane within existing road width
- white dashed line with "XE MÁY" text at intervals on road surface
- cluster of helmeted motorcyclists riding within the designated lane
- cars in adjacent lane, clear orderly separation visible

[IF recommended_strategies includes "Intersection management"]:
INTERSECTION IMPROVEMENTS:
- clearer stop lines and junction box markings painted on road surface
- directional arrows for each lane painted in white (Vietnamese standard)
- zebra crossing with freshly painted white stripes, refuge island if road is wide enough
- one pedestrian mid-crossing, vehicles stopped at stop line

[IF recommended_strategies includes "One-way street systems"]:
ONE-WAY TRAFFIC ORGANISATION:
- one-way signs (white arrow on blue background, Vietnamese standard) at entry point
- lane arrows all pointing same direction painted on road surface
- clear, orderly traffic flow in one direction, parked motorcycles on one side only

[IF recommended_strategies includes "Real-time monitoring & control"]:
SMART TRAFFIC MONITORING:
- one small traffic camera or sensor on existing signal pole (compact ~20cm box)
- small variable message sign (~60×40cm LED board) showing speed or congestion status
- subtle, no large gantry structures — integrated into existing infrastructure

─── Walkability and Station Access ───

[IF recommended_strategies includes "Pedestrian street improvement"]:
IMPROVED SIDEWALK:
- sidewalk repaved with clean interlocking tiles or smooth concrete (max ~1m wider using kerb space only)
- 2–3 mature tropical trees in tree pits with grating (~6m tall, natural canopy)
- pedestrians walking comfortably, some under tree shade
- continuous clear path with no motorcycles parked on sidewalk

[IF recommended_strategies includes "Safe street crossings"]:
SAFE STREET CROSSING:
- freshly painted zebra crossing with white stripes (Vietnamese standard)
- pedestrian signal pole with countdown timer (~2.5m, compact head)
- one pedestrian mid-crossing, 2–3 vehicles stopped and waiting at stop line
- tactile paving strip at kerb ramp

[IF recommended_strategies includes "Accessibility improvements"]:
ACCESSIBILITY IMPROVEMENTS:
- kerb ramps at crossing points (flush concrete, Vietnamese standard)
- yellow tactile paving strips along key pedestrian routes
- one person with a bicycle or elderly person using the accessible path naturally

[IF recommended_strategies includes "Station-area pedestrian design"]:
STATION AREA PEDESTRIAN ZONE:
- clear pedestrian priority markings around bus stop or metro entrance
- wayfinding signs in Vietnamese showing walking routes to nearby destinations
- one bench (~0.9m high) and a waste bin with lid as street furniture
- small group of pedestrians navigating the improved space naturally

STYLE (always include):
- photorealistic, indistinguishable from a real street photograph
- same lighting as original: match the Before photo time of day, shadow direction, sky
- real materials with natural wear: asphalt, concrete, metal, glass, slight imperfections
- authentic Ho Chi Minh City street energy: high motorbike density, natural pedestrian activity
- documentary street photography, eye-level perspective, no bird's eye view
- all motorcycle and scooter riders wear helmets

====================
OUTPUT — return valid JSON only, no markdown:
====================
The "final_prompt" value MUST follow this exact paragraph format with section headers and bullet points.
Each section header is on its own line in ALL CAPS followed by a colon.
Each bullet point starts with "- ".
Only include improvement sections justified by the recommended strategies.
The RULES, REMOVE, STYLE, and CAMERA sections are ALWAYS required.

{{
  "final_prompt": "Transform this street photograph into a highly realistic improved urban scene in Ho Chi Minh City. [1-2 sentences describing the specific street from the Before image: road width, number of lanes, key buildings, time of day, lighting conditions.]\\n\\nRULES (non-negotiable):\\n- CAMERA LOCK: maintain the exact same camera angle, position, height, focal length, and direction as the original photo — do not change the viewpoint under any circumstances\\n- LIGHTING LOCK: identical time of day, shadow direction, shadow length, sky colour, and light temperature as the Before photo — do not brighten or change lighting angle\\n- MINIMAL INTERVENTION: 60% of the scene must remain unchanged — only modify what the strategies require\\n- SCALE: use pedestrian height ~1.6m as reference for all new elements\\n- ROAD PRESERVATION: do not convert vehicle lanes into pedestrian space — road improvements are paint and markings only within existing width\\n- PHOTOREALISM: result must be indistinguishable from a real street photograph — no CGI, no renders, no concept art\\n- CLEAN: all new infrastructure is newly installed and well-maintained — no rust, cracks, litter, or vandalism\\n- VIETNAMESE: all signage in Vietnamese, white lane lines, yellow centre line, all riders wear helmets, high motorbike density\\n\\nREMOVE:\\n- all floating UI icons, overlay symbols, and map markers\\n- any unrealistic graphic indicators or interface elements\\n\\n[strategy-justified improvement sections here, one per recommended strategy]\\n\\nSTYLE:\\n- photorealistic, indistinguishable from a real street photograph\\n- identical lighting to original: [time of day], shadows falling [direction], [sky colour and cloud description]\\n- real materials with natural wear: asphalt, concrete, painted markings, metal, glass, slight imperfections\\n- authentic Ho Chi Minh City street energy: high motorbike density, natural pedestrian activity, mid-action poses\\n- documentary street photography, eye-level perspective\\n- all motorcycle and scooter riders wear helmets\\n\\nCAMERA:\\n- eye-level street perspective, identical to original photo\\n- same focal length, same horizontal angle, same vertical tilt\\n- no change to viewpoint, zoom, or framing",
  "negative_prompt": "3D render, CGI, architectural visualization, concept art, illustration, cartoon, painting, anime, futuristic design, sci-fi, fantasy, floating icons, UI elements, map markers, arrows, overlays, bird's eye view, aerial view, isometric view, drone shot, changed camera angle, different lighting direction, changed time of day, zoomed in, zoomed out, different street, neon lights, cyberpunk, studio lighting, white background, solar panels underneath structures, unrealistic scale, oversized elements, holographic displays, litter, garbage, dirty sidewalks, rusty equipment, cracked screens, broken infrastructure, vandalism, graffiti, aged devices, empty street, no people, deserted, staged poses, helmets-free riders, English-only signs, foreground blocking improvements, wholesale street redesign, removed vehicle lanes, pedestrianized carriageway, unsolicited bus stop, unsolicited metro entrance, unsolicited bike lane"
}}
"""





def generate_image_from_prompt(prompt: str, negative_prompt: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    使用 Nano Banana 2 生图。
    image_bytes 存在时：图生图模式（模型看到 Before 图 → 在原图基础上编辑 → 视角天然一致）
    image_bytes 为空时：纯文生图（降级）
    """
    HARD_PREFIX = (
        "You are editing this street photograph to show a realistic smart city urban mobility improvement in Ho Chi Minh City. "
        "CRITICAL RULE #1 — CAMERA LOCK: preserve the exact camera angle, position, "
        "height, focal length, and facing direction of the original photo. Do not change the viewpoint. "
        "CRITICAL RULE #2 — PHOTOREALISM: result must look like a real photograph, not a render or illustration. "
        "Documentary street photography. Real materials with natural wear. Smart city elements look like "
        "they exist TODAY in real Asian cities — realistic industrial design, no sci-fi. "
        "CRITICAL RULE #3 — CLEAN AND WELL-MAINTAINED: clean road surface, no litter or debris. "
        "All new infrastructure is newly installed: fresh paint, intact screens, no rust or cracks. "
        "CRITICAL RULE #4 — BEHAVIORAL COHERENCE: people and vehicles respond logically to improvements. "
        "Bus stop added → people waiting. Bike lane → cyclists using it. Metro entrance → people entering. "
        "Scene is lived-in and activated, never empty or staged. Authentic HCMC urban density. "
        "CRITICAL RULE #5 — SCALE CONSISTENCY: use pedestrian height (~1.6m) as scale reference. "
        "Bus shelter ~3m, display poles ~2.5m, scooter dock ~1.1m, metro canopy ~3.2m. "
        "No element taller than nearby buildings. Everything correctly sized relative to visible people. "
        "CRITICAL RULE #6 — LIGHTING CONSISTENCY: exactly match the Before photo lighting — "
        "same time of day, same shadow direction and length, same sky colour, same light temperature. "
        "Do not brighten or change the lighting angle. New elements cast shadows matching existing light. "
        "CRITICAL RULE #7 — VIETNAMESE LOCAL SPECIFICITY: bus stop signs say 'Trạm xe buýt', "
        "road markings use white lane lines and yellow centre line per Vietnamese traffic law, "
        "motorcycle density is high, ALL riders wear helmets (Vietnamese law, no exceptions), "
        "Vietnamese shop signage and street character preserved. "
        "Metro entrance (if applicable): HCMC Metro green design, bilingual 'Metro' signage, underground stairs with canopy. "
        "CRITICAL RULE #8 — FOREGROUND OCCLUSION CONTROL: new elements must not block more than 15% "
        "of image width in foreground. The key improvement zone must be clearly visible for comparison. "
        "CRITICAL RULE #9 — MINIMAL INTERVENTION: 60% of the scene must be unchanged from Before. "
        "Only change what the strategies require. Do not redesign the whole street. "
        "CRITICAL RULE #10 — ROAD SPACE PRESERVATION: do not convert vehicle lanes into pedestrian space. "
        "Road improvements are optimization within existing width: bus lane paint, motorcycle lane, bus bay. "
    )
    HARD_SUFFIX = (
        " Identical perspective to the input photo. "
        "Natural light. Real camera lens. Photorealistic. "
        "Clean, well-maintained, litter-free street environment."
    )
    if negative_prompt:
        HARD_SUFFIX += (
            f" Avoid: {negative_prompt}, "
            "litter, garbage, trash on ground, dirty sidewalks, rusty equipment, "
            "cracked screens, broken devices, peeling paint, water damage on equipment, "
            "vandalism, graffiti, puddles of dirty water, debris, street clutter, "
            "aged or deteriorated smart devices, malfunctioning displays."
        )
    else:
        HARD_SUFFIX += (
            " Avoid: litter, garbage, trash on ground, dirty sidewalks, rusty equipment, "
            "cracked screens, broken devices, peeling paint, water damage on equipment, "
            "vandalism, graffiti, puddles of dirty water, debris, aged or deteriorated smart devices."
        )

    full_prompt = HARD_PREFIX + prompt + HARD_SUFFIX

    try:
        if image_bytes:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": full_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    ]
                }
            ]
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=contents,
            )
        else:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=full_prompt,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation model error: {str(e)}")

    for candidate in (response.candidates or []):
        for part in (candidate.content.parts if candidate.content else []):
            if hasattr(part, "inline_data") and part.inline_data:
                raw_data = part.inline_data.data
                mime = part.inline_data.mime_type or "image/png"
                if isinstance(raw_data, bytes):
                    raw_data = base64.b64encode(raw_data).decode("utf-8")
                return {"image_base64": raw_data, "mime_type": mime}

    raw = response.text if hasattr(response, "text") else str(response)
    raise HTTPException(
        status_code=502,
        detail=f"Image generation returned no image data. Response: {raw[:500]}"
    )

# =========================================================
# Mapillary 图片代理
# =========================================================
MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "MLY|25832260446429053|f248da43d32da93ed9936e7b0c2a9535")

# Mapillary v4 API 官方端点（Mapillary 2020 年被 Meta 收购后迁移至此）
# 文档: https://www.mapillary.com/developer/api-documentation
_MLY_API_BASE = "https://graph.facebook.com/v4"
# 缩略图字段优先级：最大 → 最小
_MLY_THUMB_FIELDS = "thumb_2048_url,thumb_1024_url,thumb_original_url"

def _fetch_mapillary_image_bytes(image_id: str) -> bytes:
    """
    Download a Mapillary image by constructing the CDN URL directly.

    Mapillary's thumbnail URLs follow a predictable pattern and do not
    require any API call or token — the image_id is sufficient.
    This completely avoids the Graph API token/permission issues.
    """
    # Try CDN URLs in order of preference (largest → smallest)
    candidate_urls = [
        f"https://images.mapillary.com/{image_id}/thumb-2048.jpg",
        f"https://images.mapillary.com/{image_id}/thumb-1024.jpg",
        f"https://images.mapillary.com/{image_id}/thumb-320.jpg",
    ]

    last_err = None
    for url in candidate_urls:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "HCMC-Urban-Tool/1.0"})
            if resp.status_code == 200:
                return resp.content
            last_err = f"{url} → HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)

    raise HTTPException(
        status_code=502,
        detail=f"Could not fetch Mapillary image {image_id}. Last error: {last_err}",
    )


@app.get("/api/mapillary-debug/{image_id}")
def mapillary_debug(image_id: str):
    """
    Debug route: 测试 Mapillary CDN URL 是否可访问。
    确认正常后可删除此路由。
    """
    results = {}
    urls = [
        f"https://images.mapillary.com/{image_id}/thumb-2048.jpg",
        f"https://images.mapillary.com/{image_id}/thumb-1024.jpg",
        f"https://images.mapillary.com/{image_id}/thumb-320.jpg",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "HCMC-Urban-Tool/1.0"})
            results[url] = {"status": resp.status_code, "content_type": resp.headers.get("Content-Type"), "bytes": len(resp.content)}
        except Exception as e:
            results[url] = {"error": str(e)}
    return results


@app.get("/api/mapillary-image/{image_id}")
def proxy_mapillary_image(image_id: str):
    """
    处理层专用：通过后端代理下载 Mapillary 图片字节并转发给前端。
    前端预览层使用 Mapillary JS Viewer，不走这里。
    只在用户点击 Analyse Scene 后才会被调用。
    """
    try:
        image_bytes = _fetch_mapillary_image_bytes(image_id)
        return Response(content=image_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mapillary proxy failed: {str(e)}")


# =========================================================
# P06 三步 AI 图像生成流水线 — Pydantic Models
# =========================================================
class P06SceneAnalysisRequest(BaseModel):
    grid_id: str
    image_id: str
    image_base64: str = Field(..., description="Base64-encoded JPEG from the browser (no data: prefix)")
    diagnosis: str
    recommended_strategies: List[RecommendedStrategyModel] = Field(default_factory=list)


class P06BuildPromptRequest(BaseModel):
    grid_id: str
    image_id: str
    scene_understanding: Union[str, List[str]]
    design_changes: Union[str, List[str]]
    diagnosis: str
    recommended_strategies: List[RecommendedStrategyModel] = Field(default_factory=list)


class P06GenerateImageRequest(BaseModel):
    grid_id: str
    image_id: str
    full_prompt: str
    negative_prompt: str = Field(default="", description="Things to avoid in the generated image")
    image_base64: str = Field(default="", description="Base64 of the Before street photo for image-to-image editing")


# =========================================================
# P06 Step 1 — 场景分析
# =========================================================
@app.post("/api/p06/scene-analysis")
def p06_scene_analysis(request: P06SceneAnalysisRequest):
    """
    处理层 Step 1：接收前端传来的 base64 图片 → Gemini 分析街景。
    图片由浏览器通过 Mapillary JS SDK 获取并编码，后端不做任何网络请求。
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    image_analysis = analyze_image_with_gemini(image_bytes)

    # Build scene understanding prompt
    scene_prompt = f"""
You are an urban street design analyst.

Below is a structured image analysis of a street scene in Ho Chi Minh City:
{json.dumps(image_analysis, indent=2)}

The grid diagnosis is:
{request.diagnosis}

Based on the image analysis AND the diagnosis, write:
1. scene_understanding: A concise 2-3 sentence description of what the street currently looks like and its key mobility problems.
2. design_changes: A list of 3-5 specific, visible physical changes that should be made, based on the recommended strategies below.

Recommended strategies:
{json.dumps([s.model_dump() for s in request.recommended_strategies], indent=2)}

Return valid JSON only:
{{
  "scene_understanding": "...",
  "design_changes": "...",
  "image_analysis": {{...}}
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=scene_prompt,
    )
    result = _parse_gemini_json_response(response.text or "")
    result["image_analysis"] = image_analysis
    return result


# =========================================================
# P06 Step 2 — 生成完整 Prompt
# =========================================================
@app.post("/api/p06/build-prompt")
def p06_build_prompt(request: P06BuildPromptRequest):
    """
    处理层 Step 2：融合场景理解 + 诊断 + 策略，生成 full_prompt。
    不再接触图片字节，只做文本推理。
    """
    # Normalize: Gemini sometimes returns these as lists instead of strings
    def to_str(v):
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return str(v) if v else ""

    prompt = build_p06_prompt({
        "image_analysis": {
            "scene_understanding": to_str(request.scene_understanding),
            "design_changes": to_str(request.design_changes),
        },
        "diagnosis": request.diagnosis,
        "recommended_strategies": [s.model_dump() for s in request.recommended_strategies],
    })
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    result = _parse_gemini_json_response(response.text or "")
    if "final_prompt" in result and "full_prompt" not in result:
        result["full_prompt"] = result["final_prompt"]
    return result


# =========================================================
# P06 Step 3 — 生成图片
# =========================================================
@app.post("/api/p06/generate-image")
def p06_generate_image_v2(request: P06GenerateImageRequest):
    """
    处理层 Step 3：图生图模式——把 Before 图片 + prompt 一起发给 Nano Banana 2。
    模型在原图基础上编辑，视角天然与 Before 一致。
    """
    # Decode Before image for image-to-image mode
    image_bytes: Optional[bytes] = None
    if request.image_base64:
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            image_bytes = None  # degrade gracefully to text-to-image

    image_result = generate_image_from_prompt(
        request.full_prompt,
        request.negative_prompt,
        image_bytes=image_bytes,
    )

    raw = image_result.get("image_base64", b"")
    mime = image_result.get("mime_type", "image/png")

    if isinstance(raw, bytes):
        b64_str = base64.b64encode(raw).decode("utf-8")
    else:
        b64_str = raw

    data_url = f"data:{mime};base64,{b64_str}" if b64_str else ""
    return {
        "generated_image_url": data_url,
        "image_base64": b64_str,
    }


# =========================================================
# P06 旧接口保留（向后兼容，不再推荐）
# =========================================================
class P06Request(BaseModel):
    grid_id: str
    diagnosis: str
    recommended_strategies: List[RecommendedStrategyModel]
    street_image_url: str


@app.post("/api/p06-generate-image")
def p06_generate_image(request: P06Request):
    """旧的单步接口，保留以兼容旧调用，新流程请使用三步 /api/p06/* 接口。"""
    image_bytes = download_image(request.street_image_url)
    image_analysis = analyze_image_with_gemini(image_bytes)
    payload = {
        "diagnosis": request.diagnosis,
        "recommended_strategies": [s.model_dump() for s in request.recommended_strategies],
        "image_analysis": image_analysis,
    }
    prompt_json = call_gemini_for_image_prompt(payload)
    final_prompt = prompt_json.get("final_prompt", "")
    negative_prompt = prompt_json.get("negative_prompt", "")
    image_result = generate_image_from_prompt(final_prompt, negative_prompt)
    return {
        "image_analysis": image_analysis,
        "prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "generated_image": image_result,
    }