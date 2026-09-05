"""
treatments.py — Agronomic knowledge base: what to actually DO about each class.

The classifier answers "which disease". This module answers the question a
grower actually has: "how bad is it, how fast must I move, and what do I spray".

Every entry carries:
    pathogen / scientific name / type   - what it is
    symptoms / spread                   - how to confirm it by eye
    urgency (1-5) + action_window_days  - how fast to act
    contagion                           - risk to neighbouring plants
    yield_loss_pct                      - (low, high) if left untreated
    organic / chemical / cultural       - three tiers of response
    do_not                              - common, expensive mistakes

Dosages are per litre of water for foliar sprays unless stated otherwise, and
follow common label rates for the listed formulation strength. Pesticide
registration differs by country and crop stage - DISCLAIMER is returned with
every lookup and is shown in the UI.

Usage:
    from treatments import get_treatment, severity_band
    info = get_treatment("Tomato_Late_blight", severity=0.31)
"""

from typing import Optional

import config

DISCLAIMER = (
    "Guidance only. Pesticide registration, permitted dose and pre-harvest "
    "interval vary by country and crop stage - confirm with the product label "
    "and your local agricultural extension office before applying."
)

# Severity bands, expressed as fraction of leaf area showing symptoms.
SEVERITY_BANDS = [
    (0.05, "Trace",    "Isolated lesions - spot-treat and monitor."),
    (0.15, "Mild",     "Early infection - act now and it stays cheap."),
    (0.35, "Moderate", "Established infection - begin a full spray programme."),
    (1.01, "Severe",   "Advanced - protect the remaining canopy and the crop around it."),
]


def severity_band(fraction: Optional[float]) -> Optional[dict]:
    """Map an affected-area fraction in [0, 1] to a named band + advice."""
    if fraction is None:
        return None
    frac = max(0.0, min(1.0, float(fraction)))
    for ceiling, label, advice in SEVERITY_BANDS:
        if frac < ceiling:
            return {"fraction": round(frac, 4),
                    "percent": round(frac * 100, 1),
                    "band": label,
                    "advice": advice}
    return {"fraction": round(frac, 4), "percent": round(frac * 100, 1),
            "band": "Severe", "advice": SEVERITY_BANDS[-1][2]}


# ==============================================================================
# The knowledge base - keyed by config.CLASS_NAMES
# ==============================================================================

DISEASE_INFO = {

    # -- 0 ---------------------------------------------------------------------
    "Tomato__Target_Spot": {
        "display": "Target Spot",
        "pathogen": "Corynespora cassiicola",
        "type": "Fungal",
        "urgency": 3,
        "action_window_days": "5-7",
        "contagion": "High",
        "yield_loss_pct": (20, 40),
        "symptoms": (
            "Small brown flecks that expand into lesions with faint concentric "
            "rings and a light-brown centre; heavy spotting on stems and fruit "
            "as well as leaves. Lower canopy first."
        ),
        "spread": (
            "Airborne conidia plus rain and irrigation splash. Needs 20-30 C "
            "with long leaf-wetness periods; worst in dense, humid canopies."
        ),
        "organic": [
            "Copper hydroxide 53.8% DF @ 2 g/L, every 7-10 days.",
            "Bacillus subtilis QST-713 @ 2-4 ml/L as a protectant rotation partner.",
            "Strip and bag the infected lower leaves - do not leave them on the bed.",
            "Open the canopy: prune suckers and widen spacing to cut leaf wetness.",
        ],
        "chemical": [
            "Chlorothalonil 720 SC @ 2 ml/L, 7-10 day interval (FRAC M05).",
            "Azoxystrobin 23 SC @ 1 ml/L (FRAC 11) - max 2 sprays before rotating.",
            "Difenoconazole 25 EC @ 0.5 ml/L (FRAC 3).",
            "Rotate FRAC groups every 2 sprays; C. cassiicola builds resistance fast.",
        ],
        "cultural": [
            "Switch overhead irrigation to drip.",
            "3-year rotation away from tomato, cucurbits and soybean.",
            "Stake and prune for airflow; irrigate in the morning.",
            "Destroy crop debris at the end of the season.",
        ],
        "do_not": [
            "Do not spray the same FRAC 11 product back to back - it is the "
            "fastest way to lose the chemistry.",
        ],
    },

    # -- 1 ---------------------------------------------------------------------
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tomato mosaic virus (Tobamovirus)",
        "type": "Viral",
        "urgency": 5,
        "action_window_days": "immediately",
        "contagion": "Very high (mechanical)",
        "yield_loss_pct": (20, 70),
        "symptoms": (
            "Light/dark green mottling and mosaic, puckered or fern-like "
            "distorted leaves, stunting. Fruit may show internal browning. "
            "Symptoms are patchy across the plant, not bottom-up."
        ),
        "spread": (
            "Mechanically - hands, clothing, tools, contaminated seed and "
            "tobacco products. Extraordinarily stable: survives years in dry "
            "debris and on stakes. NOT insect-transmitted."
        ),
        "organic": [
            "There is no cure. Rogue infected plants: bag them in place and "
            "remove them from the field - never compost.",
            "Wash hands with soap and disinfect tools in 10% household bleach "
            "or 3% trisodium phosphate between every plant.",
            "Ban tobacco use in and around the crop; smokers must wash first.",
            "Plant certified virus-free seed of Tm-2a / Tm-2^2 resistant hybrids.",
        ],
        "chemical": [
            "None. No fungicide, bactericide or insecticide affects a virus.",
            "Seed treatment only: soak in 10% trisodium phosphate for 15 min, "
            "then rinse - this inactivates surface virus on seed.",
        ],
        "cultural": [
            "Handle plants only when dry, and work healthy blocks before infected ones.",
            "Steam or bleach-soak stakes, trellis clips and trays between seasons.",
            "Remove volunteer tomatoes and solanaceous weeds.",
        ],
        "do_not": [
            "Do not spend money on fungicides for this - it is a virus and they "
            "will do nothing.",
            "Do not prune or tie infected plants and then touch healthy ones.",
        ],
    },

    # -- 2 ---------------------------------------------------------------------
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus, vectored by whitefly (Bemisia tabaci)",
        "type": "Viral (insect-vectored)",
        "urgency": 5,
        "action_window_days": "1-2",
        "contagion": "Very high (via whitefly)",
        "yield_loss_pct": (50, 100),
        "symptoms": (
            "Upward-cupped leaflets with yellow margins, sharply reduced leaf "
            "size, bushy stunted growth, and heavy flower drop. Infected early, "
            "the plant sets almost no fruit."
        ),
        "spread": (
            "Only by whitefly, persistently - one whitefly stays infective for "
            "life. Not seed-borne, not spread by handling."
        ),
        "organic": [
            "Kill the whiteflies BEFORE roguing: spray the plant with "
            "insecticidal soap, then bag and remove it.",
            "Yellow sticky traps at 1 per 10 m2 for monitoring and mass trapping.",
            "Neem (azadirachtin 1500 ppm) @ 3-5 ml/L, or Beauveria bassiana, "
            "targeting the leaf underside every 5-7 days.",
            "UV-reflective silver plastic mulch - repels incoming whitefly.",
            "50-mesh insect netting over the nursery and tunnel vents.",
        ],
        "chemical": [
            "Treat the vector, not the virus.",
            "Imidacloprid 17.8 SL @ 0.3 ml/L as a soil drench at transplant (IRAC 4A).",
            "Spiromesifen 22.9 SC @ 1 ml/L (IRAC 23) - strong on nymphs.",
            "Pyriproxyfen 10 EC @ 1 ml/L (IRAC 7C) - insect growth regulator.",
            "Diafenthiuron 50 WP @ 1 g/L (IRAC 12A).",
            "Rotate IRAC groups every generation (about 3 weeks).",
        ],
        "cultural": [
            "Plant Ty-1 / Ty-3 resistant hybrids where TYLCV is endemic.",
            "Destroy the previous crop completely before transplanting the next - "
            "no overlapping plantings.",
            "Clear weed hosts and volunteers within 30 m of the field.",
        ],
        "do_not": [
            "Do not pull an infected plant while whiteflies are still on it - you "
            "release a cloud of viruliferous vectors into the crop.",
            "Do not run pyrethroid-only programmes: they flare whitefly and "
            "destroy the natural enemies keeping it in check.",
        ],
    },

    # -- 3 ---------------------------------------------------------------------
    "Tomato_Bacterial_spot": {
        "display": "Bacterial Spot",
        "pathogen": "Xanthomonas euvesicatoria / perforans / gardneri",
        "type": "Bacterial",
        "urgency": 4,
        "action_window_days": "2-3",
        "contagion": "High",
        "yield_loss_pct": (10, 50),
        "symptoms": (
            "Small, dark, water-soaked spots that turn brown-black with a "
            "yellow halo; spots stay angular and greasy-looking, and become "
            "scabby raised lesions on fruit. Leaves shot-hole and drop."
        ),
        "spread": (
            "Wind-driven rain and splash, plus workers and tools moving through "
            "a wet crop. Seed-borne. Explodes at 24-30 C with rain."
        ),
        "organic": [
            "Copper hydroxide 53.8% DF @ 2 g/L tank-mixed with mancozeb - the "
            "mix is markedly better than copper alone.",
            "Acibenzolar-S-methyl 50 WG @ 0.35 g/L every 7 days - switches on "
            "the plant's own systemic resistance.",
            "Bacillus subtilis or B. amyloliquefaciens as a rotation partner.",
            "Hot-water seed treatment: 50 C for 25 minutes, then cool and dry.",
        ],
        "chemical": [
            "Copper hydroxide + mancozeb 75 WP @ 2.5 g/L, weekly under pressure.",
            "Acibenzolar-S-methyl 50 WG @ 0.35 g/L on a 7-day cycle.",
            "Streptomycin 200 ppm - nursery only, and only where registered.",
            "Copper resistance is widespread: if 2 sprays do nothing, stop and "
            "lean on sanitation and ASM instead.",
        ],
        "cultural": [
            "Never scout, prune, tie or harvest while the foliage is wet.",
            "Drip irrigation only - overhead watering spreads it down the row.",
            "2-3 year rotation away from tomato and pepper.",
            "Certified pathogen-free seed and clean transplant trays.",
        ],
        "do_not": [
            "Do not treat this like a fungal disease - triazoles and "
            "strobilurins have no effect on bacteria.",
        ],
    },

    # -- 4 ---------------------------------------------------------------------
    "Tomato_Early_blight": {
        "display": "Early Blight",
        "pathogen": "Alternaria linariae (A. solani)",
        "type": "Fungal",
        "urgency": 3,
        "action_window_days": "5-7",
        "contagion": "Moderate-High",
        "yield_loss_pct": (20, 50),
        "symptoms": (
            "Dark brown lesions with concentric 'target' rings surrounded by a "
            "yellow halo, always starting on the OLDEST lower leaves and moving "
            "up. Stem collar rot and dark sunken lesions at the fruit stem end."
        ),
        "spread": (
            "Conidia from infected debris and soil, moved by wind and splash. "
            "Favoured by warm, humid weather alternating with dry spells, and "
            "by plants stressed for nitrogen or carrying a heavy fruit load."
        ),
        "organic": [
            "Copper hydroxide 53.8% DF @ 2 g/L every 7-10 days.",
            "Bacillus subtilis QST-713 @ 2-4 ml/L, alternated with copper.",
            "Potassium bicarbonate @ 5 g/L for light early infection.",
            "Remove all leaves below the first fruit cluster and bag them.",
            "Straw or plastic mulch to stop soil splashing spores onto leaves.",
        ],
        "chemical": [
            "Chlorothalonil 720 SC @ 2 ml/L, 7-10 day interval (FRAC M05).",
            "Mancozeb 75 WP @ 2.5 g/L (FRAC M03) as the protectant backbone.",
            "Azoxystrobin 23 SC @ 1 ml/L (FRAC 11) or difenoconazole 25 EC @ "
            "0.5 ml/L (FRAC 3) when pressure is high.",
            "Alternate single-site (11, 3) with multi-site (M03, M05) chemistry.",
        ],
        "cultural": [
            "Mulch, stake and prune the lower canopy.",
            "Keep nitrogen adequate - deficient plants are hit far harder.",
            "2-3 year rotation; destroy debris rather than ploughing it in.",
        ],
        "do_not": [
            "Do not wait for the upper canopy to show symptoms - by then you "
            "have already lost the lower third of the plant.",
        ],
    },

    # -- 5 ---------------------------------------------------------------------
    "Tomato_healthy": {
        "display": "Healthy",
        "pathogen": "-",
        "type": "Healthy",
        "urgency": 0,
        "action_window_days": "-",
        "contagion": "None",
        "yield_loss_pct": (0, 0),
        "symptoms": (
            "Uniform green colour, no lesions, no mottling, no curling and no "
            "stippling on either leaf surface."
        ),
        "spread": "-",
        "organic": [
            "Scout twice a week, and check the leaf UNDERSIDE - mites, whitefly "
            "and leaf mould all start there.",
            "Keep straw or plastic mulch in place to block soil-splash infection.",
            "Preventive copper or Bacillus only when a wet, warm spell is forecast.",
        ],
        "chemical": [
            "No treatment needed. Do not spray on a calendar out of habit - it "
            "costs money and accelerates resistance.",
        ],
        "cultural": [
            "Drip irrigation, watering in the morning so leaves dry by evening.",
            "Balanced NPK with adequate calcium to prevent blossom-end rot.",
            "Remove leaves below the first cluster once fruit has set.",
            "Disinfect pruning tools between plants.",
        ],
        "do_not": [
            "Do not work the crop while foliage is wet, even when it looks clean.",
        ],
    },

    # -- 6 ---------------------------------------------------------------------
    "Tomato_Late_blight": {
        "display": "Late Blight",
        "pathogen": "Phytophthora infestans (oomycete, not a true fungus)",
        "type": "Oomycete",
        "urgency": 5,
        "action_window_days": "within 24 hours",
        "contagion": "Extreme - regional",
        "yield_loss_pct": (50, 100),
        "symptoms": (
            "Large greasy grey-green blotches that turn brown-black, often with "
            "a pale halo, and white fuzzy sporulation on the leaf underside in "
            "humid mornings. Firm brown greasy patches on green fruit. Spreads "
            "across the whole canopy, not bottom-up."
        ),
        "spread": (
            "The emergency case. Sporangia travel kilometres on wind and a "
            "field can collapse in 7-10 days. Needs cool 10-24 C with over 90% "
            "humidity or extended leaf wetness."
        ),
        "organic": [
            "Copper hydroxide 53.8% DF @ 2-3 g/L on a strict 5-7 day PREVENTIVE "
            "cycle - organic options are protectant only and will not rescue an "
            "established infection.",
            "Destroy infected plants the same day: bag in place, remove from the "
            "field, burn or bury deeply. Never compost, never leave a cull pile.",
            "Destroy volunteer potatoes and cull piles - they are the season's "
            "primary inoculum source.",
        ],
        "chemical": [
            "Mandipropamid 23.4 SC @ 0.8 ml/L (FRAC 40).",
            "Cymoxanil + mancozeb 8+64 WP @ 3 g/L (FRAC 27 + M03).",
            "Dimethomorph 50 WP @ 1 g/L (FRAC 40).",
            "Fluopicolide + propamocarb (FRAC 43 + 28) for rotation.",
            "5-7 day intervals while conditions stay favourable; tighten to 5 "
            "days in continuous rain.",
        ],
        "cultural": [
            "Grow Ph-2 / Ph-3 resistant varieties ('Mountain Magic', 'Defiant') "
            "where late blight is an annual event.",
            "Wide spacing and aggressive pruning so the canopy dries by midday.",
            "Watch regional late-blight alerts and spray ahead of the front.",
            "Tell your neighbours - your spores are their problem within days.",
        ],
        "do_not": [
            "Do not rely on strobilurins or triazoles: this is an oomycete, and "
            "true-fungus chemistry is largely ineffective against it.",
            "Do not delay a day to 'see how it develops'. There is no disease on "
            "this list where 24 hours costs more.",
        ],
    },

    # -- 7 ---------------------------------------------------------------------
    "Tomato_Leaf_Mold": {
        "display": "Leaf Mold",
        "pathogen": "Passalora fulva (syn. Fulvia fulva, Cladosporium fulvum)",
        "type": "Fungal",
        "urgency": 3,
        "action_window_days": "4-6",
        "contagion": "High in enclosed structures",
        "yield_loss_pct": (10, 40),
        "symptoms": (
            "Pale green to yellow blotches on the UPPER leaf surface with no "
            "sharp margin, and directly beneath each blotch an olive-green to "
            "grey-brown velvety mould on the underside. Turn the leaf over to "
            "confirm - that velvet is diagnostic."
        ),
        "spread": (
            "Overwhelmingly a greenhouse and high-tunnel disease. Conidia move "
            "on air currents, hands and tools; needs relative humidity above "
            "85%, and essentially stops below 70%."
        ),
        "organic": [
            "Humidity is the real control: vent and heat together so the air "
            "never reaches dew point, especially at night.",
            "Copper hydroxide 53.8% DF @ 2 g/L every 7-10 days.",
            "Bacillus subtilis QST-713 @ 2-4 ml/L as a rotation partner.",
            "Remove and bag the lower leaves to open the base of the canopy.",
        ],
        "chemical": [
            "Chlorothalonil 720 SC @ 2 ml/L (FRAC M05).",
            "Difenoconazole 25 EC @ 0.5 ml/L (FRAC 3).",
            "Azoxystrobin 23 SC @ 1 ml/L (FRAC 11).",
            "Mancozeb 75 WP @ 2.5 g/L (FRAC M03).",
        ],
        "cultural": [
            "Vent at night and run a little heat - condensation on the leaf is "
            "what starts this.",
            "Wider plant spacing, de-leafing and sucker removal.",
            "Grow Cf-gene resistant cultivars; note that new races overcome "
            "single Cf genes, so rotate cultivars too.",
            "Drip irrigation only, and never mist the foliage.",
        ],
        "do_not": [
            "Do not close the house tight overnight to hold heat - it drives "
            "humidity to 100% and hands the fungus the crop.",
        ],
    },

    # -- 8 ---------------------------------------------------------------------
    "Tomato_Septoria_leaf_spot": {
        "display": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici",
        "type": "Fungal",
        "urgency": 4,
        "action_window_days": "3-5",
        "contagion": "High",
        "yield_loss_pct": (30, 50),
        "symptoms": (
            "Many small circular spots, 2-4 mm, with a dark brown margin and a "
            "distinctly GREY or tan centre speckled with tiny black pycnidia "
            "(visible with a hand lens). Starts on the lowest leaves and "
            "defoliates the plant upward. Rarely touches the fruit."
        ),
        "spread": (
            "Splash-dispersed spores from infected debris and nightshade weeds. "
            "Needs prolonged leaf wetness at 20-25 C."
        ),
        "organic": [
            "Copper hydroxide 53.8% DF @ 2 g/L every 7-10 days.",
            "Bacillus subtilis QST-713 @ 2-4 ml/L, alternated with copper.",
            "Strip and bag infected lower leaves at the first spots - this alone "
            "buys weeks.",
            "Straw mulch to stop splash from the soil surface.",
        ],
        "chemical": [
            "Chlorothalonil 720 SC @ 2 ml/L, 7-10 day interval (FRAC M05).",
            "Mancozeb 75 WP @ 2.5 g/L (FRAC M03).",
            "Azoxystrobin 23 SC @ 1 ml/L (FRAC 11) or difenoconazole 25 EC @ "
            "0.5 ml/L (FRAC 3), rotated with the protectants above.",
        ],
        "cultural": [
            "Mulch and stake so no leaf touches soil.",
            "3-year rotation; control nightshade and horsenettle weeds nearby.",
            "Morning drip irrigation, never overhead.",
            "Remove and destroy all debris at season end.",
        ],
        "do_not": [
            "Do not ignore the defoliation - stripped plants sunscald their "
            "fruit, which is where most of the loss actually comes from.",
        ],
    },

    # -- 9 ---------------------------------------------------------------------
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Two-Spotted Spider Mites",
        "pathogen": "Tetranychus urticae (arachnid pest, not a pathogen)",
        "type": "Pest (mite)",
        "urgency": 4,
        "action_window_days": "2-4",
        "contagion": "High",
        "yield_loss_pct": (20, 60),
        "symptoms": (
            "Fine pale stippling that gives the leaf a sand-blasted, bronzed "
            "look, then fine webbing between the leaf and stem. Turn the leaf "
            "over: the mites are tiny moving dots on the underside."
        ),
        "spread": (
            "Wind, clothing and tools. Populations double roughly every 3 days "
            "above 30 C, so a light infestation becomes a crop failure in two "
            "weeks. Hot, dry, dusty conditions are ideal for it."
        ),
        "organic": [
            "Predatory mites - Phytoseiulus persimilis or Neoseiulus "
            "californicus - released early are the durable fix.",
            "Strong water jets directed at the leaf underside knock populations "
            "down and raise humidity.",
            "Insecticidal soap @ 5 ml/L or horticultural / neem oil @ 5 ml/L, "
            "repeated 3 times at 5-7 day intervals (poor egg kill, so repeat).",
            "Raise humidity and suppress dust on field roads and headlands.",
        ],
        "chemical": [
            "Use a true MITICIDE - most insecticides do not control mites.",
            "Abamectin 1.9 EC @ 0.5 ml/L (IRAC 6).",
            "Spiromesifen 22.9 SC @ 1 ml/L (IRAC 23).",
            "Etoxazole 10 SC @ 0.4 ml/L (IRAC 10B) - eggs and juveniles.",
            "Fenpyroximate 5 EC @ 1 ml/L (IRAC 21A).",
            "Rotate IRAC groups strictly; never spray the same group twice.",
            "Coverage of the leaf UNDERSIDE decides whether the spray works.",
        ],
        "cultural": [
            "Avoid drought stress - water-stressed plants build mites fastest.",
            "Avoid excess nitrogen, which boosts mite reproduction.",
            "Remove weed hosts around the field.",
        ],
        "do_not": [
            "Do not use broad-spectrum pyrethroids or carbaryl - they wipe out "
            "predatory mites and reliably cause a worse flare-up than the one "
            "you started with.",
        ],
    },
}


# ==============================================================================
# Lookup API
# ==============================================================================

URGENCY_LABELS = {
    0: "None",
    1: "Routine",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Critical",
}


def get_treatment(class_name: str,
                  severity: Optional[float] = None,
                  confidence: Optional[float] = None) -> dict:
    """
    Full advisory for one class.

    Args:
        class_name : a value from config.CLASS_NAMES (or a DISPLAY_NAME).
        severity   : optional affected-leaf-area fraction in [0, 1]; when given,
                     the response carries a severity band and a tuned headline.
        confidence : optional model confidence, echoed back so the UI can warn
                     when advice rests on a shaky prediction.

    Raises KeyError for an unknown class.
    """
    key = _resolve_key(class_name)
    info = DISEASE_INFO[key]

    # A healthy leaf still measures a percent or two of off-green pixels (leaf
    # shadow, midrib, specular highlight). Reporting that as "Trace disease"
    # would be noise dressed up as a finding, so the band is dropped instead.
    band = None if info["type"] == "Healthy" else severity_band(severity)
    return {
        "class": key,
        "display": info["display"],
        "pathogen": info["pathogen"],
        "type": info["type"],
        "urgency": info["urgency"],
        "urgency_label": URGENCY_LABELS[info["urgency"]],
        "action_window_days": info["action_window_days"],
        "contagion": info["contagion"],
        "yield_loss_pct": list(info["yield_loss_pct"]),
        "symptoms": info["symptoms"],
        "spread": info["spread"],
        "organic": list(info["organic"]),
        "chemical": list(info["chemical"]),
        "cultural": list(info["cultural"]),
        "do_not": list(info["do_not"]),
        "severity": band,
        "confidence": confidence,
        "headline": _headline(info, band),
        "disclaimer": DISCLAIMER,
    }


def _resolve_key(name: str) -> str:
    """Accept either a CLASS_NAMES entry or a DISPLAY_NAMES entry."""
    if name in DISEASE_INFO:
        return name
    if name in config.DISPLAY_NAMES:
        return config.CLASS_NAMES[config.DISPLAY_NAMES.index(name)]
    stripped = name.replace("✓", "").strip().lower()
    for i, d in enumerate(config.DISPLAY_NAMES):
        if d.replace("✓", "").strip().lower() == stripped:
            return config.CLASS_NAMES[i]
    raise KeyError(f"No treatment entry for '{name}'.")


def _headline(info: dict, band: Optional[dict]) -> str:
    """One sentence a grower can act on without reading the rest."""
    if info["type"] == "Healthy":
        return "No disease detected - keep to the preventive routine below."

    window = info["action_window_days"]
    if window == "immediately":
        when = "Act immediately."
    elif window.startswith("within"):
        when = f"Act {window}."
    else:
        when = f"Act within {window} days."

    if band is None:
        return f"{info['display']} detected. {when}"

    return (f"{band['band']} {info['display']} - roughly {band['percent']}% of "
            f"the leaf is affected. {when}")


def summary_line(class_name: str) -> str:
    """Compact one-liner for CLI output and camera overlays."""
    info = DISEASE_INFO[_resolve_key(class_name)]
    if info["type"] == "Healthy":
        return "Healthy - no action needed"
    lo, hi = info["yield_loss_pct"]
    return (f"{info['type']} - urgency {info['urgency']}/5 "
            f"({URGENCY_LABELS[info['urgency']]}), act {info['action_window_days']}, "
            f"{lo}-{hi}% yield loss untreated")


# Fail fast if config and the knowledge base ever drift apart.
_missing = [c for c in config.CLASS_NAMES if c not in DISEASE_INFO]
if _missing:
    raise RuntimeError(f"treatments.py is missing entries for: {_missing}")
