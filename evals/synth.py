"""Scenario-based synthetic claims for the golden set.

Labels come from the *scenario*, never from running the pipeline. The generator decides
"this is a legitimate rear-end collision" or "this hail claim has no storm behind it",
writes the evidence that scenario would produce (photos, detections, EXIF, statement,
API cassettes) and labels it with the business outcome. If the pipeline disagrees, the
pipeline is wrong, and the eval shows where.

Business rules the labels encode:
  1. any injected fraud signal         -> escalate
  2. coverage part not on the policy    -> deny
  3. major loss (structural damage)     -> escalate (over the auto-approval limit)
  4. low-quality evidence               -> escalate
  5. otherwise                          -> approve

The evidence is deliberately noisy: false-positive detections, missed detections,
claimants who omit EXIF, statement phrasings the critic's cue list does not know, and
"hard negatives" that look suspicious but are legitimate.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from claimsight.tools.photos import hamming
from claimsight.tools.weather import cassette_key

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # VINs never use I, O or Q
LOSS_START, LOSS_DAYS = date(2026, 3, 1), 245


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    hail: bool = False
    flood: bool = False


CITIES = [
    City("Dallas, TX", 32.777, -96.797, hail=True),
    City("Plano, TX", 33.020, -96.699, hail=True),
    City("Denver, CO", 39.739, -104.990, hail=True),
    City("Oklahoma City, OK", 35.468, -97.516, hail=True),
    City("Omaha, NE", 41.257, -95.995, hail=True),
    City("Wichita, KS", 37.687, -97.330, hail=True),
    City("Kansas City, MO", 39.100, -94.578, hail=True),
    City("Houston, TX", 29.760, -95.370, flood=True),
    City("New Orleans, LA", 29.951, -90.072, flood=True),
    City("Miami, FL", 25.762, -80.192, flood=True),
    City("Chicago, IL", 41.878, -87.630),
    City("Phoenix, AZ", 33.448, -112.074),
    City("Atlanta, GA", 33.749, -84.388),
    City("Seattle, WA", 47.606, -122.332),
    City("Minneapolis, MN", 44.978, -93.265),
    City("Austin, TX", 30.267, -97.743),
]


@dataclass(frozen=True)
class Vehicle:
    make: str
    model: str
    wmi: str  # world manufacturer identifier: first 3 VIN characters
    years: tuple[int, int]
    spoken_model: str | None = None  # how owners say it, e.g. "Silverado"


VEHICLES = [
    Vehicle("Toyota", "Camry", "4T1", (2015, 2025)),
    Vehicle("Honda", "Accord", "1HG", (2014, 2025)),
    Vehicle("Ford", "F-150", "1FT", (2015, 2025)),
    Vehicle("Tesla", "Model 3", "5YJ", (2018, 2025)),
    Vehicle("Chevrolet", "Silverado 1500", "3GC", (2016, 2025), spoken_model="Silverado"),
    Vehicle("Hyundai", "Tucson", "KM8", (2016, 2025)),
    Vehicle("Subaru", "Outback", "4S4", (2015, 2025)),
    Vehicle("Nissan", "Rogue", "JN8", (2014, 2025)),
    Vehicle("BMW", "3 Series", "WBA", (2015, 2025)),
    Vehicle("Jeep", "Wrangler", "1C4", (2014, 2025)),
    Vehicle("Kia", "Telluride", "5XY", (2020, 2025)),
    Vehicle("Mazda", "CX-5", "JM3", (2016, 2025)),
]

RELEVANT_CLAUSES = {
    "collision": ["PP-D-COLL", "PP-DEC-DED"],
    "glass": ["PP-D-COMP", "PP-D-GLASS", "PP-DEC-DED"],
    **{k: ["PP-D-COMP", "PP-DEC-DED"] for k in ("hail", "flood", "animal", "vandalism")},
}

# (part, damage types, severity range, probability the item is present)
Profile = list[tuple[str, tuple[str, ...], tuple[float, float], float]]
PROFILES: dict[str, Profile] = {
    "rear": [
        ("bumper_rear", ("dent", "crack"), (0.3, 0.8), 1.0),
        ("taillamp", ("lamp_broken",), (0.4, 0.8), 0.5),
        ("trunk", ("dent",), (0.3, 0.7), 0.3),
    ],
    "front": [
        ("bumper_front", ("dent", "crack"), (0.3, 0.8), 1.0),
        ("hood", ("dent",), (0.3, 0.7), 0.6),
        ("headlamp", ("lamp_broken",), (0.4, 0.8), 0.5),
        ("fender", ("dent",), (0.3, 0.6), 0.3),
    ],
    "side": [
        ("door", ("dent",), (0.3, 0.8), 1.0),
        ("fender", ("dent", "scratch"), (0.2, 0.6), 0.4),
    ],
    "major_front": [
        ("bumper_front", ("crack",), (0.8, 0.95), 1.0),
        ("hood", ("dent",), (0.7, 0.95), 1.0),
        ("headlamp", ("lamp_broken",), (0.7, 0.95), 0.8),
        ("fender", ("dent",), (0.6, 0.9), 0.7),
        ("frame", ("deform",), (0.7, 0.95), 1.0),
    ],
    "hail": [
        ("hood", ("dent",), (0.3, 0.7), 1.0),
        ("roof", ("dent",), (0.4, 0.8), 1.0),
        ("trunk", ("dent",), (0.3, 0.7), 0.5),
    ],
    "flood": [
        ("door", ("water_line",), (0.5, 0.9), 1.0),
        ("bumper_front", ("water_line",), (0.4, 0.8), 0.6),
    ],
    "glass": [("windshield", ("crack", "glass_shatter"), (0.3, 0.8), 1.0)],
    "vandalism": [
        ("door", ("scratch",), (0.3, 0.7), 1.0),
        ("windshield", ("glass_shatter",), (0.5, 0.9), 0.4),
    ],
}
FALSE_POSITIVES = [("door", "scratch"), ("fender", "scratch"), ("bumper_front", "scratch")]

PLACES = [
    "a red light on Main Street",
    "the grocery store parking lot",
    "a four-way stop",
    "the highway on-ramp",
    "the school pickup line",
    "my office garage",
]
OTHERS = ["pickup truck", "delivery van", "sedan", "SUV", "rideshare car"]
OBJECTS = ["a concrete pole", "a guardrail", "a parked car", "a mailbox"]

# Some phrasings deliberately fall outside the critic's cue list. That gap is real: the
# eval should expose it, and the week 7-8 LLM consistency check should close it.
STATEMENTS: dict[str, list[str]] = {
    "rear": [
        "I was stopped at {place} and a {other} rear-ended me.",
        "Got rear-ended by a {other} at {place}.",
        "A {other} hit me from behind at {place}.",
        "I was waiting at {place} when a {other} slammed into the back of my car.",
    ],
    "front": [
        "I hit {obj} at {place}.",
        "I rear-ended a {other} at {place} when it braked suddenly.",
        "I ran into the {other} ahead of me at {place}.",
        "Lost control on wet pavement and the front of my car struck {obj}.",
    ],
    "side": [
        "A {other} sideswiped me while changing lanes near {place}.",
        "Someone opened their door into mine at {place}.",
    ],
    "hail": [
        "Hail came down hard while my car was parked at {place}. The hood and roof are dented.",
        "A hailstorm passed through in the evening and dented my car all over.",
        "Golf-ball sized hail hit the car while I was at {place}.",
    ],
    "flood": [
        "Heavy rain flooded the street and the water came up past the doors.",
        "My car was parked at {place} when flash flooding hit. The water line is above the sills.",
    ],
    "glass": [
        "A rock kicked up by a {other} cracked my windshield on the highway.",
        "Debris hit the windshield yesterday and now a crack is spreading across it.",
    ],
    "animal": [
        "I hit a deer at night on a rural road.",
        "A deer jumped out near {place} and I could not stop in time.",
    ],
    "vandalism": [
        "Someone keyed both doors while I was parked at {place}.",
        "My car was vandalized overnight. The doors are scratched and the windshield is smashed.",
    ],
}

KIND_WEIGHTS = {
    "legit": 0.52,
    "fraud": 0.15,
    "hard_negative": 0.10,
    "deny": 0.08,
    "major": 0.08,
    "low_evidence": 0.07,
}
FRAUD_WEIGHTS = {
    "weather": 0.2,
    "duplicate_photo": 0.2,
    "exif_backdate": 0.15,
    "exif_location": 0.1,
    "statement_flip": 0.2,
    "vin_mismatch": 0.15,
}
FRAUD_FLAG = {
    "weather": "weather_corroboration",
    "duplicate_photo": "duplicate_photos",
    "exif_backdate": "exif_consistency",
    "exif_location": "exif_consistency",
    "statement_flip": "statement_consistency",
    "vin_mismatch": "vin_decode",
}
HARD_WEIGHTS = {
    "photo_prev_evening": 0.3,  # taken the evening before the reported date: timezone slip
    "near_duplicate": 0.25,  # similar-looking photo, but not the same image
    "photo_nearby_town": 0.2,  # taken 20-45 km away, e.g. at the body shop
    "hail_code_only": 0.25,  # no SPC report nearby, but the weather station logged hail
}
LOSS_WEIGHTS = {
    "collision": 0.40,
    "hail": 0.20,
    "glass": 0.12,
    "vandalism": 0.10,
    "flood": 0.10,
    "animal": 0.08,
}
ZONE_WEIGHTS = {"rear": 0.45, "front": 0.35, "side": 0.2}


@dataclass
class GeneratedSet:
    cases: list[dict]
    weather: dict[str, dict] = field(default_factory=dict)
    vins: dict[str, dict] = field(default_factory=dict)
    prior_photos: list[dict] = field(default_factory=list)

    def stats(self) -> dict:
        labels = [c["labels"] for c in self.cases]
        return {
            "n_cases": len(self.cases),
            "decisions": dict(Counter(lab["decision"] for lab in labels)),
            "loss_types": dict(Counter(c["claim"]["loss_type"] for c in self.cases)),
            "fraud_flags": dict(Counter(f for lab in labels for f in lab["fraud_flags"])),
            "tags": dict(Counter(t for lab in labels for t in lab["tags"])),
            "prior_photos": len(self.prior_photos),
        }


class ClaimGenerator:
    def __init__(self, seed: int = 7, existing_hashes: list[str] | None = None):
        self.rng = random.Random(seed)
        self.out = GeneratedSet(cases=[])
        self._hashes = list(existing_hashes or [])

    # -- primitives ---------------------------------------------------------------

    def _pick(self, weights: dict[str, float]) -> str:
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def _offset(self, lat: float, lon: float, km: float) -> tuple[float, float]:
        bearing = self.rng.uniform(0, 2 * math.pi)
        dlat = km / 111.0 * math.cos(bearing)
        dlon = km / (111.0 * math.cos(math.radians(lat))) * math.sin(bearing)
        return round(lat + dlat, 3), round(lon + dlon, 3)

    def _fresh_hash(self) -> str:
        while True:
            h = f"{self.rng.getrandbits(64):016x}"
            if all(hamming(h, other) > 16 for other in self._hashes):
                self._hashes.append(h)
                return h

    def _near_hash(self, base: str, lo: int, hi: int) -> str:
        bits = self.rng.sample(range(64), self.rng.randint(lo, hi))
        value = int(base, 16)
        for b in bits:
            value ^= 1 << b
        return f"{value:016x}"

    def _vin(self, v: Vehicle, year: int) -> str:
        vin = v.wmi + "".join(self.rng.choice(VIN_CHARS) for _ in range(14))
        # Shaped like NHTSAVinDecoder output, so replay and live look the same to the agent.
        self.out.vins[vin] = {
            "make": v.make.title(),
            "model": v.model,
            "year": str(year),
            "error_code": "0",
        }
        return vin

    def _statement(self, key: str) -> str:
        template = self.rng.choice(STATEMENTS[key])
        return template.format(
            place=self.rng.choice(PLACES),
            other=self.rng.choice(OTHERS),
            obj=self.rng.choice(OBJECTS),
        )

    # -- evidence -----------------------------------------------------------------

    def make_priors(self, n: int) -> None:
        for i in range(n):
            self.out.prior_photos.append(
                {
                    "claim_id": f"CLM-2025-S{i:03d}",
                    "photo_id": f"p-s{i:03d}",
                    "phash": self._fresh_hash(),
                }
            )

    def _damage(self, profile: str) -> list[dict]:
        items = []
        for part, types, (lo, hi), prob in PROFILES[profile]:
            if self.rng.random() < prob:
                items.append(
                    {
                        "part": part,
                        "damage_type": self.rng.choice(types),
                        "severity": round(self.rng.uniform(lo, hi), 2),
                    }
                )
        return items

    def _detections(self, truth: list[dict], low_evidence: bool) -> list[list[dict]]:
        """Spread true damage over 1-3 photos the way a detector would see it."""
        n_photos = self.rng.randint(1, 3)
        photos: list[list[dict]] = [[] for _ in range(n_photos)]
        conf = (0.51, 0.59) if low_evidence else (0.75, 0.97)
        missed = None
        if len(truth) > 1 and self.rng.random() < 0.08:
            candidates = [t for t in truth if t["part"] != "frame"]
            missed = self.rng.choice(candidates) if candidates else None
        for item in truth:
            if item is missed:
                continue
            noisy = {
                **item,
                "severity": round(
                    min(1, max(0, item["severity"] + self.rng.uniform(-0.05, 0.05))), 2
                ),
            }
            first = self.rng.randrange(n_photos)
            photos[first].append({**noisy, "confidence": round(self.rng.uniform(*conf), 2)})
            if n_photos > 1 and not low_evidence and self.rng.random() < 0.3:
                other = (first + 1) % n_photos
                c = round(self.rng.uniform(conf[0] - 0.1, conf[1] - 0.1), 2)
                photos[other].append({**noisy, "confidence": c})
        if not low_evidence and self.rng.random() < 0.2:
            present = {(t["part"], t["damage_type"]) for t in truth}
            options = [fp for fp in FALSE_POSITIVES if fp not in present]
            if options:
                part, dtype = self.rng.choice(options)
                photos[self.rng.randrange(n_photos)].append(
                    {
                        "part": part,
                        "damage_type": dtype,
                        "severity": round(self.rng.uniform(0.1, 0.3), 2),
                        "confidence": round(self.rng.uniform(0.2, 0.6), 2),
                    }
                )
        return photos

    def _photos(
        self,
        detections: list[list[dict]],
        lat: float,
        lon: float,
        loss_date: date,
        exif_mode: str,
        hash_mode: str,
    ) -> list[dict]:
        photos = []
        for i, dets in enumerate(detections):
            photo: dict = {"photo_id": f"p{i + 1}", "recorded_detections": dets}
            if exif_mode != "none":
                if exif_mode == "backdate":
                    day = loss_date - timedelta(days=self.rng.randint(5, 60))
                    hour = self.rng.randint(8, 20)
                elif exif_mode == "prev_evening":
                    day, hour = loss_date - timedelta(days=1), self.rng.randint(20, 23)
                else:
                    day = loss_date + timedelta(days=self.rng.choice([0, 0, 0, 1, 2, 3]))
                    hour = self.rng.randint(7, 21)
                taken = datetime(day.year, day.month, day.day, hour, self.rng.randint(0, 59))
                km = {
                    "far": self.rng.uniform(90, 400),
                    "nearby_town": self.rng.uniform(20, 45),
                }.get(exif_mode, self.rng.uniform(0, 2))
                photo["exif_taken_at"] = taken.isoformat()
                photo["exif_lat"], photo["exif_lon"] = self._offset(lat, lon, km)
            if i == 0 and hash_mode in ("duplicate", "near_duplicate"):
                prior = self.rng.choice(self.out.prior_photos)["phash"]
                lo, hi = (0, 4) if hash_mode == "duplicate" else (9, 14)
                photo["phash"] = self._near_hash(prior, lo, hi)
            else:
                photo["phash"] = self._fresh_hash()
            photos.append(photo)
        return photos

    def _weather(self, loss: str, lat: float, lon: float, day: date, mode: str) -> None:
        """mode: legit | code_only (hard negative) | fraud."""

        def report(km: float) -> dict:
            rlat, rlon = self._offset(lat, lon, km)
            return {
                "time_utc": f"{self.rng.randint(18, 23):02d}{self.rng.randint(0, 59):02d}",
                "lat": rlat,
                "lon": rlon,
                "size_in": self.rng.choice([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]),
                "location": "synthetic",
                "state": None,
            }

        far = [report(self.rng.uniform(35, 90)) for _ in range(self.rng.randint(0, 2))]
        if loss == "hail":
            if mode == "legit":
                near = [report(self.rng.uniform(3, 22)) for _ in range(self.rng.randint(1, 3))]
                reports, code = near + far, self.rng.choice([95, 96, 99, 3])
            elif mode == "code_only":
                reports, code = far, self.rng.choice([96, 99])
            else:
                reports, code = far, self.rng.choice([0, 1, 2, 3, 61])
            precip = round(self.rng.uniform(0, 30), 1)
        else:  # flood
            reports = []
            if mode == "fraud":
                precip, code = round(self.rng.uniform(0, 8), 1), self.rng.choice([0, 1, 2, 3])
            else:
                precip, code = round(self.rng.uniform(35, 140), 1), self.rng.choice([63, 65, 82])
        self.out.weather[cassette_key(lat, lon, day)] = {
            "hail_reports": reports,
            "daily": {
                "weather_code": code,
                "precipitation_sum_mm": precip,
                "wind_gusts_max_kmh": round(self.rng.uniform(15, 90), 1),
            },
        }

    # -- scenarios ----------------------------------------------------------------

    def case(self, idx: int) -> dict:
        kind = self._pick(KIND_WEIGHTS)
        fraud = self._pick(FRAUD_WEIGHTS) if kind == "fraud" else None
        hard = self._pick(HARD_WEIGHTS) if kind == "hard_negative" else None

        if fraud == "weather":
            loss = self.rng.choices(["hail", "flood"], weights=[0.7, 0.3])[0]
        elif fraud == "statement_flip" or kind == "deny":
            loss = "collision"
        elif hard == "hail_code_only":
            loss = "hail"
        elif kind == "major":
            loss = self.rng.choice(["collision", "animal"])
        else:
            loss = self._pick(LOSS_WEIGHTS)

        zone = None
        if loss == "collision":
            if kind == "major":
                zone = "front"
            elif fraud == "statement_flip":
                zone = self.rng.choice(["rear", "front"])
            else:
                zone = self._pick(ZONE_WEIGHTS)
        if kind == "major":
            profile = "major_front"
        elif loss == "collision":
            profile = zone
        elif loss == "animal":
            profile = "front"
        else:
            profile = loss

        statement_key = zone if loss == "collision" else loss
        if fraud == "statement_flip":
            statement_key = "front" if zone == "rear" else "rear"

        if loss == "hail":
            cities = [c for c in CITIES if c.hail]
        elif loss == "flood":
            cities = [c for c in CITIES if c.flood]
        else:
            cities = CITIES
        city = self.rng.choice(cities)
        loss_date = LOSS_START + timedelta(days=self.rng.randrange(LOSS_DAYS))
        while True:  # two weather claims must not share a cassette key
            lat, lon = self._offset(city.lat, city.lon, self.rng.uniform(0, 6))
            if cassette_key(lat, lon, loss_date) not in self.out.weather:
                break

        if kind == "deny":
            policy = "P-200"  # comprehensive only, so a collision is not covered
        elif loss == "collision":
            policy = self.rng.choice(["P-100", "P-300"])
        else:
            policy = self.rng.choice(["P-100", "P-200", "P-300"])

        vehicle = self.rng.choice(VEHICLES)
        year = self.rng.randint(*vehicle.years)
        vin = self._vin(vehicle, year)
        reported = f"{year} {vehicle.make} {vehicle.spoken_model or vehicle.model}"
        if fraud == "vin_mismatch":
            other = self.rng.choice([v for v in VEHICLES if v.make != vehicle.make])
            reported = f"{self.rng.randint(*other.years)} {other.make} {other.model}"

        truth = self._damage(profile)
        exif_mode = {
            "exif_backdate": "backdate",
            "exif_location": "far",
        }.get(
            fraud or "",
            {"photo_prev_evening": "prev_evening", "photo_nearby_town": "nearby_town"}.get(
                hard or "", "normal"
            ),
        )
        if exif_mode == "normal" and self.rng.random() < 0.05:
            exif_mode = "none"  # screenshots and messaging apps strip EXIF
        hash_mode = (
            "duplicate"
            if fraud == "duplicate_photo"
            else ("near_duplicate" if hard == "near_duplicate" else "fresh")
        )
        photos = self._photos(
            self._detections(truth, low_evidence=kind == "low_evidence"),
            lat,
            lon,
            loss_date,
            exif_mode,
            hash_mode,
        )
        if loss in ("hail", "flood"):
            mode = (
                "fraud"
                if fraud == "weather"
                else ("code_only" if hard == "hail_code_only" else "legit")
            )
            self._weather(loss, lat, lon, loss_date, mode)

        if fraud:
            decision = "escalate"
        elif kind == "deny":
            decision = "deny"
        elif kind in ("major", "low_evidence"):
            decision = "escalate"
        else:
            decision = "approve"

        tags = [kind, loss] + ([f"zone:{zone}"] if zone else [])
        tags += [f"fraud:{fraud}"] if fraud else []
        tags += [f"hard:{hard}"] if hard else []
        tags += ["no_exif"] if exif_mode == "none" else []

        return {
            "claim": {
                "claim_id": f"SYN-{idx:04d}",
                "policy_id": policy,
                "loss_type": loss,
                "loss_date": loss_date.isoformat(),
                "location": {"lat": lat, "lon": lon, "label": city.name},
                "vin": vin,
                "reported_vehicle": reported,
                "statement": self._statement(statement_key),
                "photos": photos,
            },
            "labels": {
                "decision": decision,
                "fraud_flags": [FRAUD_FLAG[fraud]] if fraud else [],
                "relevant_clause_ids": RELEVANT_CLAUSES[loss],
                "damage": [[t["part"], t["damage_type"]] for t in truth],
                "tags": tags,
            },
        }


def generate(
    n: int, seed: int = 7, existing_hashes: list[str] | None = None, n_priors: int = 40
) -> GeneratedSet:
    gen = ClaimGenerator(seed=seed, existing_hashes=existing_hashes)
    gen.make_priors(n_priors)
    gen.out.cases = [gen.case(i) for i in range(1, n + 1)]
    return gen.out
