"""IATA metropolitan (city) code mappings.

Maps IATA city / metropolitan codes to the set of commercial airports they
encompass.  Used to let users pass a single metro code (e.g. ``LON``) instead
of expanding to its constituent airports (``LHR``, ``LGW``, ``STN``, ``LCY``,
``LTN``, ``SEN``).

Google Flights' UI groups these airports natively; the underlying API accepts
multiple airports per slot via the ``departure_airport`` / ``arrival_airport``
list-of-pairs structure.

When a code exists in both this map and the :class:`Airport` enum (e.g.
``SHA`` denotes both the Shanghai metropolitan area and Shanghai Hongqiao
airport specifically, ``SAO`` denotes São Paulo metro and Campo de Marte),
the city expansion takes precedence so that searches match Google Flights'
default behavior.  Code collisions are kept intentionally narrow — only
``SHA`` and ``SAO`` here — and Campo de Marte (``SAO``) is a small domestic
airport with no commercial international flights, so the override is safe
in practice.
"""

from .airport import Airport

CITY_TO_AIRPORTS: dict[str, list[Airport]] = {
    # Europe
    "LON": [Airport.LHR, Airport.LGW, Airport.STN, Airport.LCY, Airport.LTN, Airport.SEN],
    "PAR": [Airport.CDG, Airport.ORY, Airport.BVA],
    "ROM": [Airport.FCO, Airport.CIA],
    "MIL": [Airport.MXP, Airport.LIN, Airport.BGY],
    "MOW": [Airport.SVO, Airport.DME, Airport.VKO, Airport.ZIA],
    "STO": [Airport.ARN, Airport.BMA, Airport.NYO, Airport.VST],
    # Asia
    "TYO": [Airport.HND, Airport.NRT],
    "OSA": [Airport.KIX, Airport.ITM, Airport.UKB],
    "SEL": [Airport.ICN, Airport.GMP],
    "SHA": [Airport.PVG, Airport.SHA],
    "BJS": [Airport.PEK, Airport.PKX],
    "JKT": [Airport.CGK, Airport.HLP],
    # North America
    "NYC": [Airport.JFK, Airport.LGA, Airport.EWR],
    "CHI": [Airport.ORD, Airport.MDW, Airport.RFD],
    "WAS": [Airport.IAD, Airport.DCA, Airport.BWI],
    "YTO": [Airport.YYZ, Airport.YTZ, Airport.YHM],
    "YMQ": [Airport.YUL, Airport.YHU, Airport.YMX],
    # South America
    "BUE": [Airport.EZE, Airport.AEP],
    "SAO": [Airport.GRU, Airport.CGH, Airport.VCP],
    "RIO": [Airport.GIG, Airport.SDU],
}


def is_city_code(code: str) -> bool:
    """Return True if the given code is a known IATA metropolitan code."""
    return code.upper() in CITY_TO_AIRPORTS


def resolve_city(code: str) -> list[Airport]:
    """Resolve a city code to its list of constituent airports.

    Args:
        code: IATA metropolitan code (e.g., 'LON', 'NYC').

    Returns:
        List of :class:`Airport` enum members for that metro.

    Raises:
        KeyError: If the code is not a known metropolitan code.

    """
    return list(CITY_TO_AIRPORTS[code.upper()])
