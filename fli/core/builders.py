"""Shared building utilities for constructing search filters.

This module provides builder functions used by both the CLI and MCP interfaces
to construct flight search filter objects.
"""

from datetime import datetime, timedelta

from fli.models import Airport, FlightSegment, TimeRestrictions, TripType


def _as_airport_list(value: "Airport | list[Airport]") -> list[Airport]:
    """Normalize a single airport or list of airports into a non-empty list."""
    airports = [value] if isinstance(value, Airport) else list(value)
    if not airports:
        raise ValueError("At least one airport is required")
    return airports


def normalize_date(date_str: str) -> str:
    """Normalize a date string to zero-padded YYYY-MM-DD format.

    Args:
        date_str: Date string in YYYY-MM-DD format (e.g., '2026-4-2' or '2026-04-02')

    Returns:
        Zero-padded date string (e.g., '2026-04-02')

    Raises:
        ValueError: If the date string is not a valid date

    """
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")


def build_time_restrictions(
    departure_window: str | None = None,
    arrival_window: str | None = None,
) -> TimeRestrictions | None:
    """Build a TimeRestrictions object from time window strings.

    Args:
        departure_window: Departure time range in 'HH-HH' format (e.g., '6-20')
        arrival_window: Arrival time range in 'HH-HH' format (e.g., '8-22')

    Returns:
        TimeRestrictions object, or None if no restrictions specified

    """
    if not departure_window and not arrival_window:
        return None

    earliest_departure = None
    latest_departure = None
    earliest_arrival = None
    latest_arrival = None

    if departure_window:
        from fli.core.parsers import parse_time_range

        earliest_departure, latest_departure = parse_time_range(departure_window)

    if arrival_window:
        from fli.core.parsers import parse_time_range

        earliest_arrival, latest_arrival = parse_time_range(arrival_window)

    return TimeRestrictions(
        earliest_departure=earliest_departure,
        latest_departure=latest_departure,
        earliest_arrival=earliest_arrival,
        latest_arrival=latest_arrival,
    )


def build_flight_segments(
    origin: Airport | list[Airport],
    destination: Airport | list[Airport],
    departure_date: str,
    return_date: str | None = None,
    time_restrictions: TimeRestrictions | None = None,
) -> tuple[list[FlightSegment], TripType]:
    """Build flight segments for a search request.

    Args:
        origin: Departure airport, or list of airports for a metro / "any of"
            search (e.g. ``[LHR, LGW, STN]`` for London).
        destination: Arrival airport, or list of airports for a metro search.
        departure_date: Outbound travel date in YYYY-MM-DD format
        return_date: Return travel date in YYYY-MM-DD format (optional)
        time_restrictions: Time restrictions to apply to segments

    Returns:
        Tuple of (list of FlightSegment objects, TripType)

    """
    origins = _as_airport_list(origin)
    destinations = _as_airport_list(destination)

    departure_date = normalize_date(departure_date)

    segments = [
        FlightSegment(
            departure_airport=[[a, 0] for a in origins],
            arrival_airport=[[a, 0] for a in destinations],
            travel_date=departure_date,
            time_restrictions=time_restrictions,
        )
    ]

    trip_type = TripType.ONE_WAY

    if return_date:
        return_date = normalize_date(return_date)
        trip_type = TripType.ROUND_TRIP
        segments.append(
            FlightSegment(
                departure_airport=[[a, 0] for a in destinations],
                arrival_airport=[[a, 0] for a in origins],
                travel_date=return_date,
                time_restrictions=time_restrictions,
            )
        )

    return segments, trip_type


def build_multi_city_segments(
    legs: list[tuple[Airport | list[Airport], Airport | list[Airport], str]],
    time_restrictions: TimeRestrictions | None = None,
) -> tuple[list[FlightSegment], TripType]:
    """Build flight segments for a multi-city search.

    Args:
        legs: List of (origin, destination, date) tuples for each leg.  Each
            origin or destination may be a single :class:`Airport` or a list
            of airports (for IATA metropolitan / "any of" searches).
        time_restrictions: Time restrictions to apply to all segments

    Returns:
        Tuple of (list of FlightSegment objects, TripType.MULTI_CITY)

    Note:
        - Legs are *not* required to be continuous (destination[N] does not
          have to equal origin[N+1]).  Google Flights itself supports
          discontinuous multi-city — e.g. arriving CTU and departing PVG with
          a positioning gap — and prices the entire trip as one fare.
        - However, multi-city searches with distinct city pairs hit an
          intermittently slow Google Flights endpoint (the "continuation"
          ``GetShoppingResults`` call after a leg is selected).  A timeout
          is *not* the same as "no flights"; the MCP layer surfaces this as
          ``error_kind: "timeout"`` so callers can retry rather than
          concluding the routing is impossible.  Round-trip-style multi-city
          (same origin and final destination) is the most reliable shape.

    """
    segments = [
        FlightSegment(
            departure_airport=[[a, 0] for a in _as_airport_list(origin)],
            arrival_airport=[[a, 0] for a in _as_airport_list(destination)],
            travel_date=normalize_date(date),
            time_restrictions=time_restrictions,
        )
        for origin, destination, date in legs
    ]

    return segments, TripType.MULTI_CITY


def build_date_search_segments(
    origin: Airport | list[Airport],
    destination: Airport | list[Airport],
    start_date: str,
    trip_duration: int | None = None,
    is_round_trip: bool = False,
    time_restrictions: TimeRestrictions | None = None,
) -> tuple[list[FlightSegment], TripType]:
    """Build flight segments for a date range search.

    Args:
        origin: Departure airport, or list of airports for a metro search.
        destination: Arrival airport, or list of airports for a metro search.
        start_date: Start date of the search range in YYYY-MM-DD format
        trip_duration: Duration of the trip in days (for round trips)
        is_round_trip: Whether to search for round-trip flights
        time_restrictions: Time restrictions to apply to segments

    Returns:
        Tuple of (list of FlightSegment objects, TripType)

    """
    origins = _as_airport_list(origin)
    destinations = _as_airport_list(destination)

    start_date = normalize_date(start_date)

    segments = [
        FlightSegment(
            departure_airport=[[a, 0] for a in origins],
            arrival_airport=[[a, 0] for a in destinations],
            travel_date=start_date,
            time_restrictions=time_restrictions,
        )
    ]

    trip_type = TripType.ONE_WAY

    if is_round_trip:
        trip_type = TripType.ROUND_TRIP
        return_date = (
            datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=trip_duration or 3)
        ).strftime("%Y-%m-%d")

        segments.append(
            FlightSegment(
                departure_airport=[[a, 0] for a in destinations],
                arrival_airport=[[a, 0] for a in origins],
                travel_date=return_date,
                time_restrictions=time_restrictions,
            )
        )

    return segments, trip_type
