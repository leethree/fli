"""Shared parsing utilities for flight search parameters.

This module provides parsing functions used by both the CLI and MCP interfaces
to convert user input into domain model objects.
"""

from enum import Enum
from typing import TypeVar

from fli.models import Airline, Airport, EmissionsFilter, MaxStops, SeatType, SortBy
from fli.models.city import is_city_code, resolve_city

T = TypeVar("T", bound=Enum)


class ParseError(ValueError):
    """Error raised when parsing fails."""

    pass


def resolve_enum(enum_cls: type[T], name: str) -> T:
    """Resolve an enum member by name with normalized errors.

    Args:
        enum_cls: The enum class to resolve from
        name: The name of the enum member (case-insensitive)

    Returns:
        The resolved enum member

    Raises:
        ParseError: If the name is not a valid enum member

    """
    try:
        return getattr(enum_cls, name.upper())
    except AttributeError as e:
        valid_values = [m.name for m in enum_cls]
        raise ParseError(
            f"Invalid {enum_cls.__name__} value: '{name}'. Valid values: {', '.join(valid_values)}"
        ) from e


def resolve_airport(code: str) -> Airport:
    """Resolve an airport code to an Airport enum.

    Args:
        code: IATA airport code (e.g., 'JFK', 'LHR')

    Returns:
        The corresponding Airport enum member

    Raises:
        ParseError: If the code is not a valid airport

    """
    try:
        return getattr(Airport, code.upper())
    except AttributeError as e:
        raise ParseError(f"Invalid airport code: '{code}'") from e


def resolve_origin_or_city(value: str | list[str]) -> list[Airport]:
    """Resolve an airport code, IATA city/metro code, or list of either.

    Accepts a single IATA code (airport like ``"JFK"`` or metropolitan like
    ``"LON"``) or a list mixing both forms.  Metropolitan codes are expanded
    into their constituent airports.  When the input is a list, results are
    deduplicated while preserving first-seen order.

    Args:
        value: A single IATA code or a list of codes.

    Returns:
        Non-empty list of resolved :class:`Airport` enum members.

    Raises:
        ParseError: If a code is neither a known city nor a valid airport,
            or if the input list is empty.

    """
    if isinstance(value, str):
        codes = [value]
    else:
        codes = [c for c in value if c and c.strip()]
        if not codes:
            raise ParseError("At least one airport or city code is required")

    seen: set[Airport] = set()
    resolved: list[Airport] = []
    for raw in codes:
        code = raw.strip().upper()
        if not code:
            continue
        if is_city_code(code):
            airports = resolve_city(code)
        else:
            try:
                airports = [getattr(Airport, code)]
            except AttributeError as e:
                raise ParseError(f"Invalid airport or city code: '{raw}'") from e
        for airport in airports:
            if airport not in seen:
                seen.add(airport)
                resolved.append(airport)

    if not resolved:
        raise ParseError("At least one airport or city code is required")

    return resolved


def parse_airlines(codes: list[str] | None) -> list[Airline] | None:
    """Parse a list of airline codes into Airline enums.

    Args:
        codes: List of IATA airline codes (e.g., ['BA', 'KL'])

    Returns:
        List of Airline enums, or None if input is empty

    Raises:
        ParseError: If any code is not a valid airline

    """
    if not codes:
        return None

    airlines = []
    for code in codes:
        code = code.strip().upper()
        if not code:
            continue
        # Airline codes starting with a digit need an underscore prefix
        # to match the Airline enum member names (e.g., "3F" -> "_3F")
        enum_key = f"_{code}" if code[0].isdigit() else code
        try:
            airline = getattr(Airline, enum_key)
            airlines.append(airline)
        except AttributeError as e:
            raise ParseError(f"Invalid airline code: '{code}'") from e

    return airlines if airlines else None


def parse_max_stops(stops: str) -> MaxStops:
    """Parse a stops parameter into a MaxStops enum.

    Accepts both string names (ANY, NON_STOP, ONE_STOP, TWO_PLUS_STOPS)
    and integer values (0, 1, 2+).

    Args:
        stops: Stops value as string or integer representation

    Returns:
        The corresponding MaxStops enum member

    Raises:
        ParseError: If the value is not valid

    """
    # Mapping for user-friendly names
    stops_map = {
        "ANY": MaxStops.ANY,
        "NON_STOP": MaxStops.NON_STOP,
        "NONSTOP": MaxStops.NON_STOP,
        "ONE_STOP": MaxStops.ONE_STOP_OR_FEWER,
        "ONE_STOP_OR_FEWER": MaxStops.ONE_STOP_OR_FEWER,
        "TWO_PLUS_STOPS": MaxStops.TWO_OR_FEWER_STOPS,
        "TWO_OR_FEWER_STOPS": MaxStops.TWO_OR_FEWER_STOPS,
    }

    # Try as integer first
    try:
        stops_int = int(stops)
        if stops_int == 0:
            return MaxStops.NON_STOP
        elif stops_int == 1:
            return MaxStops.ONE_STOP_OR_FEWER
        elif stops_int >= 2:
            return MaxStops.TWO_OR_FEWER_STOPS
        else:
            return MaxStops.ANY
    except ValueError:
        pass

    # Try as string name
    upper_stops = stops.upper()
    if upper_stops in stops_map:
        return stops_map[upper_stops]

    raise ParseError(
        f"Invalid max_stops value: '{stops}'. "
        f"Valid values: ANY, NON_STOP, ONE_STOP, TWO_PLUS_STOPS, or 0/1/2"
    )


def parse_cabin_class(cabin_class: str) -> SeatType:
    """Parse a cabin class string into a SeatType enum.

    Args:
        cabin_class: Cabin class name (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)

    Returns:
        The corresponding SeatType enum member

    Raises:
        ParseError: If the value is not valid

    """
    try:
        return getattr(SeatType, cabin_class.upper())
    except AttributeError as e:
        valid_values = [m.name for m in SeatType]
        raise ParseError(
            f"Invalid cabin_class value: '{cabin_class}'. Valid values: {', '.join(valid_values)}"
        ) from e


def parse_sort_by(sort_by: str) -> SortBy:
    """Parse a sort_by string into a SortBy enum.

    Args:
        sort_by: Sort option (TOP_FLIGHTS, BEST, CHEAPEST,
            DEPARTURE_TIME, ARRIVAL_TIME, DURATION, EMISSIONS)

    Returns:
        The corresponding SortBy enum member

    Raises:
        ParseError: If the value is not valid

    """
    try:
        return getattr(SortBy, sort_by.upper())
    except AttributeError as e:
        valid_values = [m.name for m in SortBy]
        raise ParseError(
            f"Invalid sort_by value: '{sort_by}'. Valid values: {', '.join(valid_values)}"
        ) from e


def parse_emissions(emissions: str) -> EmissionsFilter:
    """Parse an emissions filter string into an EmissionsFilter enum.

    Args:
        emissions: Emissions filter (ALL, LESS)

    Returns:
        The corresponding EmissionsFilter enum member

    Raises:
        ParseError: If the value is not valid

    """
    return resolve_enum(EmissionsFilter, emissions)


def parse_time_range(time_range: str) -> tuple[int, int]:
    """Parse a time range string into start and end hours.

    Args:
        time_range: Time range in 'HH-HH' format (e.g., '6-20')

    Returns:
        Tuple of (start_hour, end_hour)

    Raises:
        ParseError: If the format is invalid

    """
    try:
        parts = time_range.split("-")
        if len(parts) != 2:
            raise ValueError("Invalid format")

        start_hour = int(parts[0].strip())
        end_hour = int(parts[1].strip())

        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError("Hours must be between 0 and 23")

        return start_hour, end_hour
    except (ValueError, AttributeError) as e:
        raise ParseError(
            f"Invalid time range format: '{time_range}'. Expected 'HH-HH' (e.g., '6-20')"
        ) from e
