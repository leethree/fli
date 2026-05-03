"""Tests for core parser utilities."""

import pytest

from fli.core.parsers import (
    ParseError,
    parse_airlines,
    parse_emissions,
    parse_sort_by,
    resolve_origin_or_city,
)
from fli.models import Airline, Airport, EmissionsFilter, SortBy


class TestParseEmissions:
    """Tests for parse_emissions."""

    def test_all(self):
        assert parse_emissions("ALL") == EmissionsFilter.ALL

    def test_less(self):
        assert parse_emissions("LESS") == EmissionsFilter.LESS

    def test_case_insensitive(self):
        assert parse_emissions("all") == EmissionsFilter.ALL
        assert parse_emissions("Less") == EmissionsFilter.LESS

    def test_invalid(self):
        with pytest.raises(ParseError, match="Invalid EmissionsFilter"):
            parse_emissions("NONE")

    def test_invalid_random(self):
        with pytest.raises(ParseError, match="Invalid EmissionsFilter"):
            parse_emissions("HIGH")


class TestParseAirlinesWithAlliances:
    """Tests for parse_airlines with alliance codes."""

    def test_alliance_star_alliance(self):
        result = parse_airlines(["STAR_ALLIANCE"])
        assert result == [Airline.STAR_ALLIANCE]

    def test_alliance_oneworld(self):
        result = parse_airlines(["ONEWORLD"])
        assert result == [Airline.ONEWORLD]

    def test_alliance_skyteam(self):
        result = parse_airlines(["SKYTEAM"])
        assert result == [Airline.SKYTEAM]

    def test_alliance_mixed_with_airlines(self):
        result = parse_airlines(["STAR_ALLIANCE", "AA"])
        assert Airline.STAR_ALLIANCE in result
        assert Airline.AA in result


class TestParseSortBy:
    """Tests for parse_sort_by with updated enum values."""

    def test_top_flights(self):
        assert parse_sort_by("TOP_FLIGHTS") == SortBy.TOP_FLIGHTS
        assert SortBy.TOP_FLIGHTS.value == 0

    def test_best(self):
        assert parse_sort_by("BEST") == SortBy.BEST
        assert SortBy.BEST.value == 1

    def test_cheapest(self):
        assert parse_sort_by("CHEAPEST") == SortBy.CHEAPEST
        assert SortBy.CHEAPEST.value == 2

    def test_emissions(self):
        assert parse_sort_by("EMISSIONS") == SortBy.EMISSIONS
        assert SortBy.EMISSIONS.value == 6

    def test_invalid(self):
        with pytest.raises(ParseError, match="Invalid sort_by value"):
            parse_sort_by("NONE")


class TestResolveOriginOrCity:
    """Tests for resolve_origin_or_city — IATA city/metro support."""

    def test_single_airport_returns_one_element_list(self):
        assert resolve_origin_or_city("JFK") == [Airport.JFK]

    def test_single_airport_case_insensitive(self):
        assert resolve_origin_or_city("jfk") == [Airport.JFK]

    def test_london_metro_expands_to_six_airports(self):
        result = resolve_origin_or_city("LON")
        assert result == [
            Airport.LHR,
            Airport.LGW,
            Airport.STN,
            Airport.LCY,
            Airport.LTN,
            Airport.SEN,
        ]

    def test_nyc_metro_expands(self):
        assert resolve_origin_or_city("NYC") == [Airport.JFK, Airport.LGA, Airport.EWR]

    def test_city_code_case_insensitive(self):
        assert resolve_origin_or_city("nyc") == resolve_origin_or_city("NYC")

    def test_shanghai_metro_includes_self_named_airport(self):
        # SHA is both a city code and an airport — city expansion wins.
        result = resolve_origin_or_city("SHA")
        assert Airport.PVG in result
        assert Airport.SHA in result
        assert len(result) == 2

    def test_explicit_array_passthrough(self):
        assert resolve_origin_or_city(["LHR", "LGW"]) == [Airport.LHR, Airport.LGW]

    def test_array_mixes_city_and_airport(self):
        result = resolve_origin_or_city(["LON", "CDG"])
        assert Airport.LHR in result
        assert Airport.LGW in result
        assert Airport.CDG in result

    def test_array_dedupes_overlap(self):
        # LON expands to LHR + others; passing LHR explicitly shouldn't duplicate.
        result = resolve_origin_or_city(["LON", "LHR"])
        assert result.count(Airport.LHR) == 1

    def test_array_dedupes_two_metros(self):
        result = resolve_origin_or_city(["LON", "LON"])
        assert len(result) == 6
        assert len(set(result)) == 6

    def test_array_strips_whitespace(self):
        assert resolve_origin_or_city([" LHR ", "lgw"]) == [Airport.LHR, Airport.LGW]

    def test_unknown_code_raises_parse_error(self):
        with pytest.raises(ParseError, match="Invalid airport or city code"):
            resolve_origin_or_city("ZZZ")

    def test_unknown_in_array_raises(self):
        with pytest.raises(ParseError, match="Invalid airport or city code"):
            resolve_origin_or_city(["LHR", "ZZZ"])

    def test_empty_list_raises(self):
        with pytest.raises(ParseError):
            resolve_origin_or_city([])

    def test_list_of_blank_strings_raises(self):
        with pytest.raises(ParseError):
            resolve_origin_or_city(["", "  "])

    def test_paris_metro(self):
        assert resolve_origin_or_city("PAR") == [Airport.CDG, Airport.ORY, Airport.BVA]

    def test_tokyo_metro(self):
        assert resolve_origin_or_city("TYO") == [Airport.HND, Airport.NRT]
