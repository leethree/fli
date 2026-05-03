import pytest

from fli.core.builders import (
    build_date_search_segments,
    build_flight_segments,
    build_multi_city_segments,
    normalize_date,
)
from fli.models import Airport, TripType


class TestNormalizeDate:
    """Tests for normalize_date."""

    def test_already_padded(self):
        assert normalize_date("2027-04-02") == "2027-04-02"

    def test_single_digit_month_and_day(self):
        assert normalize_date("2027-4-2") == "2027-04-02"

    def test_single_digit_day(self):
        assert normalize_date("2027-12-5") == "2027-12-05"

    def test_single_digit_month(self):
        assert normalize_date("2027-1-15") == "2027-01-15"

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            normalize_date("not-a-date")

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            normalize_date("2027-13-01")


class TestBuildFlightSegments:
    """Tests for date normalization in build_flight_segments."""

    def test_normalizes_departure_date(self):
        segments, _ = build_flight_segments(
            origin=Airport.JFK,
            destination=Airport.LAX,
            departure_date="2027-1-15",
        )
        assert segments[0].travel_date == "2027-01-15"

    def test_normalizes_return_date(self):
        segments, trip_type = build_flight_segments(
            origin=Airport.JFK,
            destination=Airport.LAX,
            departure_date="2027-1-15",
            return_date="2027-1-22",
        )
        assert trip_type == TripType.ROUND_TRIP
        assert segments[0].travel_date == "2027-01-15"
        assert segments[1].travel_date == "2027-01-22"


class TestBuildDateSearchSegments:
    """Tests for date normalization in build_date_search_segments."""

    def test_normalizes_start_date(self):
        segments, _ = build_date_search_segments(
            origin=Airport.JFK,
            destination=Airport.LAX,
            start_date="2027-1-15",
        )
        assert segments[0].travel_date == "2027-01-15"

    def test_normalizes_start_date_round_trip(self):
        segments, trip_type = build_date_search_segments(
            origin=Airport.JFK,
            destination=Airport.LAX,
            start_date="2027-1-15",
            is_round_trip=True,
            trip_duration=7,
        )
        assert trip_type == TripType.ROUND_TRIP
        assert segments[0].travel_date == "2027-01-15"
        assert segments[1].travel_date == "2027-01-22"


class TestMultipleAirportsPerSlot:
    """Builders should accept multiple airports per slot for metro searches."""

    def test_flight_segments_accept_origin_list(self):
        segments, _ = build_flight_segments(
            origin=[Airport.LHR, Airport.LGW, Airport.STN],
            destination=Airport.JFK,
            departure_date="2027-06-01",
        )
        assert segments[0].departure_airport == [
            [Airport.LHR, 0],
            [Airport.LGW, 0],
            [Airport.STN, 0],
        ]
        assert segments[0].arrival_airport == [[Airport.JFK, 0]]

    def test_flight_segments_accept_destination_list(self):
        segments, _ = build_flight_segments(
            origin=Airport.JFK,
            destination=[Airport.LHR, Airport.LGW],
            departure_date="2027-06-01",
        )
        assert segments[0].arrival_airport == [[Airport.LHR, 0], [Airport.LGW, 0]]

    def test_round_trip_swaps_lists(self):
        segments, trip_type = build_flight_segments(
            origin=[Airport.LHR, Airport.LGW],
            destination=[Airport.JFK, Airport.EWR],
            departure_date="2027-06-01",
            return_date="2027-06-08",
        )
        assert trip_type == TripType.ROUND_TRIP
        assert segments[0].departure_airport == [[Airport.LHR, 0], [Airport.LGW, 0]]
        assert segments[0].arrival_airport == [[Airport.JFK, 0], [Airport.EWR, 0]]
        assert segments[1].departure_airport == [[Airport.JFK, 0], [Airport.EWR, 0]]
        assert segments[1].arrival_airport == [[Airport.LHR, 0], [Airport.LGW, 0]]

    def test_date_search_accepts_lists(self):
        segments, _ = build_date_search_segments(
            origin=[Airport.LHR, Airport.LGW],
            destination=[Airport.JFK, Airport.EWR],
            start_date="2027-06-01",
        )
        assert segments[0].departure_airport == [[Airport.LHR, 0], [Airport.LGW, 0]]
        assert segments[0].arrival_airport == [[Airport.JFK, 0], [Airport.EWR, 0]]

    def test_multi_city_accepts_lists(self):
        segments, trip_type = build_multi_city_segments(
            legs=[
                ([Airport.LHR, Airport.LGW], Airport.JFK, "2027-06-01"),
                (Airport.JFK, [Airport.CDG, Airport.ORY], "2027-06-05"),
            ],
        )
        assert trip_type == TripType.MULTI_CITY
        assert segments[0].departure_airport == [[Airport.LHR, 0], [Airport.LGW, 0]]
        assert segments[1].arrival_airport == [[Airport.CDG, 0], [Airport.ORY, 0]]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_flight_segments(
                origin=[],
                destination=Airport.JFK,
                departure_date="2027-06-01",
            )
