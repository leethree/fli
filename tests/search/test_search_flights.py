"""Tests for Search class."""

from datetime import datetime, timedelta

import pytest
from tenacity import retry, stop_after_attempt, wait_exponential

from fli.models import (
    Airport,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
)
from fli.models.google_flights.base import TripType
from fli.search import SearchFlights


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def search_with_retry(search: SearchFlights, search_params):
    """Search with retry logic for flaky API responses."""
    results = search.search(search_params)
    if not results:
        raise ValueError("Empty results, retrying...")
    return results


@pytest.fixture
def search():
    """Create a reusable Search instance."""
    return SearchFlights()


@pytest.fixture
def basic_search_params():
    """Create basic search params for testing."""
    today = datetime.now()
    future_date = today + timedelta(days=30)
    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=1,
            children=0,
            infants_in_seat=0,
            infants_on_lap=0,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.PHX, 0]],
                arrival_airport=[[Airport.SFO, 0]],
                travel_date=future_date.strftime("%Y-%m-%d"),
            )
        ],
        stops=MaxStops.NON_STOP,
        seat_type=SeatType.ECONOMY,
        sort_by=SortBy.CHEAPEST,
        show_all_results=False,
    )


@pytest.fixture
def complex_search_params():
    """Create more complex search params for testing."""
    today = datetime.now()
    future_date = today + timedelta(days=60)
    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=2,
            children=1,
            infants_in_seat=0,
            infants_on_lap=1,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=future_date.strftime("%Y-%m-%d"),
            )
        ],
        stops=MaxStops.ONE_STOP_OR_FEWER,
        seat_type=SeatType.FIRST,
        sort_by=SortBy.TOP_FLIGHTS,
        show_all_results=False,
    )


@pytest.fixture
def round_trip_search_params():
    """Create basic round trip search params for testing."""
    today = datetime.now()
    outbound_date = today + timedelta(days=30)
    return_date = outbound_date + timedelta(days=7)

    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=1,
            children=0,
            infants_in_seat=0,
            infants_on_lap=0,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.SFO, 0]],
                arrival_airport=[[Airport.JFK, 0]],
                travel_date=outbound_date.strftime("%Y-%m-%d"),
            ),
            FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.SFO, 0]],
                travel_date=return_date.strftime("%Y-%m-%d"),
            ),
        ],
        stops=MaxStops.NON_STOP,
        seat_type=SeatType.ECONOMY,
        sort_by=SortBy.CHEAPEST,
        trip_type=TripType.ROUND_TRIP,
        show_all_results=False,
    )


@pytest.fixture
def complex_round_trip_params():
    """Create more complex round trip search params for testing."""
    today = datetime.now()
    outbound_date = today + timedelta(days=60)
    return_date = outbound_date + timedelta(days=14)

    return FlightSearchFilters(
        passenger_info=PassengerInfo(
            adults=2,
            children=1,
            infants_in_seat=0,
            infants_on_lap=1,
        ),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.LAX, 0]],
                arrival_airport=[[Airport.ORD, 0]],
                travel_date=outbound_date.strftime("%Y-%m-%d"),
            ),
            FlightSegment(
                departure_airport=[[Airport.ORD, 0]],
                arrival_airport=[[Airport.LAX, 0]],
                travel_date=return_date.strftime("%Y-%m-%d"),
            ),
        ],
        stops=MaxStops.ONE_STOP_OR_FEWER,
        seat_type=SeatType.BUSINESS,
        sort_by=SortBy.TOP_FLIGHTS,
        trip_type=TripType.ROUND_TRIP,
        show_all_results=False,
    )


@pytest.mark.parametrize(
    "search_params_fixture",
    [
        "basic_search_params",
        "complex_search_params",
    ],
)
def test_search_functionality(search, search_params_fixture, request):
    """Test flight search functionality with different data sets."""
    search_params = request.getfixturevalue(search_params_fixture)
    results = search.search(search_params)
    assert isinstance(results, list)


def test_multiple_searches(search, basic_search_params, complex_search_params):
    """Test performing multiple searches with the same Search instance."""
    # First search
    results1 = search.search(basic_search_params)
    assert isinstance(results1, list)

    # Second search with different data
    results2 = search.search(complex_search_params)
    assert isinstance(results2, list)

    # Third search reusing first search data
    results3 = search.search(basic_search_params)
    assert isinstance(results3, list)


# TODO: These round-trip tests hit the live Google Flights API with multiple
# sequential requests (outbound + return for each result), causing frequent
# timeouts on CI runners. They should be refactored to mock the HTTP client
# instead of making real API calls. See GitHub issue for follow-up.
#
# def test_basic_round_trip_search(search, round_trip_search_params):
# def test_complex_round_trip_search(search, complex_round_trip_params):
# def test_round_trip_with_selected_outbound(search, round_trip_search_params):
# def test_round_trip_result_structure(search, search_params_fixture, request):


class TestParsePriceInfo:
    """Tests for _parse_price_info method handling missing/malformed price data."""

    def test_parse_price_info_valid_data(self):
        """Test _parse_price_info with valid price data."""
        data = [None, [[100, 200, 299.99]]]
        price, currency = SearchFlights._parse_price_info(data)
        assert price == 299.99
        assert currency is None

    def test_parse_price_info_empty_inner_list(self):
        """Test _parse_price_info returns 0.0 when inner price list is empty."""
        data = [None, [[]]]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_empty_outer_list(self):
        """Test _parse_price_info returns 0.0 when outer price list is empty."""
        data = [None, []]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_none_price_section(self):
        """Test _parse_price_info returns 0.0 when price section is None."""
        data = [None, None]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_missing_price_section(self):
        """Test _parse_price_info returns 0.0 when data has no price section."""
        data = [None]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_price_info_inner_list_none(self):
        """Test _parse_price_info returns 0.0 when inner list is None."""
        data = [None, [None]]
        price, _ = SearchFlights._parse_price_info(data)
        assert price == 0.0

    def test_parse_currency_from_live_price_token(self):
        """_parse_currency should decode the returned currency from a live token sample."""
        data = [
            None,
            [
                [None, 118],
                "CjRIQktCNmV1UjNqNjhBR043X0FCRy0tLS0tLS0tLS12dGpkN0FBQUFBR25JcWZNS2pGTTBBEgZV"
                "QTIyMDkaCgjcWxACGgNVU0Q4HHDcWw==",
            ],
        ]
        assert SearchFlights._parse_currency(data) == "USD"

    def test_parse_price_info_combines_price_and_currency(self):
        """_parse_price_info should preserve price and extract the returned currency."""
        data = [
            None,
            [
                [None, 118],
                "CjRIQktCNmV1UjNqNjhBR043X0FCRy0tLS0tLS0tLS12dGpkN0FBQUFBR25JcWZNS2pGTTBBEgZV"
                "QTIyMDkaCgjcWxACGgNVU0Q4HHDcWw==",
            ],
        ]
        assert SearchFlights._parse_price_info(data) == (118.0, "USD")


class TestEmptyBodyRetry:
    """``_do_single_search`` must handle a *truly* empty HTTP body without
    raising — both the anti-XSSI-prefix-only case and the all-whitespace
    case — and the multi-leg retry path must actually fire on those bodies
    (not be bypassed by ``json.loads('')`` raising first).

    Regression test for the CodeRabbit review on leethree/fli#2.
    """

    @staticmethod
    def _multi_leg_filters() -> FlightSearchFilters:
        today = datetime.now()
        d1 = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        d2 = (today + timedelta(days=37)).strftime("%Y-%m-%d")
        return FlightSearchFilters(
            trip_type=TripType.ROUND_TRIP,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=[
                FlightSegment(
                    departure_airport=[[Airport.JFK, 0]],
                    arrival_airport=[[Airport.LHR, 0]],
                    travel_date=d1,
                ),
                FlightSegment(
                    departure_airport=[[Airport.LHR, 0]],
                    arrival_airport=[[Airport.JFK, 0]],
                    travel_date=d2,
                ),
            ],
        )

    @staticmethod
    def _one_way_filters() -> FlightSearchFilters:
        today = datetime.now()
        d = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        return FlightSearchFilters(
            trip_type=TripType.ONE_WAY,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=[
                FlightSegment(
                    departure_airport=[[Airport.JFK, 0]],
                    arrival_airport=[[Airport.LHR, 0]],
                    travel_date=d,
                ),
            ],
        )

    def _make_response(self, text: str):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.text = text
        return resp

    def test_truly_empty_body_does_not_raise_on_multi_leg(self):
        """An HTTP 200 with empty body must NOT raise JSONDecodeError;
        the retry must fire and ultimately return None on persistent empty.
        """
        from unittest.mock import patch

        sf = SearchFlights()
        empty_response = self._make_response("")
        with patch.object(sf.client, "post", return_value=empty_response) as mock_post:
            result = sf._do_single_search(self._multi_leg_filters())

        assert result is None
        # Multi-leg = 1 + 1 retry attempt = 2 calls total.  If the parse
        # raised before the retry guard, we'd see 1 call and an exception.
        assert mock_post.call_count == 2

    def test_anti_xssi_prefix_only_body_does_not_raise(self):
        """The legacy anti-XSSI prefix on its own is still 'empty'."""
        from unittest.mock import patch

        sf = SearchFlights()
        prefix_only = self._make_response(")]}'")
        with patch.object(sf.client, "post", return_value=prefix_only) as mock_post:
            result = sf._do_single_search(self._multi_leg_filters())

        assert result is None
        assert mock_post.call_count == 2

    def test_one_way_does_not_retry_on_empty_body(self):
        """One-way queries are reliable enough that retry-on-empty just
        slows down legitimate "no flights" results — single attempt only.
        """
        from unittest.mock import patch

        sf = SearchFlights()
        empty_response = self._make_response("")
        with patch.object(sf.client, "post", return_value=empty_response) as mock_post:
            result = sf._do_single_search(self._one_way_filters())

        assert result is None
        assert mock_post.call_count == 1

    def test_retry_recovers_when_second_attempt_returns_data(self):
        """If the first body is empty but the retry succeeds, the parsed
        flights must be returned (the retry isn't decorative)."""
        from unittest.mock import patch

        # Minimal valid envelope.  Outer JSON is a list whose ``[0][2]``
        # is itself a JSON-encoded string; that string decodes to a list
        # where indices 2 and 3 are ``None`` so the ``isinstance(...,
        # list)`` filter in ``_do_single_search`` skips them and we get
        # an empty flights list rather than an IndexError.
        warm_payload = (
            ")]}'\n"
            + '[[null, null, "[null, null, null, null]"]]'
        )
        sf = SearchFlights()
        responses = [self._make_response(""), self._make_response(warm_payload)]
        with patch.object(sf.client, "post", side_effect=responses) as mock_post:
            result = sf._do_single_search(self._multi_leg_filters())

        # Empty inner list slots → zero flights, but non-None means we
        # actually parsed the warm response and didn't bail early.
        assert result == []
        assert mock_post.call_count == 2
