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
    """Regression tests for the empty-body retry path in ``_do_single_search``.

    ``_do_single_search`` must handle a *truly* empty HTTP body without
    raising — both the anti-XSSI-prefix-only case and the all-whitespace
    case — and the multi-leg retry path must actually fire on those bodies
    (not be bypassed by ``json.loads('')`` raising first).

    Regression test for the CodeRabbit review on leethree/fli#2.
    """

    @staticmethod
    def _multi_leg_filters() -> FlightSearchFilters:
        """Build round-trip filters used to exercise the multi-leg retry path.

        Returns:
            A round-trip JFK↔LHR ``FlightSearchFilters`` with a
            ``trip_type`` that triggers the multi-leg retry-on-empty rule.

        """
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
        """Build one-way filters used to verify retry is *not* triggered.

        Returns:
            A one-way JFK→LHR ``FlightSearchFilters``; one-way trips
            should accept an empty body on the first attempt rather than
            paying the retry latency for what's almost always a real
            "no flights" answer.

        """
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

    @staticmethod
    def _make_response(text: str) -> "MagicMock":
        """Create a minimal response-shaped mock with a configurable body.

        Args:
            text: The string the mock should expose on ``.text``.

        Returns:
            A ``MagicMock`` whose ``.text`` attribute returns *text*; all
            other attributes/methods are auto-stubbed by ``MagicMock``.

        """
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.text = text
        return resp

    def test_truly_empty_body_does_not_raise_on_multi_leg(self) -> None:
        """An HTTP 200 with an empty body must not raise ``JSONDecodeError``.

        The empty-body guard returns ``None`` cleanly instead of
        propagating a parser exception.  Application-layer retry is
        disabled by default to fit MCP transport budgets — see the
        ``_EMPTY_RETRIES_MULTI_LEG`` constant; the MCP layer's
        retry-with-hint pattern handles cold-cache transients better
        because each retry gets a fresh transport budget.
        """
        from unittest.mock import patch

        sf = SearchFlights()
        empty_response = self._make_response("")
        with patch.object(sf.client, "post", return_value=empty_response) as mock_post:
            result = sf._do_single_search(self._multi_leg_filters())

        assert result is None
        # Single attempt by default; the parser must not raise on empty.
        assert mock_post.call_count == 1

    def test_anti_xssi_prefix_only_body_does_not_raise(self) -> None:
        """A body containing only the anti-XSSI prefix is still 'empty'.

        Google's frontend prefixes responses with ``)]}'`` to defeat
        cross-site script inclusion; a body that's *only* the prefix is
        no payload at all and must be treated identically to a
        zero-length body — no parser exception, ``None`` returned.
        """
        from unittest.mock import patch

        sf = SearchFlights()
        prefix_only = self._make_response(")]}'")
        with patch.object(sf.client, "post", return_value=prefix_only) as mock_post:
            result = sf._do_single_search(self._multi_leg_filters())

        assert result is None
        assert mock_post.call_count == 1

    def test_one_way_does_not_retry_on_empty_body(self) -> None:
        """One-way queries return ``None`` cleanly on empty body — single attempt.

        The one-way endpoint is reliable enough that an empty response
        is almost always a real "no flights" outcome; this test guards
        the single-attempt invariant against a regression that would
        wake retry on one-way (e.g. a future change to the
        ``is_multi_leg`` check).
        """
        from unittest.mock import patch

        sf = SearchFlights()
        empty_response = self._make_response("")
        with patch.object(sf.client, "post", return_value=empty_response) as mock_post:
            result = sf._do_single_search(self._one_way_filters())

        assert result is None
        assert mock_post.call_count == 1

    def test_explicit_retry_override_enables_recovery(self) -> None:
        """Callers outside MCP (e.g. the CLI) can opt back into retry.

        The ``_EMPTY_RETRIES_MULTI_LEG`` constant is exposed so a
        caller with a more relaxed wall-time budget can enable empty-
        body retries.  This test pins the recovery semantics so a
        future refactor doesn't silently break that escape hatch.
        """
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
        with (
            patch.object(SearchFlights, "_EMPTY_RETRIES_MULTI_LEG", 1),
            patch.object(sf.client, "post", side_effect=responses) as mock_post,
        ):
            result = sf._do_single_search(self._multi_leg_filters())

        assert result == []
        assert mock_post.call_count == 2


class TestRecursionTimeoutPolicy:
    """``SearchFlights.search``'s per-iteration timeout handling must
    distinguish 'every continuation timed out' (real API failure, raise)
    from 'one branch timed out, others legitimately had no flights'
    (real zero-result, return empty).

    Regression test for the CodeRabbit review on leethree/fli#2 — the
    previous condition ``timeout_skipped > 0 and not flight_combos``
    raised on mixed outcomes and surfaced a false-positive timeout.
    """

    @staticmethod
    def _round_trip_filters() -> FlightSearchFilters:
        """Build round-trip filters that exercise the recursion path.

        Returns:
            A round-trip ``FlightSearchFilters`` whose ``trip_type``
            triggers the multi-leg recursion in ``SearchFlights.search``.

        """
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
    def _flight_with_legs(price: float = 100.0) -> "MagicMock":
        """Build a minimal ``FlightResult``-shaped mock the recursion can deepcopy.

        ``SearchFlights.search`` calls ``deepcopy(filters)`` on each
        iteration and assigns the picked option to
        ``flight_segments[N].selected_flight``.  The mock just needs to
        survive ``deepcopy`` and not interfere with the recursion's
        bookkeeping (it isn't introspected).

        Returns:
            A ``MagicMock`` that walks like a ``FlightResult`` enough for
            the recursion's purposes.

        """
        from unittest.mock import MagicMock

        f = MagicMock()
        f.price = price
        return f

    def test_mixed_timeout_and_empty_returns_empty_not_raise(self) -> None:
        """A mix of one timeout and several empty continuations must return ``[]``.

        Without this distinction, the previous code raised on any
        ``timeout_skipped > 0 and not flight_combos`` outcome — which
        wrongly told MCP callers to retry queries that had already
        exhaustively answered "no onward flights on those branches".
        """
        from unittest.mock import patch

        sf = SearchFlights()
        leg1_options = [self._flight_with_legs(p) for p in (100.0, 200.0, 300.0)]
        timeout_exc = Exception("Timeout: Operation timed out")

        # ``search`` recurses into itself; mock the recursive call so the
        # outer entry is the real method (so we exercise the real
        # post-loop ``raise_condition`` from production code).  The
        # recursive calls get a fake that emits one timeout, two empties.
        original_search = SearchFlights.search

        recursive_call_count = {"n": 0}

        def fake_recursive(self_, filters, top_n=5):
            # The outer call sees no selected_flight; let it through to
            # the real implementation.
            if all(s.selected_flight is None for s in filters.flight_segments):
                # Bypass the real network call by feeding leg-1 results
                # directly via ``_do_single_search`` patch (below).
                return original_search(self_, filters, top_n)
            # Recursive (continuation) call: deterministic outcome.
            i = recursive_call_count["n"]
            recursive_call_count["n"] += 1
            if i == 0:
                raise timeout_exc
            return None  # legitimate "no onward flights"

        with patch.object(SearchFlights, "search", autospec=True, side_effect=fake_recursive):
            with patch.object(sf, "_do_single_search", return_value=leg1_options):
                result = sf.search(self._round_trip_filters(), top_n=3)

        # 3 continuations attempted, 1 timed out, 2 empties → flight_combos
        # is empty but the trip is real "no flights", so we return [] cleanly.
        assert result == []
        assert recursive_call_count["n"] == 3

    def test_all_continuations_timeout_does_raise(self) -> None:
        """When *every* continuation times out, surface the timeout.

        The outer call must raise so the MCP layer can report a transient
        backend stall rather than a misleading "no flights" result.
        """
        from unittest.mock import patch

        sf = SearchFlights()
        leg1_options = [self._flight_with_legs(p) for p in (100.0, 200.0, 300.0)]
        timeout_exc = Exception("Timeout: Operation timed out")

        original_search = SearchFlights.search

        def fake_recursive(self_, filters, top_n=5):
            if all(s.selected_flight is None for s in filters.flight_segments):
                return original_search(self_, filters, top_n)
            raise timeout_exc

        with patch.object(SearchFlights, "search", autospec=True, side_effect=fake_recursive):
            with patch.object(sf, "_do_single_search", return_value=leg1_options):
                with pytest.raises(Exception) as exc_info:
                    sf.search(self._round_trip_filters(), top_n=3)

        # The outer ``except Exception`` wrapper rebrands the message,
        # but the underlying "All N timed out" text must survive.
        assert "timed out" in str(exc_info.value).lower()
