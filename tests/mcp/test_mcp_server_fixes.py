"""Tests for MCP server bug fixes.

Covers:
  1. list_tools FastMCP 3.x registration and annotations
  2. Round-trip price doubling (Google Flights returns combined RT price on outbound leg)
  3. Per-leg fallback when Google's multi-city curator drops options
  4. Sticky degraded state across multi-city continuation steps
  5. Multi-airport / IATA city-code expansion at the MCP boundary
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from fli.mcp.server import (
    DateSearchParams,
    FlightSearchParams,
    MultiCityLeg,
    MultiCitySearchParams,
    _execute_flight_search,
    _execute_multi_city_step,
    _search_sessions,
    _serialize_flight_result,
    mcp,
)
from fli.models.google_flights.base import TripType

# ---------------------------------------------------------------------------
# Bug 1: list_tools — FastMCP 3.x compatibility
# ---------------------------------------------------------------------------


class TestListTools:
    """FliMCP.list_tools() should expose native FastMCP 3 tool metadata."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_registered_tools_with_annotations(self):
        """Registered tools should be listed with their schemas and annotations."""
        server = FastMCP("test")

        @server.tool(
            description="Search flights",
            annotations={"title": "Search Flights", "readOnlyHint": True, "idempotentHint": True},
        )
        def search_flights(origin: str, destination: str) -> dict[str, str]:
            return {"origin": origin, "destination": destination}

        tools = await server.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_flights"
        assert tools[0].description == "Search flights"
        assert tools[0].parameters["type"] == "object"
        assert tools[0].parameters["properties"]["origin"]["type"] == "string"
        assert tools[0].parameters["properties"]["destination"]["type"] == "string"
        assert tools[0].annotations.title == "Search Flights"
        assert tools[0].annotations.readOnlyHint is True
        assert tools[0].annotations.idempotentHint is True

    def test_tool_decorator_preserves_function_usage(self):
        """The FastMCP 3 decorator should still leave a normal callable behind."""
        server = FastMCP("test")

        @server.tool()
        def search_flights(origin: str, destination: str) -> dict[str, str]:
            return {"route": f"{origin}-{destination}"}

        assert search_flights("JFK", "LHR") == {"route": "JFK-LHR"}


class TestPrompts:
    """Module prompts should use FastMCP 3's native prompt registration."""

    @pytest.mark.asyncio
    async def test_builtin_prompts_are_registered_and_render(self):
        """The module-level prompts should be listable and render expected guidance."""
        prompts = await mcp.list_prompts()
        prompt_names = {prompt.name for prompt in prompts}

        assert "search-direct-flight" in prompt_names
        assert "find-budget-window" in prompt_names

        result = await mcp.render_prompt(
            "search-direct-flight",
            {"origin": "jfk", "destination": "lhr", "prefer_non_stop": "true"},
        )
        assert result.messages[0].content.text.startswith(
            "Use the `search_flights` tool to look for flights from JFK to LHR"
        )
        assert "NON_STOP" in result.messages[0].content.text


# ---------------------------------------------------------------------------
# Bug 2: Round-trip price doubling
# ---------------------------------------------------------------------------


def _make_leg(airport_from="TLV", airport_to="ATH"):
    leg = MagicMock()
    leg.departure_airport = airport_from
    leg.arrival_airport = airport_to
    leg.departure_datetime = None
    leg.arrival_datetime = None
    leg.duration = 145
    leg.airline = "Wizz Air"
    leg.flight_number = "W6100"
    return leg


def _make_flight(price, legs=None):
    flight = MagicMock()
    flight.price = price
    flight.currency = "USD"
    flight.legs = legs or [_make_leg()]
    return flight


class TestSerializeFlightResult:
    """_serialize_flight_result must not double round-trip prices."""

    def test_one_way_price_unchanged(self):
        """One-way flight price should pass through unchanged."""
        flight = _make_flight(price=250.0)
        result = _serialize_flight_result(flight, is_round_trip=False)
        assert result["price"] == 250.0
        assert result["currency"] == "USD"

    def test_round_trip_uses_outbound_price_only(self):
        """Round-trip price must equal outbound.price (Google already includes full RT price)."""
        outbound = _make_flight(price=454.0, legs=[_make_leg("TLV", "ATH")])
        return_flight = _make_flight(price=454.0, legs=[_make_leg("ATH", "TLV")])

        result = _serialize_flight_result((outbound, return_flight), is_round_trip=True)

        # Must NOT be 454 + 454 = 908
        assert result["price"] == 454.0, (
            f"Expected 454.0 (outbound price only), got {result['price']}. "
            "Google Flights already includes the full RT price on the outbound leg."
        )

    def test_round_trip_price_not_doubled(self):
        """Explicit check that the price is not the sum of both legs."""
        outbound = _make_flight(price=300.0)
        return_flight = _make_flight(price=300.0)

        result = _serialize_flight_result((outbound, return_flight), is_round_trip=True)

        assert result["price"] != 600.0, "Price must not be doubled"
        assert result["price"] == 300.0

    def test_round_trip_includes_legs_from_both_directions(self):
        """Round-trip result must include legs from both outbound and return flights."""
        outbound_leg = _make_leg("TLV", "ATH")
        return_leg = _make_leg("ATH", "TLV")
        outbound = _make_flight(price=454.0, legs=[outbound_leg])
        return_flight = _make_flight(price=454.0, legs=[return_leg])

        result = _serialize_flight_result((outbound, return_flight), is_round_trip=True)

        assert len(result["legs"]) == 2

    def test_round_trip_non_tuple_falls_back_to_single_flight(self):
        """If flight is not a tuple, treat it as a one-way even if is_round_trip=True."""
        flight = _make_flight(price=500.0)
        result = _serialize_flight_result(flight, is_round_trip=True)
        assert result["price"] == 500.0

    def test_multi_city_three_legs(self):
        """Multi-city (3-leg) tuple should serialize without crashing."""
        leg1 = _make_flight(price=0.0, legs=[_make_leg("JFK", "LAX")])
        leg2 = _make_flight(price=0.0, legs=[_make_leg("LAX", "ORD")])
        leg3 = _make_flight(price=750.0, legs=[_make_leg("ORD", "JFK")])

        result = _serialize_flight_result((leg1, leg2, leg3), is_round_trip=False)

        assert result["price"] == 750.0
        assert len(result["legs"]) == 3

    def test_multi_city_not_treated_as_round_trip(self):
        """A 3-leg tuple must not be treated as round-trip even if flag is True."""
        leg1 = _make_flight(price=0.0, legs=[_make_leg("JFK", "LAX")])
        leg2 = _make_flight(price=0.0, legs=[_make_leg("LAX", "ORD")])
        leg3 = _make_flight(price=900.0, legs=[_make_leg("ORD", "JFK")])

        result = _serialize_flight_result((leg1, leg2, leg3), is_round_trip=True)

        assert result["price"] == 900.0
        assert len(result["legs"]) == 3

    def test_two_tuple_without_round_trip_uses_outbound_price(self):
        """A 2-element tuple with is_round_trip=False should use outbound price."""
        outbound = _make_flight(price=350.0, legs=[_make_leg("JFK", "LAX")])
        return_flight = _make_flight(price=350.0, legs=[_make_leg("LAX", "JFK")])

        result = _serialize_flight_result((outbound, return_flight), is_round_trip=False)

        assert result["price"] == 350.0
        assert len(result["legs"]) == 2

    def test_uses_flight_currency_when_available(self):
        """Serialization should emit the per-result returned currency."""
        flight = _make_flight(price=275.0)
        flight.currency = "EUR"

        result = _serialize_flight_result(flight, is_round_trip=False)

        assert result["currency"] == "EUR"


# ---------------------------------------------------------------------------
# Bug 3: 4-leg multi-city dropping bookable carriers
# ---------------------------------------------------------------------------
#
# Reported case: a 4-leg LGW↔China itinerary returned either 0 flights
# or only-unpriced Air China options, even though every leg priced fine
# via the one-way ``search_flights`` path (CZ 690 LGW→CAN at £421
# standalone).  The 3-leg variant of the same routing returned priced
# CZ correctly.  Hypothesis: Google's multi-city curator restricts
# candidates to single-carrier or alliance bundles for 4+-leg trips,
# dropping carriers (CZ/SkyTeam) that don't have full European-partner
# coverage of the routing.  The fix surfaces a per-leg one-way fallback
# whenever the multi-city call returns empty/unpriced for a leg.


def _future_date(days: int = 30) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def _flight_result(airline_code: str, flight_num: str, price: float):
    """Build a FlightResult-shaped MagicMock with one leg, suitable for serialization."""
    leg = MagicMock()
    leg.departure_airport = "LGW"
    leg.arrival_airport = "SHA"
    leg.departure_datetime = datetime(2026, 5, 18, 11, 40)
    leg.arrival_datetime = datetime(2026, 5, 19, 6, 20)
    leg.duration = 600
    airline = MagicMock()
    airline.name = airline_code
    leg.airline = airline
    leg.flight_number = flight_num

    flight = MagicMock()
    flight.price = price
    flight.currency = "GBP"
    flight.legs = [leg]
    return flight


def _four_leg_params() -> MultiCitySearchParams:
    legs = [
        MultiCityLeg(origin="LGW", destination="SHA", date=_future_date(30)),
        MultiCityLeg(origin="SHA", destination="CTU", date=_future_date(33)),
        MultiCityLeg(origin="CTU", destination="CAN", date=_future_date(44)),
        MultiCityLeg(origin="CAN", destination="LGW", date=_future_date(47)),
    ]
    return MultiCitySearchParams(legs=legs, sort_by="CHEAPEST")


@pytest.fixture(autouse=True)
def _clear_multi_city_sessions():
    _search_sessions.clear()
    yield
    _search_sessions.clear()


class TestMultiCityFallback:
    """Per-leg fallback when Google's multi-city curator returns empty/unpriced."""

    def test_empty_multi_city_falls_back_to_one_way(self):
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                None,  # multi-city call: empty body
                ([_flight_result("CZ", "690", 421.0)], {}),  # fallback: priced CZ
            ]

            result = _execute_multi_city_step(_four_leg_params(), selection=None)

        assert result["success"] is True
        assert result["count"] == 1
        assert result["combined_pricing"] is False
        assert result["flights"][0]["per_leg_price"] is True
        assert result["flights"][0]["price"] == 421.0
        assert "per-leg standalone prices" in result["message"]
        assert instance._do_single_search.call_count == 2

    def test_all_unpriced_multi_city_falls_back_to_one_way(self):
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                # Mirrors Repro B: multi-city returns only unpriced Air China
                (
                    [_flight_result("CA", "100", 0.0), _flight_result("CA", "200", 0.0)],
                    {},
                ),
                ([_flight_result("CZ", "690", 421.0)], {}),
            ]

            result = _execute_multi_city_step(_four_leg_params(), selection=None)

        assert result["combined_pricing"] is False
        assert result["count"] == 1
        assert result["flights"][0]["per_leg_price"] is True
        assert instance._do_single_search.call_count == 2

    def test_priced_multi_city_does_not_invoke_fallback(self):
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                (
                    [_flight_result("CZ", "690", 820.0)],
                    {"price_range": [820, 1200]},
                ),
            ]

            result = _execute_multi_city_step(_four_leg_params(), selection=None)

        assert result["combined_pricing"] is True
        assert "per_leg_price" not in result["flights"][0]
        assert result["price_range"] == {"min": 820, "max": 1200}
        assert instance._do_single_search.call_count == 1

    def test_fallback_also_unpriced_keeps_empty_response(self):
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                None,
                ([_flight_result("CA", "999", 0.0)], {}),  # also unpriced
            ]

            result = _execute_multi_city_step(_four_leg_params(), selection=None)

        # Original empty response surfaces with retry hint; we don't show
        # an unpriced-only list as a "real" result.
        assert result["count"] == 0
        assert "hint" in result
        assert instance._do_single_search.call_count == 2

    def test_fallback_returns_empty_keeps_empty_response(self):
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [None, None]

            result = _execute_multi_city_step(_four_leg_params(), selection=None)

        assert result["count"] == 0
        assert "hint" in result
        assert instance._do_single_search.call_count == 2

    def test_fallback_filter_targets_current_leg_at_step_one(self):
        """At step 1, the fallback's one-way segment must come from leg 0.

        Sanity coverage of the simple case before the more interesting
        continuation-step test below.
        """
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                None,
                ([_flight_result("CZ", "690", 421.0)], {}),
            ]

            _execute_multi_city_step(_four_leg_params(), selection=None)

            assert instance._do_single_search.call_count == 2
            mc_filters = instance._do_single_search.call_args_list[0].args[0]
            ow_filters = instance._do_single_search.call_args_list[1].args[0]

            assert len(mc_filters.flight_segments) == 4
            assert len(ow_filters.flight_segments) == 1
            # Leg 0 of _four_leg_params is LGW -> SHA.  "SHA" is an IATA
            # metro code that expands to {PVG, SHA}, so the fallback should
            # carry the same expanded airport set, not just one of them.
            ow_origin = {a[0].name for a in ow_filters.flight_segments[0].departure_airport}
            ow_destination = {a[0].name for a in ow_filters.flight_segments[0].arrival_airport}
            assert ow_origin == {"LGW"}
            assert ow_destination == {"PVG", "SHA"}

    def test_fallback_filter_targets_current_leg_on_continuation(self):
        """On a continuation step the fallback must use *that* leg's airports.

        Regression guard: the previous version of this test only exercised
        leg 0 (``selection=None``), so a code path that accidentally
        re-fetched leg 0 every time the multi-city call failed on a
        later leg would silently pass.  Drive the handler through to
        step 2 with a successful step 1, then make the multi-city call
        for step 2 fail and assert the fallback's one-way filter
        carries leg 1's airports (SHA -> CTU), not leg 0's.
        """
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                # Step 1: multi-city succeeds with a priced leg-1 option.
                ([_flight_result("CZ", "690", 421.0)], {}),
                # Step 2: multi-city call empty → fallback fires.
                None,
                # Step 2: fallback returns priced one-way for leg 1 (SHA→CTU).
                ([_flight_result("CZ", "3471", 75.0)], {}),
            ]

            params = _four_leg_params()
            step1 = _execute_multi_city_step(params, selection=None)
            assert step1["combined_pricing"] is True  # no fallback yet
            assert step1["count"] == 1

            step2 = _execute_multi_city_step(params, selection=0)
            assert step2["combined_pricing"] is False  # fallback fired
            assert step2["count"] == 1
            assert step2["flights"][0]["per_leg_price"] is True

            # Three calls total: 1 (step1 multi-city) + 2 (step2 multi-city + fallback).
            assert instance._do_single_search.call_count == 3
            fallback_filters = instance._do_single_search.call_args_list[2].args[0]
            assert len(fallback_filters.flight_segments) == 1

            # Leg 1 of _four_leg_params is SHA -> CTU (no metro expansion
            # here — both are single airports in this fixture).  The
            # fallback MUST target this leg, not leg 0 (LGW → SHA/PVG).
            fb_origin = {a[0].name for a in fallback_filters.flight_segments[0].departure_airport}
            fb_destination = {
                a[0].name for a in fallback_filters.flight_segments[0].arrival_airport
            }
            assert fb_origin == {"PVG", "SHA"}, (
                f"Fallback origin should match leg 1 (SHA metro = PVG+SHA), "
                f"got {fb_origin}.  Regression: fallback re-targeted leg 0."
            )
            assert fb_destination == {"CTU"}, (
                f"Fallback destination should match leg 1 (CTU), got {fb_destination}."
            )


# ---------------------------------------------------------------------------
# Sticky degraded state across continuation steps
# ---------------------------------------------------------------------------
#
# Once Google's multi-city curator drops options for one leg of a trip,
# repeating the broken call for every subsequent leg would burn ~2 minutes
# per leg with the same outcome.  After the first fallback the session is
# marked ``degraded`` and continuation calls skip straight to the per-leg
# one-way query — turning a 4-leg degraded trip from ~8 minutes of futile
# retries into ~12 seconds of useful results.


class TestStickyDegradedState:
    """Once a trip is degraded, every step is degraded."""

    def test_continuation_after_fallback_skips_multi_city_call(self):
        """Step 2 of a degraded session must NOT call the multi-city endpoint.

        Without this short-circuit the agent pays the full ~60-120 s
        multi-city wall time on every continuation, even though we
        already know it won't return useful results.
        """
        params = _four_leg_params()

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                # Step 1: multi-city empty, fallback succeeds with priced CZ
                None,
                ([_flight_result("CZ", "690", 421.0)], {}),
                # Step 2: ONLY one call expected (fallback only — multi-city
                # is skipped because the session is degraded).
                ([_flight_result("CZ", "3443", 75.0)], {}),
            ]

            step1 = _execute_multi_city_step(params, selection=None)
            assert step1["combined_pricing"] is False
            assert step1["count"] == 1

            calls_after_step1 = instance._do_single_search.call_count
            assert calls_after_step1 == 2  # multi-city + fallback

            step2 = _execute_multi_city_step(params, selection=0)
            assert step2["combined_pricing"] is False
            assert step2["flights"][0]["per_leg_price"] is True

            # Crucial: step 2 added exactly ONE API call (the fallback).
            # If degraded was not sticky, we'd see 2 more (multi-city + fallback).
            assert instance._do_single_search.call_count == calls_after_step1 + 1

    def test_continuation_uses_message_for_carry_over_degradation(self):
        """The message should distinguish 'this leg failed' vs 'trip already degraded'."""
        params = _four_leg_params()

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                None,
                ([_flight_result("CZ", "690", 421.0)], {}),
                ([_flight_result("CZ", "3443", 75.0)], {}),
            ]

            step1 = _execute_multi_city_step(params, selection=None)
            step2 = _execute_multi_city_step(params, selection=0)

            assert "unavailable for leg 1" in step1["message"]
            assert "Continuing in per-leg pricing mode" in step2["message"]

    def test_degraded_flag_persists_across_empty_intermediate_step(self):
        """Keep the degraded flag set when a continuation step's fallback is empty.

        If a continuation step's fallback also turns up empty, the session
        must keep the degraded flag so the *next* retry still skips the
        multi-city call.
        """
        params = _four_leg_params()

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                # Step 1: degraded, fallback succeeds
                None,
                ([_flight_result("CZ", "690", 421.0)], {}),
                # Step 2: fallback also empty (transient)
                None,
                # Step 2 retry: fallback succeeds
                ([_flight_result("CZ", "3443", 75.0)], {}),
            ]

            _execute_multi_city_step(params, selection=None)
            empty_step2 = _execute_multi_city_step(params, selection=0)
            assert empty_step2["count"] == 0
            assert empty_step2["combined_pricing"] is False  # still degraded

            # Retry step 2 with same selection — should still skip multi-city
            calls_before_retry = instance._do_single_search.call_count
            retry_step2 = _execute_multi_city_step(params, selection=0)
            assert retry_step2["count"] == 1
            assert retry_step2["combined_pricing"] is False
            # Only one new call (the successful fallback), not multi-city + fallback
            assert instance._do_single_search.call_count == calls_before_retry + 1

    def test_fresh_session_starts_undegraded(self):
        """Sanity: a brand new session (cached is None) starts with degraded=False."""
        params = _four_leg_params()

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            # Multi-city succeeds with a priced result on first try
            instance._do_single_search.side_effect = [
                ([_flight_result("CZ", "690", 820.0)], {"price_range": [820, 1200]}),
            ]

            result = _execute_multi_city_step(params, selection=None)

            assert result["combined_pricing"] is True
            assert "per_leg_price" not in result["flights"][0]
            # Only the multi-city call ran — no fallback
            assert instance._do_single_search.call_count == 1


# ---------------------------------------------------------------------------
# City / metro code support in MCP params
# ---------------------------------------------------------------------------


class TestCityCodeParams:
    """Pydantic params accept city codes and explicit airport arrays."""

    def test_flight_params_accept_string_origin(self):
        params = FlightSearchParams(origin="LON", destination="JFK", departure_date="2027-06-01")
        assert params.origin == "LON"

    def test_flight_params_accept_list_origin(self):
        params = FlightSearchParams(
            origin=["LHR", "LGW"], destination="JFK", departure_date="2027-06-01"
        )
        assert params.origin == ["LHR", "LGW"]

    def test_date_params_accept_list(self):
        params = DateSearchParams(
            origin=["LHR", "LGW"],
            destination="JFK",
            start_date="2027-06-01",
            end_date="2027-06-30",
        )
        assert params.origin == ["LHR", "LGW"]

    def test_multi_city_leg_accepts_city_code(self):
        leg = MultiCityLeg(origin="LON", destination="NYC", date="2027-06-01")
        assert leg.origin == "LON"
        assert leg.destination == "NYC"

    def test_multi_city_leg_accepts_list(self):
        leg = MultiCityLeg(origin=["LHR", "LGW"], destination=["JFK", "EWR"], date="2027-06-01")
        assert leg.origin == ["LHR", "LGW"]
        assert leg.destination == ["JFK", "EWR"]

    def test_multi_city_params_serialize(self):
        params = MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="NYC", destination="LON", date="2027-06-01"),
                MultiCityLeg(origin="LON", destination="PAR", date="2027-06-05"),
            ]
        )
        assert len(params.legs) == 2


class TestCityResolutionFlow:
    """City codes flow through to FlightSegment with multiple airports per slot."""

    def test_city_origin_expands_to_multiple_airports(self):
        """LON in origin should produce 6-airport departure slot."""
        captured = {}

        class _FakeSearchClient:
            def search(self, filters):
                captured["segments"] = filters.flight_segments
                return []

        params = FlightSearchParams(origin="LON", destination="JFK", departure_date="2027-06-01")
        with patch("fli.mcp.server.SearchFlights", return_value=_FakeSearchClient()):
            result = _execute_flight_search(params)

        assert result["success"]
        segments = captured["segments"]
        assert len(segments) == 1
        # 6 London airports as departure
        assert len(segments[0].departure_airport) == 6
        # Single JFK destination
        assert len(segments[0].arrival_airport) == 1

    def test_invalid_city_code_returns_parse_error(self):
        """Unknown code in origin/destination should bubble up as a parse error."""
        params = FlightSearchParams(origin="ZZZ", destination="JFK", departure_date="2027-06-01")
        result = _execute_flight_search(params)
        assert not result["success"]
        assert "Invalid airport or city code" in result["error"]

    def test_explicit_airport_array_passes_through(self):
        captured = {}

        class _FakeSearchClient:
            def search(self, filters):
                captured["segments"] = filters.flight_segments
                return []

        params = FlightSearchParams(
            origin=["LHR", "LGW"], destination="JFK", departure_date="2027-06-01"
        )
        with patch("fli.mcp.server.SearchFlights", return_value=_FakeSearchClient()):
            _execute_flight_search(params)

        assert len(captured["segments"][0].departure_airport) == 2


class TestSessionKeyCanonicalisation:
    """Multi-city session key is stable across equivalent endpoint inputs."""

    def test_metro_string_and_explicit_list_produce_same_key(self):
        from fli.mcp.server import _session_key

        a = _session_key([MultiCityLeg(origin="LON", destination="JFK", date="2027-06-01")])
        b = _session_key(
            [
                MultiCityLeg(
                    origin=["LHR", "LGW", "STN", "LCY", "LTN", "SEN"],
                    destination="JFK",
                    date="2027-06-01",
                )
            ]
        )
        assert a == b

    def test_whitespace_variants_match(self):
        from fli.mcp.server import _session_key

        a = _session_key([MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01")])
        b = _session_key([MultiCityLeg(origin=" jfk ", destination="lhr", date="2027-06-01")])
        assert a == b

    def test_different_endpoints_produce_different_keys(self):
        from fli.mcp.server import _session_key

        a = _session_key([MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01")])
        b = _session_key([MultiCityLeg(origin="JFK", destination="CDG", date="2027-06-01")])
        assert a != b


class TestNegativeSelection:
    """Multi-city selection should reject negative indices."""

    def test_negative_selection_returns_error(self):
        from fli.mcp.server import _execute_multi_city_step, _search_sessions

        # Seed a fake cached session so the handler reaches the selection check.
        fake_filters = MagicMock()
        fake_filters.flight_segments = [MagicMock(selected_flight=None)] * 2
        params = MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01"),
                MultiCityLeg(origin="LHR", destination="CDG", date="2027-06-05"),
            ]
        )
        # Match the full cache key the handler will compute (legs + filter
        # params, not just legs — see _session_cache_key for the rationale).
        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)
        # Session value is a 3-tuple ``(filters, last_results, degraded)``;
        # the autouse ``_clear_multi_city_sessions`` fixture handles cleanup.
        _search_sessions[key] = (
            fake_filters,
            ["flight_a", "flight_b", "flight_c"],
            False,
        )

        result = _execute_multi_city_step(params, selection=-1)
        assert result["success"] is False
        assert result["error_kind"] == "validation"
        assert "out of range" in result["error"]


# ---------------------------------------------------------------------------
# Review fixes: addresses comments on punitarani/fli#145 (closed) /
# leethree/fli#2 (active).
# ---------------------------------------------------------------------------


class TestSessionKeyIncludesFilters:
    """Cache key must include filter params so reformulated calls miss the cache.

    Without folding the filters in, a continuation call that changed
    cabin_class / sort_by / etc. would silently reuse the original
    session and return results computed under the previous filters.
    """

    def _params(self, **overrides):
        base = {
            "legs": [
                MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01"),
                MultiCityLeg(origin="LHR", destination="CDG", date="2027-06-05"),
            ],
        }
        base.update(overrides)
        return MultiCitySearchParams(**base)

    def test_same_filters_produce_same_cache_key(self):
        from fli.mcp.server import _session_cache_key

        a = _session_cache_key(self._params(sort_by="CHEAPEST"))
        b = _session_cache_key(self._params(sort_by="CHEAPEST"))
        assert a == b

    def test_changing_sort_by_misses_cache(self):
        from fli.mcp.server import _session_cache_key

        cheapest = _session_cache_key(self._params(sort_by="CHEAPEST"))
        duration = _session_cache_key(self._params(sort_by="DURATION"))
        assert cheapest != duration

    def test_changing_max_stops_misses_cache(self):
        from fli.mcp.server import _session_cache_key

        any_stops = _session_cache_key(self._params(max_stops="ANY"))
        non_stop = _session_cache_key(self._params(max_stops="NON_STOP"))
        assert any_stops != non_stop

    def test_changing_cabin_class_misses_cache(self):
        from fli.mcp.server import _session_cache_key

        eco = _session_cache_key(self._params(cabin_class="ECONOMY"))
        biz = _session_cache_key(self._params(cabin_class="BUSINESS"))
        assert eco != biz

    def test_changing_airlines_misses_cache(self):
        from fli.mcp.server import _session_cache_key

        none_airlines = _session_cache_key(self._params(airlines=None))
        ba_airlines = _session_cache_key(self._params(airlines=["BA"]))
        ba_aa_airlines = _session_cache_key(self._params(airlines=["BA", "AA"]))
        assert none_airlines != ba_airlines != ba_aa_airlines

    def test_airline_order_does_not_matter(self):
        """Caller passing ['BA','AA'] should hit the same session as ['AA','BA']."""
        from fli.mcp.server import _session_cache_key

        a = _session_cache_key(self._params(airlines=["BA", "AA"]))
        b = _session_cache_key(self._params(airlines=["AA", "BA"]))
        assert a == b

    def test_legs_changing_misses_cache(self):
        """Sanity: the legs-portion of the key still discriminates routes."""
        from fli.mcp.server import _session_cache_key

        jfk = _session_cache_key(self._params())
        params_other = MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="EWR", destination="LHR", date="2027-06-01"),
                MultiCityLeg(origin="LHR", destination="CDG", date="2027-06-05"),
            ],
        )
        ewr = _session_cache_key(params_other)
        assert jfk != ewr


class TestMultiCityToolAnnotations:
    """search_multi_city is stateful; readOnly/idempotent hints must be False."""

    @pytest.mark.asyncio
    async def test_search_multi_city_is_not_marked_readonly_or_idempotent(self):
        tools = await mcp.list_tools()
        tool = next(t for t in tools if t.name == "search_multi_city")
        # ``False`` is the explicit signal to MCP clients that this tool
        # mutates server state (``_search_sessions``) and that retried
        # invocations are not safe — skipping a leg by silently advancing
        # the session pointer would be a real footgun.
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.idempotentHint is False

    @pytest.mark.asyncio
    async def test_search_flights_remains_readonly_and_idempotent(self):
        """Sanity check that we didn't over-correct the read-only tools."""
        tools = await mcp.list_tools()
        sf = next(t for t in tools if t.name == "search_flights")
        sd = next(t for t in tools if t.name == "search_dates")
        assert sf.annotations.readOnlyHint is True
        assert sf.annotations.idempotentHint is True
        assert sd.annotations.readOnlyHint is True
        assert sd.annotations.idempotentHint is True


# ---------------------------------------------------------------------------
# Issue C from BUGREPORT 2026-05-03: airlines / origin / destination /
# legs passed as JSON-encoded strings (some MCP transports re-encode list
# arguments before they reach Pydantic, so the model receives the literal
# string ``'["CZ", "BA"]'`` and rejects it as "not a list").
# ---------------------------------------------------------------------------


class TestJsonStringListCoercion:
    """List-shaped MCP params accept JSON-encoded string forms transparently."""

    def test_airlines_from_json_string_on_flight_params(self) -> None:
        """``airlines='["CZ", "BA"]'`` must coerce to ``['CZ', 'BA']``."""
        p = FlightSearchParams(
            origin="JFK",
            destination="LHR",
            departure_date="2027-06-01",
            airlines='["CZ", "BA"]',
        )
        assert p.airlines == ["CZ", "BA"]

    def test_airlines_from_json_string_on_multi_city(self) -> None:
        """Same coercion applies to MultiCitySearchParams.airlines."""
        p = MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01"),
                MultiCityLeg(origin="LHR", destination="CDG", date="2027-06-05"),
            ],
            airlines='["CZ", "BA"]',
        )
        assert p.airlines == ["CZ", "BA"]

    def test_airlines_from_json_string_on_date_params(self) -> None:
        """Same coercion applies to DateSearchParams.airlines."""
        p = DateSearchParams(
            origin="JFK",
            destination="LHR",
            start_date="2027-06-01",
            end_date="2027-06-30",
            airlines='["BA"]',
        )
        assert p.airlines == ["BA"]

    def test_origin_from_json_string(self) -> None:
        """``origin='["LHR", "LGW"]'`` must coerce to a list (FlightSearchParams)."""
        p = FlightSearchParams(
            origin='["LHR", "LGW"]',
            destination="JFK",
            departure_date="2027-06-01",
        )
        assert p.origin == ["LHR", "LGW"]

    def test_destination_from_json_string(self) -> None:
        """``destination='["JFK", "EWR"]'`` must coerce on MultiCityLeg too."""
        leg = MultiCityLeg(
            origin="LHR",
            destination='["JFK", "EWR"]',
            date="2027-06-01",
        )
        assert leg.destination == ["JFK", "EWR"]

    def test_legs_from_json_string(self) -> None:
        """``legs`` passed as a JSON-encoded array of objects must parse cleanly."""
        legs_json = (
            '[{"origin":"JFK","destination":"LHR","date":"2027-06-01"},'
            '{"origin":"LHR","destination":"CDG","date":"2027-06-05"}]'
        )
        p = MultiCitySearchParams(legs=legs_json)
        assert len(p.legs) == 2
        assert p.legs[0].origin == "JFK"
        assert p.legs[1].destination == "CDG"

    def test_bare_metro_code_string_not_coerced(self) -> None:
        """A legitimate bare string like ``'LON'`` must NOT be touched.

        The coercion only fires on strings that look like a JSON list
        (``[...]``); IATA codes and metro codes flow through to the
        existing ``str | list[str]`` union path.
        """
        p = FlightSearchParams(origin="LON", destination="JFK", departure_date="2027-06-01")
        assert p.origin == "LON"
        assert isinstance(p.origin, str)

    def test_bare_airport_code_string_not_coerced(self) -> None:
        """``origin='JFK'`` stays a string."""
        p = FlightSearchParams(origin="JFK", destination="LHR", departure_date="2027-06-01")
        assert p.origin == "JFK"

    def test_already_a_list_passes_through(self) -> None:
        """The happy-path list input must remain unchanged."""
        p = FlightSearchParams(
            origin=["LHR", "LGW"],
            destination="JFK",
            departure_date="2027-06-01",
            airlines=["CZ"],
        )
        assert p.origin == ["LHR", "LGW"]
        assert p.airlines == ["CZ"]

    def test_none_airlines_unchanged(self) -> None:
        """``airlines=None`` is the default and must stay ``None``."""
        p = FlightSearchParams(
            origin="JFK",
            destination="LHR",
            departure_date="2027-06-01",
            airlines=None,
        )
        assert p.airlines is None

    def test_malformed_json_string_still_rejected_for_airlines(self) -> None:
        """Malformed JSON-list-like strings are not coerced.

        They fall through to Pydantic, which then rejects them as
        not-a-list rather than swallowing the bad input.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FlightSearchParams(
                origin="JFK",
                destination="LHR",
                departure_date="2027-06-01",
                airlines="[CZ, BA]",  # not valid JSON (unquoted)
            )

    def test_non_list_json_not_coerced(self) -> None:
        """Non-list JSON strings are not coerced.

        Strings that parse as non-list JSON (object, number) flow
        through to the type validator unchanged.
        """
        # ``'{"x": 1}'`` parses but isn't a list, so the coercer returns
        # the original string and Pydantic decides what to do.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FlightSearchParams(
                origin="JFK",
                destination="LHR",
                departure_date="2027-06-01",
                airlines='{"x": 1}',
            )


class TestCoerceJsonListStringDirect:
    """Direct unit tests for the _coerce_json_list_string helper.

    The field_validator tests above exercise the function through Pydantic;
    these tests target the function's own contract (None passthrough, numeric
    JSON list, whitespace-padded brackets, integer input, etc.) to guard
    against regressions in the helper's logic independent of Pydantic.
    """

    def _fn(self, value):
        from fli.mcp.server import _coerce_json_list_string

        return _coerce_json_list_string(value)

    def test_none_passes_through(self) -> None:
        assert self._fn(None) is None

    def test_integer_passes_through(self) -> None:
        assert self._fn(42) == 42

    def test_plain_list_passes_through(self) -> None:
        original = ["BA", "CZ"]
        assert self._fn(original) is original

    def test_empty_list_json_string_returns_empty_list(self) -> None:
        assert self._fn("[]") == []

    def test_numeric_json_array_returns_parsed_list(self) -> None:
        result = self._fn("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_whitespace_padded_brackets_are_stripped_before_check(self) -> None:
        """Leading/trailing whitespace around the JSON array must not block coercion."""
        result = self._fn('  ["LHR", "LGW"]  ')
        assert result == ["LHR", "LGW"]

    def test_json_number_string_not_bracket_passes_through(self) -> None:
        """A string that looks like a JSON number (no brackets) is unchanged."""
        assert self._fn("42") == "42"

    def test_json_object_string_passes_through_unchanged(self) -> None:
        """A JSON object string (starts with ``{``) must not be altered."""
        assert self._fn('{"a": 1}') == '{"a": 1}'

    def test_nested_json_list_string_is_parsed(self) -> None:
        """JSON array containing nested structures parses without crashing."""
        raw = '[{"origin": "JFK", "destination": "LHR", "date": "2027-06-01"}]'
        result = self._fn(raw)
        assert isinstance(result, list)
        assert result[0]["origin"] == "JFK"

    def test_invalid_json_inside_brackets_returns_original_string(self) -> None:
        """Bracketed strings that are not valid JSON pass through unchanged.

        Returned to Pydantic as-is so it can reject them with the
        correct error rather than swallowing malformed input.
        """
        raw = "[not, valid, json"  # missing closing bracket
        assert self._fn(raw) == raw

    def test_string_starting_with_bracket_but_no_closing_passes_through(self) -> None:
        raw = "[BA, CZ"
        assert self._fn(raw) == raw

    def test_json_array_of_nulls_returns_list(self) -> None:
        result = self._fn("[null, null]")
        assert result == [None, None]

    def test_dict_value_passes_through(self) -> None:
        original = {"key": "value"}
        assert self._fn(original) is original


# ---------------------------------------------------------------------------
# Unit tests for _normalize_flight_prices
# ---------------------------------------------------------------------------


class TestNormalizeFlightPrices:
    """Direct unit tests for the _normalize_flight_prices helper.

    This function tags price-unavailable entries and moves them to the
    end of the list when sort_by is CHEAPEST.  It mutates the input list
    in-place and returns it.
    """

    def _normalize(self, results, sort_by=None):
        from fli.mcp.server import _normalize_flight_prices

        return _normalize_flight_prices(results, sort_by=sort_by)

    def _make_sort_by(self, name):
        """Return a minimal object with a .name attribute for sort_by duck-typing."""
        obj = MagicMock()
        obj.name = name
        return obj

    def test_zero_price_becomes_none_and_tagged_unavailable(self) -> None:
        results = [{"price": 0.0, "currency": "USD"}]
        out = self._normalize(results)
        assert out[0]["price"] is None
        assert out[0]["price_unavailable"] is True

    def test_none_price_tagged_unavailable(self) -> None:
        results = [{"price": None, "currency": "USD"}]
        out = self._normalize(results)
        assert out[0]["price"] is None
        assert out[0]["price_unavailable"] is True

    def test_positive_price_tagged_available(self) -> None:
        results = [{"price": 420.0, "currency": "GBP"}]
        out = self._normalize(results)
        assert out[0]["price"] == 420.0
        assert out[0]["price_unavailable"] is False

    def test_mixed_prices_all_tagged(self) -> None:
        results = [
            {"price": 100.0, "currency": "USD"},
            {"price": 0.0, "currency": "USD"},
            {"price": 200.0, "currency": "USD"},
        ]
        out = self._normalize(results)
        assert out[0]["price_unavailable"] is False
        assert out[1]["price_unavailable"] is True
        assert out[2]["price_unavailable"] is False

    def test_cheapest_sort_pushes_unpriced_to_end(self) -> None:
        results = [
            {"price": 0.0},  # unpriced: should move to end
            {"price": 500.0},
            {"price": 300.0},
        ]
        out = self._normalize(results, sort_by=self._make_sort_by("CHEAPEST"))
        # First two entries must be priced; the unpriced entry lands last.
        assert out[0]["price_unavailable"] is False
        assert out[1]["price_unavailable"] is False
        assert out[2]["price_unavailable"] is True

    def test_cheapest_sort_preserves_relative_order_of_priced(self) -> None:
        """Stable sort: priced entries keep their original server-side order."""
        results = [
            {"price": 500.0},
            {"price": 0.0},
            {"price": 300.0},
        ]
        out = self._normalize(results, sort_by=self._make_sort_by("CHEAPEST"))
        # Original server order (500 then 300) should be preserved among priced.
        priced = [r for r in out if not r["price_unavailable"]]
        assert priced[0]["price"] == 500.0
        assert priced[1]["price"] == 300.0

    def test_non_cheapest_sort_does_not_reorder(self) -> None:
        """DURATION sort_by must not trigger the reordering step."""
        results = [
            {"price": 0.0},  # would move to end only under CHEAPEST
            {"price": 200.0},
        ]
        out = self._normalize(results, sort_by=self._make_sort_by("DURATION"))
        # Order must be unchanged.
        assert out[0]["price_unavailable"] is True
        assert out[1]["price_unavailable"] is False

    def test_no_sort_by_does_not_reorder(self) -> None:
        results = [{"price": 0.0}, {"price": 200.0}]
        out = self._normalize(results, sort_by=None)
        assert out[0]["price_unavailable"] is True

    def test_all_unpriced_cheapest_stays_all_at_end(self) -> None:
        results = [{"price": 0.0}, {"price": None}]
        out = self._normalize(results, sort_by=self._make_sort_by("CHEAPEST"))
        assert all(r["price_unavailable"] for r in out)

    def test_empty_list_returns_empty_list(self) -> None:
        assert self._normalize([]) == []

    def test_mutates_in_place_and_returns_same_object(self) -> None:
        results = [{"price": 100.0}]
        out = self._normalize(results)
        assert out is results


# ---------------------------------------------------------------------------
# Unit tests for _classify_search_error
# ---------------------------------------------------------------------------


class TestClassifySearchError:
    """Direct unit tests for _classify_search_error.

    Ensures the error classifier maps exception messages to the correct
    ``error_kind`` tags so callers can branch without string-matching.
    """

    def _classify(self, msg):
        from fli.mcp.server import _classify_search_error

        return _classify_search_error(Exception(msg))

    def test_timed_out_phrase_classified_as_timeout(self) -> None:
        result = self._classify("Operation timed out after 25s")
        assert result["error_kind"] == "timeout"
        assert result["success"] is False
        assert result["flights"] == []

    def test_curl_error_28_classified_as_timeout(self) -> None:
        result = self._classify("curl: (28) Operation timed out")
        assert result["error_kind"] == "timeout"
        assert "retry" in result["error"].lower()

    def test_mixed_case_timed_out_classified_as_timeout(self) -> None:
        """Case-insensitive match for 'timed out'."""
        result = self._classify("Timed Out: connection stalled")
        assert result["error_kind"] == "timeout"

    def test_validation_error_phrase_classified(self) -> None:
        result = self._classify("1 validation error for FlightSearchFilters")
        assert result["error_kind"] == "validation"
        assert result["error"] == "Invalid parameter value"

    def test_generic_exception_classified_as_unknown(self) -> None:
        result = self._classify("Some random network failure")
        assert result["error_kind"] == "unknown"
        assert "Search failed:" in result["error"]
        assert "Some random network failure" in result["error"]

    def test_timeout_response_shape(self) -> None:
        """Timeout result must include all required keys for MCP consumers."""
        result = self._classify("curl: (28) request timed out")
        assert set(result.keys()) >= {"success", "error_kind", "error", "flights"}
        assert result["flights"] == []

    def test_validation_response_shape(self) -> None:
        result = self._classify("validation error in field cabin_class")
        assert result["success"] is False
        assert result["flights"] == []

    def test_unknown_response_shape(self) -> None:
        result = self._classify("unexpected internal error")
        assert result["success"] is False
        assert result["flights"] == []


# ---------------------------------------------------------------------------
# _execute_flight_search and _execute_date_search error paths
# ---------------------------------------------------------------------------


class TestExecuteFlightSearchErrorPaths:
    """_execute_flight_search must surface error_kind on all failure modes."""

    def test_parse_error_returns_parse_kind(self) -> None:
        from fli.core.parsers import ParseError
        from fli.mcp.server import _execute_flight_search

        params = FlightSearchParams(origin="JFK", destination="LHR", departure_date="2027-06-01")
        with patch("fli.mcp.server.resolve_origin_or_city", side_effect=ParseError("bad code")):
            result = _execute_flight_search(params)

        assert result["success"] is False
        assert result["error_kind"] == "parse"
        assert "bad code" in result["error"]
        assert result["flights"] == []

    def test_timeout_exception_returns_timeout_kind(self) -> None:
        from fli.mcp.server import _execute_flight_search

        params = FlightSearchParams(origin="JFK", destination="LHR", departure_date="2027-06-01")
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            mock_cls.return_value.search.side_effect = Exception("timed out after 25s")
            result = _execute_flight_search(params)

        assert result["success"] is False
        assert result["error_kind"] == "timeout"
        assert result["flights"] == []

    def test_validation_exception_returns_validation_kind(self) -> None:
        from fli.mcp.server import _execute_flight_search

        params = FlightSearchParams(origin="JFK", destination="LHR", departure_date="2027-06-01")
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            mock_cls.return_value.search.side_effect = Exception("1 validation error for model")
            result = _execute_flight_search(params)

        assert result["success"] is False
        assert result["error_kind"] == "validation"

    def test_generic_exception_returns_unknown_kind(self) -> None:
        from fli.mcp.server import _execute_flight_search

        params = FlightSearchParams(origin="JFK", destination="LHR", departure_date="2027-06-01")
        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            mock_cls.return_value.search.side_effect = Exception("connection refused")
            result = _execute_flight_search(params)

        assert result["success"] is False
        assert result["error_kind"] == "unknown"
        assert "connection refused" in result["error"]


class TestExecuteDateSearchErrorPaths:
    """_execute_date_search errors must use 'dates' as the empty-list key."""

    def _date_params(self):
        return DateSearchParams(
            origin="JFK",
            destination="LHR",
            start_date="2027-06-01",
            end_date="2027-06-30",
        )

    def test_parse_error_returns_parse_kind_with_dates_key(self) -> None:
        from fli.core.parsers import ParseError
        from fli.mcp.server import _execute_date_search

        with patch("fli.mcp.server.resolve_origin_or_city", side_effect=ParseError("bad code")):
            result = _execute_date_search(self._date_params())

        assert result["success"] is False
        assert result["error_kind"] == "parse"
        assert result["dates"] == []
        assert "flights" not in result

    def test_timeout_exception_returns_timeout_kind_with_dates_key(self) -> None:
        from fli.mcp.server import _execute_date_search

        with patch("fli.mcp.server.SearchDates") as mock_cls:
            mock_cls.return_value.search.side_effect = Exception("timed out")
            result = _execute_date_search(self._date_params())

        assert result["success"] is False
        assert result["error_kind"] == "timeout"
        assert result["dates"] == []
        assert "flights" not in result

    def test_generic_exception_returns_unknown_kind_with_dates_key(self) -> None:
        from fli.mcp.server import _execute_date_search

        with patch("fli.mcp.server.SearchDates") as mock_cls:
            mock_cls.return_value.search.side_effect = Exception("network error")
            result = _execute_date_search(self._date_params())

        assert result["success"] is False
        assert result["error_kind"] == "unknown"
        assert result["dates"] == []
        assert "flights" not in result


# ---------------------------------------------------------------------------
# Unit tests for _all_unpriced
# ---------------------------------------------------------------------------


class TestAllUnpriced:
    """Direct unit tests for _all_unpriced helper."""

    def _all_unpriced(self, items):
        from fli.mcp.server import _all_unpriced

        return _all_unpriced(items)

    def _mock_flight(self, price):
        f = MagicMock()
        f.price = price
        return f

    def test_empty_list_returns_true(self) -> None:
        """Vacuous all() on empty list must return True."""
        assert self._all_unpriced([]) is True

    def test_all_zero_price_returns_true(self) -> None:
        flights = [self._mock_flight(0.0), self._mock_flight(0.0)]
        assert self._all_unpriced(flights) is True

    def test_all_none_price_returns_true(self) -> None:
        flights = [self._mock_flight(None), self._mock_flight(None)]
        assert self._all_unpriced(flights) is True

    def test_mix_zero_and_none_returns_true(self) -> None:
        flights = [self._mock_flight(0.0), self._mock_flight(None)]
        assert self._all_unpriced(flights) is True

    def test_one_priced_returns_false(self) -> None:
        flights = [self._mock_flight(0.0), self._mock_flight(300.0)]
        assert self._all_unpriced(flights) is False

    def test_all_priced_returns_false(self) -> None:
        flights = [self._mock_flight(100.0), self._mock_flight(200.0)]
        assert self._all_unpriced(flights) is False

    def test_object_without_price_attribute_treated_as_zero(self) -> None:
        """Getattr fallback: object with no price attribute counts as unpriced."""
        obj = object()
        assert self._all_unpriced([obj]) is True


# ---------------------------------------------------------------------------
# Unit tests for _build_per_leg_fallback_filters
# ---------------------------------------------------------------------------


class TestBuildPerLegFallbackFilters:
    """Per-leg fallback filters target a single leg in ONE_WAY mode.

    ``_build_per_leg_fallback_filters`` must produce a ONE_WAY filter
    for only the current leg, copying filter fields from the original.
    """

    def _make_filters(self, n_legs=4):
        """Build a minimal FlightSearchFilters-shaped mock."""
        from fli.models import Airport, FlightSearchFilters, FlightSegment, PassengerInfo

        today = datetime.now()
        segments = []
        for i in range(n_legs):
            seg = FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.LHR, 0]],
                travel_date=(today + timedelta(days=30 + i * 3)).strftime("%Y-%m-%d"),
            )
            segments.append(seg)

        return FlightSearchFilters(
            trip_type=TripType.MULTI_CITY,
            passenger_info=PassengerInfo(adults=2),
            flight_segments=segments,
        )

    def test_result_is_one_way(self) -> None:
        from fli.mcp.server import _build_per_leg_fallback_filters

        filters = self._make_filters(n_legs=4)
        fb = _build_per_leg_fallback_filters(filters, current_step=0)
        assert fb.trip_type == TripType.ONE_WAY

    def test_result_has_exactly_one_segment(self) -> None:
        from fli.mcp.server import _build_per_leg_fallback_filters

        filters = self._make_filters(n_legs=4)
        fb = _build_per_leg_fallback_filters(filters, current_step=0)
        assert len(fb.flight_segments) == 1

    def test_uses_current_step_segment_not_step_zero(self) -> None:
        """Step 2 fallback must use segment[2], not segment[0]."""
        from fli.mcp.server import _build_per_leg_fallback_filters
        from fli.models import Airport, FlightSearchFilters, FlightSegment, PassengerInfo

        today = datetime.now()

        def d(i: int) -> str:
            return (today + timedelta(days=30 + i * 3)).strftime("%Y-%m-%d")

        airports = [Airport.JFK, Airport.LHR, Airport.CDG, Airport.AMS]
        segments = [
            FlightSegment(
                departure_airport=[[airports[i], 0]],
                arrival_airport=[[airports[(i + 1) % 4], 0]],
                travel_date=d(i),
            )
            for i in range(4)
        ]
        filters = FlightSearchFilters(
            trip_type=TripType.MULTI_CITY,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=segments,
        )

        fb = _build_per_leg_fallback_filters(filters, current_step=2)
        # Segment 2 departs CDG; segment 0 departs JFK — they must differ.
        assert fb.flight_segments[0].departure_airport == [[Airport.CDG, 0]]

    def test_selected_flight_is_cleared(self) -> None:
        """The fallback segment must have ``selected_flight=None``.

        Even if the original has a selection set (which can happen
        during continuation steps), the fallback's segment is reset.
        """
        from fli.mcp.server import _build_per_leg_fallback_filters
        from fli.models import Airport, FlightSearchFilters, FlightSegment, PassengerInfo

        today = datetime.now()
        d = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        seg = FlightSegment(
            departure_airport=[[Airport.JFK, 0]],
            arrival_airport=[[Airport.LHR, 0]],
            travel_date=d,
        )
        fake_flight = MagicMock()
        seg.selected_flight = fake_flight

        filters = FlightSearchFilters(
            trip_type=TripType.MULTI_CITY,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=[
                seg,
                FlightSegment(
                    departure_airport=[[Airport.LHR, 0]],
                    arrival_airport=[[Airport.CDG, 0]],
                    travel_date=d,
                ),
            ],
        )
        fb = _build_per_leg_fallback_filters(filters, current_step=0)
        assert fb.flight_segments[0].selected_flight is None

    def test_passenger_info_preserved(self) -> None:
        from fli.mcp.server import _build_per_leg_fallback_filters

        filters = self._make_filters(n_legs=2)
        fb = _build_per_leg_fallback_filters(filters, current_step=0)
        assert fb.passenger_info.adults == 2


# ---------------------------------------------------------------------------
# Multi-city session error handling (timeout vs non-timeout)
# ---------------------------------------------------------------------------


class TestMultiCitySessionErrorHandling:
    """Session lifecycle when ``_execute_multi_city_step`` raises.

    Timeout exceptions must preserve the cached session so the next
    retry resumes from the same leg.  All other exceptions (ParseError,
    generic) must discard the session.
    """

    def _two_leg_params(self) -> MultiCitySearchParams:
        return MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="JFK", destination="LHR", date=_future_date(30)),
                MultiCityLeg(origin="LHR", destination="CDG", date=_future_date(35)),
            ]
        )

    def test_timeout_exception_preserves_session(self) -> None:
        """A timeout during the multi-city API call must NOT evict the session."""
        params = self._two_leg_params()

        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            # First call succeeds and seeds the session.
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                ([_flight_result("BA", "117", 350.0)], {}),
                # Step 2: timeout on the continuation call
                Exception("Operation timed out after 25s"),
            ]

            step1 = _execute_multi_city_step(params, selection=None)
            assert step1["success"] is True
            assert key in _search_sessions

            step2 = _execute_multi_city_step(params, selection=0)
            # Timeout must be classified and returned …
            assert step2["success"] is False
            assert step2["error_kind"] == "timeout"
            # … and the session must still be present for a retry.
            assert key in _search_sessions

    def test_parse_error_discards_session(self) -> None:
        """A ParseError on the fresh-session path must not leave a session behind.

        ParseError fires during parameter parsing (``parse_cabin_class``,
        ``resolve_origin_or_city``, etc.) which only runs on ``selection=None``
        (the initial leg-1 call).  The test verifies that the session is NOT
        stored when that path raises.
        """
        from fli.core.parsers import ParseError

        params = self._two_leg_params()

        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)

        # Patch a parser that runs early in the fresh-session path.
        with patch(
            "fli.mcp.server.resolve_origin_or_city",
            side_effect=ParseError("bad airport code"),
        ):
            result = _execute_multi_city_step(params, selection=None)

        assert result["success"] is False
        assert result["error_kind"] == "parse"
        assert key not in _search_sessions

    def test_generic_exception_discards_session(self) -> None:
        """A non-timeout exception during the search must evict the session."""
        params = self._two_leg_params()

        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            instance._do_single_search.side_effect = [
                ([_flight_result("BA", "117", 350.0)], {}),
                Exception("unexpected serialization failure"),
            ]

            _execute_multi_city_step(params, selection=None)
            assert key in _search_sessions

            result = _execute_multi_city_step(params, selection=0)

        assert result["success"] is False
        assert result["error_kind"] == "unknown"
        assert key not in _search_sessions

    def test_empty_step_on_fresh_start_does_not_store_session(self) -> None:
        """First-call empty results must not create a session entry.

        When the very first call returns empty (both multi-city and
        fallback), no session is stored — there are no results to
        resume from.
        """
        params = self._two_leg_params()

        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)

        with patch("fli.mcp.server.SearchFlights") as mock_cls:
            instance = mock_cls.return_value
            # Both multi-city and fallback return empty.
            instance._do_single_search.side_effect = [None, None]

            result = _execute_multi_city_step(params, selection=None)

        assert result["count"] == 0
        assert "hint" in result
        # No session stored — nothing to resume from; the retry is a fresh call.
        assert key not in _search_sessions

    def test_out_of_range_selection_returns_error_kind(self) -> None:
        """Selection beyond the result count must return error_kind='validation'."""
        params = self._two_leg_params()

        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(params)
        fake_filters = MagicMock()
        fake_filters.flight_segments = [MagicMock(selected_flight=None)] * 2
        _search_sessions[key] = (fake_filters, ["flight_a"], False)

        result = _execute_multi_city_step(params, selection=5)
        assert result["success"] is False
        assert result["error_kind"] == "validation"
        assert "out of range" in result["error"]


# ---------------------------------------------------------------------------
# _session_cache_key additional coverage
# ---------------------------------------------------------------------------


class TestSessionCacheKeyAdditional:
    """Additional edge-case coverage for _session_cache_key."""

    def _params(self, **overrides):
        base = {
            "legs": [
                MultiCityLeg(origin="JFK", destination="LHR", date="2027-06-01"),
                MultiCityLeg(origin="LHR", destination="CDG", date="2027-06-05"),
            ],
        }
        base.update(overrides)
        return MultiCitySearchParams(**base)

    def test_key_contains_legs_part_and_digest(self) -> None:
        """Key must have at least one '|' separator between legs part and digest."""
        from fli.mcp.server import _session_cache_key

        key = _session_cache_key(self._params())
        assert "|" in key

    def test_changing_passengers_misses_cache(self) -> None:
        from fli.mcp.server import _session_cache_key

        one = _session_cache_key(self._params(passengers=1))
        two = _session_cache_key(self._params(passengers=2))
        assert one != two

    def test_changing_exclude_basic_economy_misses_cache(self) -> None:
        from fli.mcp.server import _session_cache_key

        incl = _session_cache_key(self._params(exclude_basic_economy=False))
        excl = _session_cache_key(self._params(exclude_basic_economy=True))
        assert incl != excl

    def test_changing_checked_bags_misses_cache(self) -> None:
        from fli.mcp.server import _session_cache_key

        no_bags = _session_cache_key(self._params(checked_bags=0))
        one_bag = _session_cache_key(self._params(checked_bags=1))
        assert no_bags != one_bag

    def test_changing_carry_on_misses_cache(self) -> None:
        from fli.mcp.server import _session_cache_key

        no_co = _session_cache_key(self._params(carry_on=False))
        with_co = _session_cache_key(self._params(carry_on=True))
        assert no_co != with_co

    def test_changing_departure_window_misses_cache(self) -> None:
        from fli.mcp.server import _session_cache_key

        no_window = _session_cache_key(self._params(departure_window=None))
        with_window = _session_cache_key(self._params(departure_window="6-20"))
        assert no_window != with_window
