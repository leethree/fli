"""HTTP client implementation with impersonation, rate limiting and retry functionality.

This module provides a robust HTTP client that handles:
- User agent impersonation (to mimic a browser)
- Rate limiting (10 requests per second)
- Automatic retries with exponential backoff
- Session management
- Error handling
"""

from typing import Any

from curl_cffi import requests
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential

client = None


class Client:
    """HTTP client with built-in rate limiting, retry and user agent impersonation functionality."""

    DEFAULT_HEADERS = {
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    }
    # Per-attempt HTTP timeout.  Sized to fit *one* ``Client.post`` call
    # within typical MCP transport budgets (~30 s client-side request
    # timeout in Claude Desktop and similar).  Google's
    # ``GetShoppingResults`` either responds quickly (< 5 s warm) or
    # hangs until its own ~60 s server-side timeout — the latter is a
    # transient backend stall, so capping our wait short and surfacing
    # the empty/timeout outcome lets the calling agent retry on a fresh
    # MCP-level budget instead of having one long call eaten by the
    # transport.
    DEFAULT_TIMEOUT = 25
    # Total wall-time cap on the tenacity retry loop, in seconds.  Just
    # under typical MCP client request timeouts (~30 s) so a hung
    # request followed by tenacity retries can't accumulate past the
    # transport budget.  ``stop_after_delay`` is checked between
    # attempts; combined with ``DEFAULT_TIMEOUT`` it bounds the worst
    # case to a single full-timeout attempt rather than three.
    RETRY_TOTAL_DELAY = 24

    def __init__(self):
        """Initialize a new client session with default headers."""
        self._client = requests.Session()
        self._client.headers.update(self.DEFAULT_HEADERS)

    def __del__(self):
        """Clean up client session on deletion."""
        if hasattr(self, "_client"):
            self._client.close()

    @sleep_and_retry
    @limits(calls=10, period=1)
    @retry(
        # Stop on either condition: 3 attempts max, OR total wall time
        # past the MCP-friendly budget.  The delay cap prevents a single
        # hung request from spawning two more 25 s waits on retry —
        # which is the failure mode that pushed worst-case wall to 78 s
        # and ate the MCP transport budget.
        stop=(stop_after_attempt(3) | stop_after_delay(RETRY_TOTAL_DELAY)),
        wait=wait_exponential(multiplier=0.1, max=1),
        reraise=True,
    )
    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited GET request with automatic retries.

        Args:
            url: Target URL for the request
            **kwargs: Additional arguments passed to requests.get()

        Returns:
            Response object from the server

        Raises:
            Exception: If request fails after all retries

        """
        try:
            kwargs.setdefault("timeout", self.DEFAULT_TIMEOUT)
            response = self._client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"GET request failed: {str(e)}") from e

    @sleep_and_retry
    @limits(calls=10, period=1)
    @retry(
        # Stop on either condition: 3 attempts max, OR total wall time
        # past the MCP-friendly budget.  The delay cap prevents a single
        # hung request from spawning two more 25 s waits on retry —
        # which is the failure mode that pushed worst-case wall to 78 s
        # and ate the MCP transport budget.
        stop=(stop_after_attempt(3) | stop_after_delay(RETRY_TOTAL_DELAY)),
        wait=wait_exponential(multiplier=0.1, max=1),
        reraise=True,
    )
    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited POST request with automatic retries.

        Args:
            url: Target URL for the request
            **kwargs: Additional arguments passed to requests.post()

        Returns:
            Response object from the server

        Raises:
            Exception: If request fails after all retries

        """
        try:
            kwargs.setdefault("timeout", self.DEFAULT_TIMEOUT)
            response = self._client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"POST request failed: {str(e)}") from e


def get_client() -> Client:
    """Get or create a shared HTTP client instance.

    Returns:
        Singleton instance of the HTTP client

    """
    global client
    if not client:
        client = Client()
    return client
