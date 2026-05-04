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
from curl_cffi.requests.exceptions import (
    ConnectionError as CurlConnectionError,
    HTTPError,
    Timeout,
)
from ratelimit import limits, sleep_and_retry
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)


def _is_retriable_request_failure(exc: BaseException) -> bool:
    """Predicate for tenacity: only retry transient request failures.

    Without this, the default ``retry-everything`` policy retries
    permanent 4xx responses and local programming errors three times
    each, which burns the MCP transport budget on requests that will
    fail identically every attempt.

    Retried:
      * Connection failures and DNS failures (network blip).
      * Timeouts (Google's own backend stalled — worth one more shot).
      * 5xx and 429 (genuinely transient on the upstream).

    Not retried:
      * Other 4xx (auth, bad request, not found — won't change).
      * Anything that isn't an HTTP/network error.
    """
    if isinstance(exc, (Timeout, CurlConnectionError)):
        return True
    if isinstance(exc, HTTPError):
        # ``raise_for_status`` attaches the response; older variants may
        # not have ``response``, so default to "transient" if we can't
        # see the status — better to retry once than miss a real 5xx.
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if status is None:
            return True
        return status >= 500 or status == 429
    return False

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
        # Only retry transient HTTP / network failures.  Permanent 4xx
        # responses, programming errors, and other non-network failures
        # don't get any additional attempts because they would just fail
        # identically and burn the MCP budget.
        retry=retry_if_exception(_is_retriable_request_failure),
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
            curl_cffi.requests.exceptions.RequestException: The original
                exception type from ``curl_cffi`` is preserved (Timeout,
                ConnectionError, HTTPError, etc.).  Callers that want to
                discriminate by failure mode can ``isinstance``-check.

        """
        kwargs.setdefault("timeout", self.DEFAULT_TIMEOUT)
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

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
        # Only retry transient HTTP / network failures.  See ``get`` for
        # the rationale.
        retry=retry_if_exception(_is_retriable_request_failure),
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
            curl_cffi.requests.exceptions.RequestException: The original
                exception type from ``curl_cffi`` is preserved.  Callers
                that need to branch on the failure mode (e.g. the MCP
                error classifier) can ``isinstance``-check the
                exception rather than parsing the message.

        """
        kwargs.setdefault("timeout", self.DEFAULT_TIMEOUT)
        response = self._client.post(url, **kwargs)
        response.raise_for_status()
        return response


def get_client() -> Client:
    """Get or create a shared HTTP client instance.

    Returns:
        Singleton instance of the HTTP client

    """
    global client
    if not client:
        client = Client()
    return client
