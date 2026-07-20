"""
broker_superr.py
-----------------------------------------------------------------------------
Superr API (Stoxkart) wrapper -- for the LIVE/PAPER EXECUTION phase, NOT
used anywhere in the backtest. Not imported by run_backtest.py.

This is built from Stoxkart's own public developer docs (developers.stoxkart.com)
and their reference implementation on GitHub (StoxKart-Tech/superrapi-python).
Worth knowing before you plug in real credentials: that reference file has a
real structural bug -- ModifyOrder / cancelorder / the AMO and BO variants are
all defined as `def` statements INSIDE placeOrder()'s body, at the wrong
indentation level, which makes them local functions that vanish as soon as
placeOrder() returns rather than usable class methods. I did not carry that
bug forward; every method below is a proper method on SuperrClient. I also
could not confirm an exact historical-candle REST path from public docs --
check the "API Documentation" section at https://developers.stoxkart.com
directly before relying on get_ltp_snapshot() for anything beyond a live
quote check.

Auth flow (matches Stoxkart's documented 3-step login):
  1. POST /auth/login          -> request_token (prompts for TOTP if 2FA is on)
  2. POST /auth/twofa/verify   -> validated request_token
  3. Build HMAC-SHA256(api_secret, api_key + validated_request_token),
     POST /auth/token          -> access_token, used on every subsequent call
-----------------------------------------------------------------------------
"""
import hashlib
import hmac
import json

import requests


def market_protected_limit_price(ltp: float, action: str) -> float:
    """
    SEBI's current retail-algo framework doesn't let algo orders go out as
    bare market orders anymore -- brokers now require (or auto-apply) a
    protection band around LTP instead, confirmed across multiple brokers'
    own developer docs/forums (Kite Connect, Angel One SmartAPI) under the
    same SEBI rule. Rather than send a raw MARKET order and trust Superr's
    auto-conversion to do something sensible, this computes an explicit
    LIMIT price with a protection band and sends that -- band widens as
    price falls, matching the tiered pattern other brokers publish:
        > Rs 500  -> 0.5%
        Rs 100-500 -> 1%
        < Rs 100   -> 2%
    Confirm Stoxkart's exact bands in their current docs before relying on
    this for size that matters -- the tiers above are the pattern observed
    elsewhere in the industry under the same SEBI rule, not something
    confirmed in Superr's docs specifically.
    """
    if ltp > 500:
        band = 0.005
    elif ltp >= 100:
        band = 0.01
    else:
        band = 0.02
    return round(ltp * (1 + band), 2) if action.upper() == "BUY" else round(ltp * (1 - band), 2)


class SuperrClient:
    ROOT_URL = "https://openapi.stoxkart.com"

    ENDPOINTS = {
        "login": "/auth/login",
        "twofa": "/auth/twofa/verify",
        "token": "/auth/token",
        "fund_details": "/fund/v1/fund-details",
        "positions": "/portfolio/v1/position",
        "holdings": "/portfolio/v1/holding",
        "order_book": "/report/v1/order-book",
        "trade_book": "/report/v1/getTradeBook",
        "position_conversion": "/order/v1/position-conversion",
        "order_normal": "/orders/normal",
        "order_amo": "/orders/amo",
        "order_bo": "/orders/bo",
    }

    def __init__(self, api_key: str, client_id: str, password: str, api_secret: str):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.api_secret = api_secret
        self.request_token = None
        self.access_token = None

    # ---- auth --------------------------------------------------------
    def _url(self, name: str, with_api_key: bool = False) -> str:
        url = self.ROOT_URL + self.ENDPOINTS[name]
        if with_api_key:
            url += f"?api-key={self.api_key}"
        return url

    def login(self):
        """Step 1: authenticate with client_id + password. Raises if the
        server rejects the request. Sets self.request_token; if 2FA is
        enabled, call verify_totp(otp) next before generate_session()."""
        payload = {"platform": "api", "data": {"client_id": self.client_id, "password": self.password}}
        resp = requests.post(self._url("login", with_api_key=True),
                              data=json.dumps(payload),
                              headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        body = resp.json()
        self.request_token = body["data"]["request_token"]
        return body["data"].get("is_2fa_enabled", False)

    def verify_totp(self, otp: str):
        """Step 2: only needed if login() reported is_2fa_enabled=True."""
        payload = {"platform": "api", "data": {
            "client_id": self.client_id, "req_token": self.request_token,
            "action": "api-key-validation", "otp": otp,
        }}
        resp = requests.post(self._url("twofa", with_api_key=True),
                              data=json.dumps(payload),
                              headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        self.request_token = resp.json()["data"]["request_token"]

    def generate_session(self):
        """Step 3: exchange the validated request_token for an access_token
        using an HMAC-SHA256 signature, as Stoxkart's docs specify."""
        signature = hmac.new(
            key=self.api_secret.encode("utf-8"),
            msg=(self.api_key + self.request_token).encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        payload = {"api_key": self.api_key, "signature": signature, "req_token": self.request_token}
        resp = requests.post(self._url("token"), data=json.dumps(payload),
                              headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        # NB: Stoxkart's response labels this field "request_token" even
        # though it's functioning as the access token from here on -- that's
        # their API's naming, not a typo in this client.
        self.access_token = resp.json()["data"]["request_token"]

    def authenticate(self, otp_provider=None):
        """Convenience wrapper: runs the full 3-step flow. otp_provider is a
        zero-arg callable returning the TOTP string (e.g. `lambda: input('OTP: ')`)
        -- only invoked if the account actually has 2FA enabled."""
        needs_2fa = self.login()
        if needs_2fa:
            otp = otp_provider() if otp_provider else input("Enter TOTP: ")
            self.verify_totp(otp)
        self.generate_session()

    def _auth_headers(self) -> dict:
        if not self.access_token:
            raise RuntimeError("Not authenticated yet -- call authenticate() first.")
        return {
            "Content-Type": "application/json",
            "Accept-Type": "application/json",
            "request_token": self.request_token,
            "x-Platform": "api",
            "X-Access-Token": self.access_token,
            "x-Api-key": self.api_key,
        }

    # ---- account / portfolio ------------------------------------------
    def fund_details(self) -> dict:
        return requests.get(self._url("fund_details"), headers=self._auth_headers()).json()

    def positions(self) -> dict:
        return requests.get(self._url("positions"), headers=self._auth_headers()).json()

    def holdings(self) -> dict:
        return requests.get(self._url("holdings"), headers=self._auth_headers()).json()

    def order_book(self) -> dict:
        return requests.get(self._url("order_book"), headers=self._auth_headers()).json()

    def trade_book(self) -> dict:
        return requests.get(self._url("trade_book"), headers=self._auth_headers()).json()

    # ---- orders ---------------------------------------------------
    def quote(self, exchange: str, tokens: list) -> dict:
        """
        Live LTP/OHLC snapshot for up to 50 instruments per call, per
        Stoxkart's public API docs. I could not confirm the exact endpoint
        PATH for this from public documentation with full confidence --
        check the "Instruments" section at developers.stoxkart.com and
        adjust ENDPOINTS['quote'] below if it doesn't match. Everything
        else in this file (auth, orders, positions/holdings/funds) is
        confirmed against their docs and reference SDK.
        """
        url = self.ROOT_URL + "/instruments/v1/quote"   # VERIFY against current docs
        params = {"exchange": exchange, "tokens": ",".join(tokens)}
        resp = requests.get(url, params=params, headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json()

    def place_protected_order(self, action: str, exchange: str, token: str,
                               product_type: str, quantity: int, ltp: float,
                               stop_loss: float = 0, tag: str = "", variety: str = "normal") -> dict:
        """
        Convenience wrapper: builds a market_protected_limit_price(...) from
        the LTP you pass in and sends it as a LIMIT order, rather than a bare
        MARKET order -- this is the method run_live.py actually calls. Use
        place_order() directly only if you specifically want to hand-pick a
        different limit price yourself.
        """
        limit_price = market_protected_limit_price(ltp, action)
        return self.place_order(
            action=action, exchange=exchange, token=token, order_type="LIMIT",
            product_type=product_type, quantity=quantity, price=limit_price,
            stop_loss=stop_loss, tag=tag, variety=variety,
        )

    def place_order(self, action: str, exchange: str, token: str, order_type: str,
                     product_type: str, quantity: int, price: float = 0,
                     trigger_price: float = 0, stop_loss: float = 0,
                     trailing_stop_loss: float = 0, disclose_quantity: int = 0,
                     validity: str = "DAY", validity_date: str = "", tag: str = "",
                     variety: str = "normal") -> dict:
        """
        action: 'BUY' or 'SELL'
        exchange: e.g. 'NSE'
        token: the exchange instrument token (NOT the ticker symbol -- map
               symbol -> token via the instruments master dump first)
        order_type: 'MARKET' / 'LIMIT' / 'SL' / 'SL-M' (confirm exact enum
               strings against current docs before going live)
        product_type: 'DELIVERY' / 'INTRADAY' / 'MARGIN' / 'BRACKET_ORDER'
               -- 'INTRADAY' is what this strategy uses for same-day legs;
               carried (multi-day) positions need 'DELIVERY'.
        variety: 'normal', 'amo', or 'bo' -- selects which endpoint is used.
        """
        payload = {
            "action": action, "exchange": exchange, "token": token,
            "order_type": order_type, "product_type": product_type,
            "quantity": quantity, "disclose_quantity": disclose_quantity,
            "price": price, "trigger_price": trigger_price,
            "square_off": 0, "stop_loss": stop_loss,
            "trailing_stop_loss": trailing_stop_loss,
            "validity": validity, "validity_date": validity_date, "tag": tag,
        }
        endpoint = {"normal": "order_normal", "amo": "order_amo", "bo": "order_bo"}[variety]
        resp = requests.post(self._url(endpoint), data=json.dumps(payload), headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json()

    def convert_position(self, action: str, exchange: str, token: str, order_type: str,
                          product_type: str, quantity: int, new_product_type: str) -> dict:
        """E.g. converting an INTRADAY fill to DELIVERY when a trend position
        earns the right to carry past the same-day square-off -- confirm this
        is how Stoxkart wants same-day-to-carried transitions handled before
        relying on it; the alternative is simply not squaring off the
        INTRADAY-tagged leg and instead holding it, if the product type
        allows that. Check current docs."""
        payload = {"action": action, "exchange": exchange, "token": token,
                   "order_type": order_type, "product_type": product_type,
                   "quantity": quantity, "new_product_type": new_product_type}
        resp = requests.post(self._url("position_conversion"), data=json.dumps(payload),
                              headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    print("This module is for the live/paper execution phase, not backtesting.")
    print("Example usage once you're ready:")
    print("""
    client = SuperrClient(api_key="...", client_id="...", password="...", api_secret="...")
    client.authenticate()   # prompts for TOTP if your account has 2FA on
    print(client.fund_details())
    """)
