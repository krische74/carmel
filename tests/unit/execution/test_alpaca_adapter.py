"""AlpacaBrokerAdapter unit tests (TradingClient mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.execution.alpaca_adapter import AlpacaBrokerAdapter


def test_alpaca_adapter_reads_equity_and_cash() -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(
        equity=12_345.67, cash=4321.0, last_equity=12_000.0
    )
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_account_equity() == 12_345.67
    assert adapter.get_cash() == 4321.0


def test_alpaca_adapter_returns_last_equity() -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(last_equity=99_999.5)
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_last_equity() == 99_999.5


def test_alpaca_adapter_position_qty_from_all_positions() -> None:
    client = MagicMock()
    p = MagicMock()
    p.symbol = "SPY"
    p.qty = "4"
    client.get_all_positions.return_value = [p]
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_position_qty("SPY") == 4.0
    assert adapter.get_position_qty("QQQ") == 0.0


@patch("src.execution.alpaca_adapter.TradingClient")
def test_alpaca_adapter_create_uses_trading_client(mock_tc: MagicMock) -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(
        equity=100.0, cash=50.0, last_equity=99.0, id="acct-1"
    )
    mock_tc.return_value = client
    adapter = AlpacaBrokerAdapter.create("key", "secret", paper=True)
    mock_tc.assert_called_once_with("key", "secret", paper=True)
    assert isinstance(adapter, AlpacaBrokerAdapter)
    client.get_account.assert_called_once()


def test_alpaca_adapter_get_account_id_uses_logical_when_set() -> None:
    client = MagicMock()
    adapter = AlpacaBrokerAdapter(client, logical_account_id="ira_roth")
    assert adapter.get_account_id() == "ira_roth"
    client.get_account.assert_not_called()


def test_alpaca_adapter_get_account_id_falls_back_to_broker_account_id() -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(id="acct-uuid-9")
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_account_id() == "acct-uuid-9"


def test_alpaca_adapter_submit_market_order_returns_order_id() -> None:
    client = MagicMock()
    client.submit_order.return_value = MagicMock(id="order-uuid-1")
    adapter = AlpacaBrokerAdapter(client)
    oid = adapter.submit_market_order("spy", 1.5, "buy")
    assert oid == "order-uuid-1"
    client.submit_order.assert_called_once()
    req = client.submit_order.call_args[0][0]
    assert getattr(req, "client_order_id", None) is not None
    assert str(req.client_order_id).startswith("cml-")


def test_alpaca_adapter_submit_market_order_passes_explicit_client_order_id() -> None:
    client = MagicMock()
    client.submit_order.return_value = MagicMock(id="order-uuid-2")
    adapter = AlpacaBrokerAdapter(client)
    oid = adapter.submit_market_order(
        "qqq",
        2.0,
        "buy",
        client_order_id="my-id-1",
    )
    assert oid == "order-uuid-2"
    req = client.submit_order.call_args[0][0]
    assert req.client_order_id == "my-id-1"


def test_cached_account_calls_client_once_for_multiple_getters() -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(equity=100.0, cash=50.0, last_equity=99.0)
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_account_equity() == 100.0
    assert adapter.get_cash() == 50.0
    assert adapter.get_last_equity() == 99.0
    client.get_account.assert_called_once()


def test_refresh_account_re_fetches() -> None:
    client = MagicMock()
    client.get_account.side_effect = [
        MagicMock(equity=100.0, cash=50.0, last_equity=99.0),
        MagicMock(equity=200.0, cash=60.0, last_equity=100.0),
    ]
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_account_equity() == 100.0
    adapter.refresh_account()
    assert adapter.get_account_equity() == 200.0
    assert client.get_account.call_count == 2


def test_account_auto_fetches_on_first_access() -> None:
    client = MagicMock()
    client.get_account.return_value = MagicMock(equity=1.0, cash=1.0, last_equity=1.0)
    adapter = AlpacaBrokerAdapter(client)
    adapter.get_cash()
    client.get_account.assert_called_once()


def test_get_order_fill_price_returns_filled_avg_price() -> None:
    client = MagicMock()
    client.get_order_by_id.return_value = MagicMock(filled_avg_price=123.45)
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_order_fill_price("ord-1") == pytest.approx(123.45)
    client.get_order_by_id.assert_called_once_with("ord-1")


def test_get_order_fill_returns_qty_and_price_from_one_lookup() -> None:
    client = MagicMock()
    client.get_order_by_id.return_value = MagicMock(
        filled_qty="6.133477", filled_avg_price="91.55"
    )
    adapter = AlpacaBrokerAdapter(client)
    qty, price = adapter.get_order_fill("e93fe3f7-c8e2-48e3-b56c-cdc4b8cc4917")
    assert qty == pytest.approx(6.133477)
    assert price == pytest.approx(91.55)
    client.get_order_by_id.assert_called_once_with("e93fe3f7-c8e2-48e3-b56c-cdc4b8cc4917")
    assert adapter.get_order_fill_price("e93fe3f7-c8e2-48e3-b56c-cdc4b8cc4917") == pytest.approx(
        91.55
    )


def test_get_order_fill_raises_when_order_missing() -> None:
    from alpaca.common.exceptions import APIError

    from src.execution.errors import OrderNotFoundError

    client = MagicMock()
    client.get_order_by_id.side_effect = APIError(
        '{"code":40410000,"message":"order not found"}', http_error=None
    )
    adapter = AlpacaBrokerAdapter(client)
    with pytest.raises(OrderNotFoundError):
        adapter.get_order_fill("gone")


def test_get_order_fill_price_returns_none_when_not_filled() -> None:
    client = MagicMock()
    client.get_order_by_id.return_value = MagicMock(filled_avg_price=None)
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_order_fill_price("ord-2") is None


def test_alpaca_adapter_list_recent_orders_maps_fields() -> None:
    from datetime import UTC, datetime

    from alpaca.trading.enums import OrderSide, OrderStatus

    o = MagicMock()
    o.id = "o-recon-1"
    o.symbol = "SPY"
    o.side = OrderSide.BUY
    o.qty = 3.0
    o.filled_qty = 3.0
    o.filled_avg_price = 100.25
    o.status = OrderStatus.FILLED
    o.client_order_id = "cml-abc123"
    o.submitted_at = datetime(2026, 6, 9, 17, 30, 11, tzinfo=UTC)
    o.created_at = None
    client = MagicMock()
    client.get_orders.return_value = [o]
    adapter = AlpacaBrokerAdapter(client)
    rows = adapter.list_recent_orders(limit=50)
    assert len(rows) == 1
    assert rows[0]["order_id"] == "o-recon-1"
    assert rows[0]["symbol"] == "SPY"
    assert rows[0]["side"] == "buy"
    assert rows[0]["filled_qty"] == 3.0
    assert rows[0]["filled_avg_price"] == 100.25
    assert rows[0]["client_order_id"] == "cml-abc123"
    assert rows[0]["timestamp"] == "2026-06-09T17:30:11+00:00"
    client.get_orders.assert_called_once()


def test_get_order_by_client_id_maps_fill_fields() -> None:
    from datetime import UTC, datetime

    from alpaca.trading.enums import OrderSide, OrderStatus

    from src.execution.errors import OrderNotFoundError

    o = MagicMock()
    o.id = "broker-oid-1"
    o.client_order_id = "cml-6ce5bc859237079c31bf1fb5b39f82a7"
    o.symbol = "VOO"
    o.side = OrderSide.BUY
    o.qty = 0.011
    o.filled_qty = 0.011
    o.filled_avg_price = 700.0
    o.status = OrderStatus.FILLED
    o.submitted_at = datetime(2026, 6, 9, 17, 30, 11, tzinfo=UTC)
    client = MagicMock()
    client.get_order_by_client_id.return_value = o
    adapter = AlpacaBrokerAdapter(client)
    row = adapter.get_order_by_client_id("cml-6ce5bc859237079c31bf1fb5b39f82a7")
    assert row["order_id"] == "broker-oid-1"
    assert row["client_order_id"] == "cml-6ce5bc859237079c31bf1fb5b39f82a7"
    assert row["symbol"] == "VOO"
    assert row["side"] == "buy"
    assert row["qty"] == 0.011
    assert row["filled_qty"] == 0.011
    assert row["filled_avg_price"] == 700.0
    assert row["status"] == str(OrderStatus.FILLED)
    assert "2026-06-09T17:30:11" in str(row["timestamp"])
    client.get_order_by_client_id.assert_called_once_with(
        "cml-6ce5bc859237079c31bf1fb5b39f82a7",
    )

    from alpaca.common.exceptions import APIError

    client.get_order_by_client_id.side_effect = APIError(
        '{"code":40410000,"message":"order not found"}',
        http_error=None,
    )
    with pytest.raises(OrderNotFoundError):
        adapter.get_order_by_client_id("missing-cid")


def test_get_account_activities_returns_unfiltered_dicts() -> None:
    from datetime import date

    client = MagicMock()
    client.get.return_value = [
        {
            "id": "act-1",
            "activity_type": "FILL",
            "symbol": "BIL",
            "qty": "9.4",
            "side": "sell",
            "transaction_time": "2026-08-19T17:45:00Z",
        },
        {
            "id": "act-2",
            "activity_type": "JNLC",
            "net_amount": "100.0",
            "date": "2026-08-20",
        },
    ]
    adapter = AlpacaBrokerAdapter(client)
    rows = adapter.get_account_activities(
        after=date(2026, 8, 18),
        until=date(2026, 8, 21),
    )
    assert len(rows) == 2
    assert rows[0]["activity_type"] == "FILL"
    assert rows[0]["symbol"] == "BIL"
    assert rows[1]["activity_type"] == "JNLC"
    client.get.assert_called()
    call_args = client.get.call_args
    assert call_args.args[0] == "/account/activities"
    params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("data")
    assert params is not None
    assert "activity_types" not in params
    assert "after" in params
    assert "until" in params


def test_submit_market_order_invalidates_account_cache() -> None:
    client = MagicMock()
    client.get_account.side_effect = [
        MagicMock(equity=100.0, cash=50.0, last_equity=99.0),
        MagicMock(equity=100.0, cash=10.0, last_equity=99.0),
    ]
    client.submit_order.return_value = MagicMock(id="mkt-1")
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_cash() == 50.0
    adapter.submit_market_order("SPY", 1.0, "buy")
    assert adapter.get_cash() == 10.0
    assert client.get_account.call_count == 2


def test_submit_limit_order_invalidates_account_cache() -> None:
    client = MagicMock()
    client.get_account.side_effect = [
        MagicMock(equity=100.0, cash=50.0, last_equity=99.0),
        MagicMock(equity=100.0, cash=20.0, last_equity=99.0),
    ]
    client.submit_order.return_value = MagicMock(id="lmt-1")
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_cash() == 50.0
    adapter.submit_limit_order("SPY", 1.0, "buy", limit_price=400.0)
    assert adapter.get_cash() == 20.0
    assert client.get_account.call_count == 2


def test_submit_stop_order_invalidates_account_cache() -> None:
    client = MagicMock()
    client.get_account.side_effect = [
        MagicMock(equity=100.0, cash=50.0, last_equity=99.0),
        MagicMock(equity=100.0, cash=30.0, last_equity=99.0),
    ]
    client.submit_order.return_value = MagicMock(id="stp-1")
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.get_cash() == 50.0
    adapter.submit_stop_order("SPY", 1.0, "sell", stop_price=390.0)
    assert adapter.get_cash() == 30.0
    assert client.get_account.call_count == 2


def test_snapshot_from_broker_sees_post_submit_cash_mutation() -> None:
    from src.portfolio.state import snapshot_from_broker

    client = MagicMock()
    client.get_account.side_effect = [
        MagicMock(equity=3433.79, cash=193.96, last_equity=3433.79),
        MagicMock(equity=3433.79, cash=1052.91, last_equity=3433.79),
    ]
    client.get_all_positions.return_value = []
    client.submit_order.return_value = MagicMock(id="rot-1")
    adapter = AlpacaBrokerAdapter(client)
    first = snapshot_from_broker(adapter, ["SPY", "SHV"])
    assert first.cash == pytest.approx(193.96)
    adapter.submit_market_order("SHV", 7.79, "sell")
    second = snapshot_from_broker(adapter, ["SPY", "SHV"])
    assert second.cash == pytest.approx(1052.91)


def test_leverage_snapshot_fully_paid_longs_are_not_leverage() -> None:
    """Regression: non-zero maintenance margin is not a borrowing signal.

    Real 2026-09-10 paper-account figures. Equity 3404.22 with cash 239.64 leaves
    3164.58 of long market value, against which Alpaca posts its standard 30%
    house requirement of 949.37. The predicate that used to live in
    ``get_leverage_snapshot`` read that non-zero maintenance as leverage and
    raised a CRITICAL on every cycle for three weeks on an account that had
    never borrowed.
    """
    client = MagicMock()
    client.get_account.return_value = MagicMock(
        cash=239.64,
        equity=3404.22,
        long_market_value=3164.58,
        maintenance_margin=949.37,
        buying_power=479.28,
    )
    snap = AlpacaBrokerAdapter(client).get_leverage_snapshot()
    assert snap.maintenance_margin == pytest.approx(949.37)
    assert snap.broker_reports_leverage is False
    assert snap.leveraged is False


def test_leverage_snapshot_reports_borrowing_when_longs_exceed_equity() -> None:
    """Long market value above equity is real margin borrowing."""
    client = MagicMock()
    client.get_account.return_value = MagicMock(
        cash=-618.62,
        equity=3404.22,
        long_market_value=4022.84,
        maintenance_margin=1206.85,
        buying_power=0.0,
    )
    snap = AlpacaBrokerAdapter(client).get_leverage_snapshot()
    assert snap.broker_reports_leverage is True
    assert snap.leveraged is True


def test_leverage_snapshot_tolerates_absent_margin_fields() -> None:
    """Missing broker fields yield no borrowing claim rather than an error."""
    acct = MagicMock()
    acct.cash = 100.0
    del acct.equity
    del acct.long_market_value
    del acct.maintenance_margin
    del acct.buying_power
    client = MagicMock()
    client.get_account.return_value = acct
    snap = AlpacaBrokerAdapter(client).get_leverage_snapshot()
    assert snap.maintenance_margin is None
    assert snap.broker_reports_leverage is False
    assert snap.leveraged is False
