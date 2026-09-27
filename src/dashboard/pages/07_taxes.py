"""Tax Center — year-end lot summary and Form 8949 CSV (informational, not tax advice)."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.charts import COLOR_NEGATIVE, COLOR_POSITIVE, apply_hub_layout
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger
from src.portfolio.wash_sales import detect_cross_account_wash_sales, partition_lots_by_account
from src.reporting.reconciliation_1099b import parse_1099b_csv_fileobj, reconcile_1099b
from src.reporting.tax_report import (
    LossCarryforward,
    compute_loss_carryforward,
    compute_tax_summary,
    consolidated_taxable_and_advantaged,
    form_8949_csv_string,
    generate_schedule_d_summary,
    schedule_d_csv_string,
)

if not require_auth(get_settings()):
    st.stop()

st.title("Tax Center")

st.caption(
    "Informational summaries from your lot ledger. Not tax advice — verify with a qualified professional.",
)

hub_db = hub_sqlite_path()
ledger = LotLedger(hub_db)
hub_sqlite = SQLiteStore(hub_db)
closed_all = ledger.get_closed_lots(account_id=None)
open_all = ledger.get_open_lots(account_id=None)

account_ids = sorted(
    {str(getattr(x, "account_id", None) or "default") for x in closed_all}
    | {str(getattr(x, "account_id", None) or "default") for x in open_all},
)
account_options = ["All accounts", *account_ids]
account_choice = st.selectbox("Account", options=account_options, index=0)
selected_account_id: str | None = None if account_choice == "All accounts" else account_choice

years_set = {int(cl.closed_at.year) for cl in closed_all if hasattr(cl.closed_at, "year")}
now_y = datetime.now(UTC).year
if not years_set:
    year_options = [now_y]
else:
    year_options = sorted(years_set, reverse=True)
    if now_y not in year_options:
        year_options = [now_y, *year_options]

default_idx = 0
if now_y in year_options:
    default_idx = year_options.index(now_y)

selected_year = st.selectbox("Tax year", options=year_options, index=default_idx)
year_int = int(selected_year)

_settings = get_settings()
_wash_win = int(_settings.tax.wash_sale_window_days)


def _hub_account_types_local() -> dict[str, str] | None:
    acc = _settings.broker.accounts
    if not acc:
        return None
    return {str(a.name).strip(): str(a.account_type) for a in acc if str(a.name).strip()}


_ht = _hub_account_types_local()
if _ht and account_choice != "All accounts":
    _t = _ht.get(account_choice, "brokerage")
    if _t == "ira_traditional":
        st.caption(
            "Tax treatment: **Tax-Deferred** (traditional IRA — informational export only)."
        )
    elif _t == "ira_roth":
        st.caption("Tax treatment: **Tax-Free** (Roth IRA — informational export only).")

_checklist_labels = [
    "Review all closed lots for the selected year",
    "Check wash sale flags — verify disallowed amounts are reasonable",
    "Review cross-account wash sales (if multiple accounts)",
    "Download Form 8949 CSV (Part I + Part II)",
    "Download Schedule D summary",
    "Reconcile against broker 1099-B statement",
    "Verify totals match 1099-B",
    "File with tax software or accountant",
]
_ck_base = f"tax_chk_{year_int}_{account_choice}".replace(" ", "_")
with st.expander("Tax filing checklist"):
    _done = 0
    for _i, _lab in enumerate(_checklist_labels):
        if st.checkbox(_lab, key=f"{_ck_base}_{_i}"):
            _done += 1
    st.caption(f"{_done} of {len(_checklist_labels)} complete")

wash_sales = ledger.detect_wash_sales(
    account_id=selected_account_id,
    window_days=_wash_win,
)

summary = compute_tax_summary(
    ledger,
    year_int,
    wash_sales=wash_sales,
    account_id=selected_account_id,
    hub_account_types=_ht,
)

_cf_aid = (selected_account_id or "default").strip() or "default"
_prev_cf = hub_sqlite.get_loss_carryforward_row(year_int - 1, account_id=_cf_aid)
_prior_cf = None
if _prev_cf is not None:
    _prior_cf = LossCarryforward(
        year=int(_prev_cf["year"]),
        account_id=str(_prev_cf["account_id"]),
        short_term_carryforward=float(_prev_cf["short_term_carryforward"]),
        long_term_carryforward=float(_prev_cf["long_term_carryforward"]),
        computed_at=str(_prev_cf.get("computed_at") or ""),
    )
_carry = compute_loss_carryforward(summary, _prior_cf, account_id=_cf_aid)

by_c, by_o = partition_lots_by_account(closed_all, open_all)
cross_flags = [
    w
    for w in detect_cross_account_wash_sales(by_c, by_o, window_days=_wash_win)
    if int(w.sell_date.year) == year_int
]

r1 = st.columns(6)
r1[0].metric("Short-term net", f"${summary.short_term_net:,.2f}")
r1[1].metric("Long-term net", f"${summary.long_term_net:,.2f}")
r1[2].metric("Total net", f"${summary.total_net:,.2f}")
r1[3].metric("Harvested losses (TLH)", f"${summary.harvested_losses:,.2f}")
r1[4].metric("Wash-sale disallowed", f"${summary.wash_sale_disallowed:,.2f}")
r1[5].metric("Loss carryforward (est.)", f"${_carry.total_carryforward:,.2f}")

if _ht and account_choice == "All accounts":
    _taxable_part, _adv_part = consolidated_taxable_and_advantaged(summary)
    st.subheader("Taxable accounts (Schedule D scope)")
    st.caption(
        "Totals above reflect **taxable** hub accounts only when account types are configured in "
        "settings (brokerage / joint).",
    )
    if _adv_part.lots:
        st.subheader("Tax-advantaged accounts (informational)")
        st.caption("IRA lots are not included in Schedule D totals below.")
        _adf = pd.DataFrame(
            [
                {
                    "Account": x.account_id,
                    "Symbol": x.symbol,
                    "Gain/Loss": x.realized_pnl,
                    "Term": x.term,
                    "Treatment": x.tax_treatment.replace("_", " ").title(),
                }
                for x in _adv_part.lots
            ],
        )
        st.dataframe(_adf, hide_index=True, use_container_width=True)

with st.expander("Loss carryforward history (SQLite)"):
    _hist = hub_sqlite.list_loss_carryforward_rows(
        account_id=None if account_choice == "All accounts" else account_choice,
        limit=25,
    )
    if not _hist:
        st.info("No stored carryforward rows yet (populated when you record year-end snapshots).")
    else:
        st.dataframe(pd.DataFrame(_hist), hide_index=True, use_container_width=True)

if summary.wash_sale_disallowed > 0:
    ws_count = sum(1 for ws in wash_sales if int(ws.sell_date.year) == year_int)
    st.warning(
        f"Wash-sale flagging: **${summary.wash_sale_disallowed:,.2f}** disallowed loss "
        f"across **{ws_count}** sale(s) in {selected_year} (informational detector; see docs).",
    )

with st.expander("Cross-account wash sale flags (informational)"):
    st.caption(
        "IRS wash rules can span accounts for the same taxpayer. "
        "These matches are for review only — they do not adjust cost basis.",
    )
    if len(account_ids) < 2:
        st.info(
            "At least two hub accounts with lot data are required for cross-account detection."
        )
    elif not cross_flags:
        st.info(f"No cross-account patterns flagged for {selected_year}.")
    else:
        cdf = pd.DataFrame(
            [
                {
                    "Loss account": w.selling_account,
                    "Buy account": w.buying_account,
                    "Symbol": w.symbol,
                    "Loss $": w.loss_amount,
                    "Sale date": w.sell_date.isoformat(),
                    "Replacement opened": w.replacement_date.isoformat(),
                }
                for w in cross_flags
            ],
        )
        st.dataframe(cdf, hide_index=True, use_container_width=True)

fig = go.Figure(
    data=[
        go.Bar(
            x=[summary.short_term_net, summary.long_term_net],
            y=["Short-term net", "Long-term net"],
            orientation="h",
            marker_color=[
                COLOR_POSITIVE if summary.short_term_net >= 0 else COLOR_NEGATIVE,
                COLOR_POSITIVE if summary.long_term_net >= 0 else COLOR_NEGATIVE,
            ],
        ),
    ],
)
fig.update_layout(showlegend=False)
apply_hub_layout(fig, title="Net realized P&L by term", height=280)
st.plotly_chart(fig, use_container_width=True)

sd = generate_schedule_d_summary(
    summary,
    wash_sales=wash_sales,
    account_id=selected_account_id,
)
st.subheader("Schedule D summary (informational)")
_sdc = st.columns(4)
_sdc[0].metric("ST proceeds", f"${sd.short_term_proceeds:,.2f}")
_sdc[1].metric("ST cost basis", f"${sd.short_term_cost_basis:,.2f}")
_sdc[2].metric("ST wash adj.", f"${sd.short_term_wash_adjustments:,.2f}")
_sdc[3].metric("ST net (8949 rows)", f"${sd.short_term_net:,.2f}")
_sdc2 = st.columns(4)
_sdc2[0].metric("LT proceeds", f"${sd.long_term_proceeds:,.2f}")
_sdc2[1].metric("LT cost basis", f"${sd.long_term_cost_basis:,.2f}")
_sdc2[2].metric("LT wash adj.", f"${sd.long_term_wash_adjustments:,.2f}")
_sdc2[3].metric("LT net (8949 rows)", f"${sd.long_term_net:,.2f}")
st.caption(f"Total net gain or loss (Schedule D style): **${sd.total_net_gain_or_loss:,.2f}**")
_sd_fn = f"schedule_d_{selected_year}"
if selected_account_id is not None:
    _safe_sd = "".join(c if c.isalnum() or c in "-_" else "_" for c in selected_account_id)[:64]
    _sd_fn += f"_{_safe_sd}"
_sd_fn += ".csv"
st.download_button(
    label="Download Schedule D CSV",
    data=schedule_d_csv_string(sd),
    file_name=_sd_fn,
    mime="text/csv",
)

if not summary.lots:
    st.info(f"No closed lots with sale dates in {selected_year} for this scope.")
else:
    rows = []
    for lot in summary.lots:
        row = {
            "Account": lot.account_id,
            "Symbol": lot.symbol,
            "Qty": lot.qty,
            "Cost basis": lot.cost_basis,
            "Proceeds": lot.proceeds,
            "Gain/Loss": lot.realized_pnl,
            "Term": lot.term,
            "Holding days": lot.holding_days,
            "TLH": "Yes" if lot.is_tlh else "",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    scope = account_choice
    st.subheader(f"Closed lots ({selected_year}) — {scope}")
    st.dataframe(
        df,
        column_config={
            "Cost basis": st.column_config.NumberColumn(format="$%.2f"),
            "Proceeds": st.column_config.NumberColumn(format="$%.2f"),
            "Gain/Loss": st.column_config.NumberColumn(format="$%.2f"),
            "Qty": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Reconcile 1099-B (broker CSV)"):
    st.caption(
        "Upload a UTF-8 CSV with columns including Symbol, Qty, Date Acquired, Date Sold, "
        "Proceeds, and Cost Basis (Alpaca-style exports). Compared to closed lots for the "
        "selected tax year and account scope.",
    )
    up = st.file_uploader("1099-B CSV", type=["csv"], key="fh_tax_1099b")
    if up is not None:
        try:
            raw_txt = up.read().decode("utf-8-sig")
            br_rows = parse_1099b_csv_fileobj(StringIO(raw_txt))
        except ValueError as exc:
            st.error(str(exc))
        else:
            _cy = [
                cl
                for cl in ledger.get_closed_lots(account_id=selected_account_id)
                if int(cl.closed_at.year) == year_int
            ]
            rep = reconcile_1099b(br_rows, _cy)
            st.metric("Broker rows", rep.total_broker_records)
            st.metric("Internal lots (year)", rep.total_internal_lots)
            st.metric("Matched", rep.matched)
            st.metric("Discrepancies", rep.discrepancies)
            _mr = [
                {
                    "Status": m.status,
                    "Broker": m.broker_record.symbol if m.broker_record else "—",
                    "Broker qty": m.broker_record.qty if m.broker_record else None,
                    "Internal": m.internal_lot.symbol if m.internal_lot else "—",
                    "Variance $": round(m.variance, 2),
                }
                for m in rep.matches
            ]
            st.dataframe(pd.DataFrame(_mr), hide_index=True, use_container_width=True)

csv_label = None if selected_account_id is None else selected_account_id
csv_text = form_8949_csv_string(summary, wash_sales=wash_sales, account_label=csv_label)
fn = f"form_8949_{selected_year}"
if selected_account_id is not None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in selected_account_id)[:64]
    fn += f"_{safe}"
fn += ".csv"
st.download_button(
    label="Download Form 8949 CSV",
    data=csv_text,
    file_name=fn,
    mime="text/csv",
    type="primary",
)
