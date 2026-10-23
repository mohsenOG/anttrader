"""Local, read-only AntTrader research dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from anttrader.dashboard import (
    DISCLAIMER,
    RECOVERY,
    STRATEGY_NAMES,
    comparison_table,
    format_metric,
    grid_chart,
    input_signature,
    load_research,
    price_chart,
    replay_state,
    selected_simulation,
)
from anttrader.features import checkpoint_features
from anttrader.pipeline import DEFAULT_CONFIG


@st.cache_data(show_spinner=False, max_entries=2)
def cached_research(config_path: str, signature: tuple):
    return load_research(Path(config_path))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_simulation(_research, decision: str, signature: tuple):
    return selected_simulation(_research, decision)


def utc(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def jump_to_checkpoint():
    st.session_state["replay_step"] = st.session_state["checkpoint_step"]


def main(config_path: Path = DEFAULT_CONFIG) -> None:
    st.set_page_config(page_title="AntTrader", layout="wide")
    st.title("AntTrader")
    st.write("Langton's Ant Signal Research — XAU/USD 1H")
    st.warning(DISCLAIMER)
    status = st.empty()
    tabs = st.tabs(["Ant Simulation", "Gold & Signals"])
    try:
        signature = input_signature(config_path)
        with st.spinner("Validating the saved research experiment..."):
            research = cached_research(str(config_path.resolve()), signature)
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        st.error(str(exc))
        st.info(RECOVERY)
        st.stop()
    metadata = research.metadata
    status.caption(
        f"Historical experiment loaded · {len(research.candles):,} candles · "
        f"{len(research.features):,} feature windows · "
        f"{len(research.signals):,} saved predictions · UTC · Read-only"
    )
    with st.expander("Experiment metadata and research configuration"):
        st.write(
            f"Candle openings: {utc(research.candles.timestamp.iloc[0])} to "
            f"{utc(research.candles.timestamp.iloc[-1])}. "
            f"Feature source cutoff: {utc(metadata['feature_source_cutoff'])}."
        )
        st.caption(f"Source SHA-256: {metadata['source_sha256']}")
        st.caption(f"Configuration SHA-256: {metadata['config_sha256']}")
        st.dataframe(pd.DataFrame(metadata["partitions"]).T, width="stretch")
        st.json(
            {
                key: value
                for key, value in metadata.items()
                if key.startswith("excluded_")
            }
        )
        st.json(research.config)
        st.caption(
            "Frozen training-only models. Probabilities are uncalibrated estimates. "
            "The four-hour horizon defines forecast labels only. "
            "One historical split cannot establish a reliable market edge."
        )

    with tabs[0]:
        st.subheader("Explore a historical market grid")
        decision = st.selectbox(
            "Decision timestamp (UTC)",
            research.features.decision_timestamp,
            index=len(research.features) - 1,
            format_func=utc,
        )
        try:
            window, result = cached_simulation(
                research, decision.isoformat(), signature
            )
        except (ValueError, TypeError, IndexError) as exc:
            st.error(f"Cannot replay selected window: {exc}")
        else:
            st.caption(
                f"Decision: {utc(decision)} · 257 completed candle openings: "
                f"{utc(window.window_start)} to {utc(window.window_end)}"
            )
            st.caption(
                f"16 × 16 binary grid · One ant · Start "
                f"({result.config.start_row}, {result.config.start_col}) "
                f"{result.config.initial_direction.title()} · Toroidal · "
                f"{result.config.steps:,} steps · Checkpoint features match saved results"
            )
            st.selectbox(
                "Jump to checkpoint",
                [0, *result.config.checkpoints],
                key="checkpoint_step",
                on_change=jump_to_checkpoint,
            )
            step = st.slider(
                "Replay step",
                0,
                result.config.steps,
                key="replay_step",
            )
            state = replay_state(result, step)
            left, right = st.columns(2)
            left.plotly_chart(
                grid_chart(result.initial_grid, None, "Original market grid"),
                width="stretch",
                key="initial_grid",
            )
            right.plotly_chart(
                grid_chart(state.grid, state, f"Replay · Step {step:,}"),
                width="stretch",
                key="replay_grid",
            )
            st.caption(
                "Light = 0 · Dark = 1 · Gold arrow = ant heading. "
                "Rows increase downward. Visits count reads before moves."
            )
            features = checkpoint_features(state) if step else {}
            values = {
                "Step": f"{step:,}",
                "Wrapped row / column": f"{state.row} / {state.col}",
                "Heading": state.heading.title(),
                "Unique cells visited": str(state.unique_cells_visited),
                "Revisits": str(state.revisits),
                "Right-turn fraction": format_metric(
                    features.get("right_turn_fraction"), percent=True
                ),
                "Black-cell fraction": format_metric(
                    state.black_cells / 256, percent=True
                ),
                "Changed-cell fraction": format_metric(
                    state.changed_cells / 256, percent=True
                ),
                "Visit entropy": format_metric(features.get("visit_entropy")),
                "Unwrapped displacement (x east, y south)": (
                    f"{state.horizontal_displacement}, {state.vertical_displacement}"
                ),
            }
            st.dataframe(
                pd.DataFrame(values.items(), columns=["Replay statistic", "Value"]),
                hide_index=True,
                width="stretch",
            )
            if step == 0:
                st.caption(
                    "Right-turn fraction and visit entropy are undefined before the first read."
                )
            saved = research.signals.loc[
                research.signals.decision_timestamp.eq(decision)
            ]
            if saved.empty:
                st.info("No saved prediction exists for this decision timestamp.")
            else:
                row = saved.iloc[0]
                st.info(
                    f"Saved AntTrader signal: {row.ant_signal} · "
                    f"P(up): {row.ant_p_up:.4f} · {row.partition.title()} · {utc(decision)}"
                )

    with tabs[1]:
        st.subheader("Latest Historical Signal")
        latest = research.signals.iloc[-1]
        st.write(
            f"**{latest.ant_signal}** · AntTrader P(up): **{latest.ant_p_up:.4f}** · "
            f"{utc(latest.decision_timestamp)} · {latest.partition.title()}"
        )
        st.caption(
            "Latest prediction across the saved experiment. Historical research only; "
            "no live market feed or current market recommendation."
        )
        one, two = st.columns(2)
        partition = one.selectbox(
            "Price chart partition", ["Test", "Validation"], key="price_partition"
        ).lower()
        strategy = two.selectbox(
            "Displayed strategy", list(STRATEGY_NAMES), format_func=STRATEGY_NAMES.get
        )
        signals = research.signals.loc[research.signals.partition.eq(partition)]
        first = signals.decision_timestamp.iloc[0].date()
        last = signals.decision_timestamp.iloc[-1].date()
        default_start = max(first, last - pd.Timedelta(days=10))
        sides = st.multiselect(
            "Signal markers",
            ["BUY", "SELL"],
            default=["BUY", "SELL"],
            key="visible_sides",
        )
        dates = st.date_input(
            "Chart date range (UTC)",
            value=(default_start, last),
            min_value=first,
            max_value=last,
            key=f"chart_dates_{partition}",
        )
        if len(dates) != 2:
            st.info("Select both the start and end date.")
        else:
            start = pd.Timestamp(dates[0], tz="UTC")
            stop = pd.Timestamp(dates[1], tz="UTC") + pd.Timedelta(days=1)
            candles = research.candles.loc[
                research.candles.timestamp.ge(start)
                & research.candles.timestamp.lt(stop)
            ]
            visible_signals = signals.loc[
                signals.decision_timestamp.ge(start)
                & signals.decision_timestamp.lt(stop)
            ]
            st.caption(
                f"Active partition: {partition.title()} · {STRATEGY_NAMES[strategy]} · "
                "Green triangles: BUY · Red triangles: SELL · HOLD remains in the table. "
                "Markers use the preceding completed candle with a visual offset, not an execution price."
            )
            if candles.empty:
                st.info("No historical candles exist in this date range.")
            else:
                st.plotly_chart(
                    price_chart(
                        research.candles,
                        visible_signals,
                        strategy,
                        partition,
                        visible_sides=tuple(sides),
                        start=start,
                        stop=stop,
                    ),
                    width="stretch",
                    key="gold_chart",
                    config={"scrollZoom": True, "displaylogo": False},
                )
            st.subheader("Signal history")
            st.caption(
                "All strategies, including HOLD. Click a column to sort. "
                "target and future_return are retrospective outcomes, never model inputs. "
                "target = 1 means a positive future return; target = 0 means zero or negative."
            )
            st.dataframe(
                visible_signals[
                    [
                        "decision_timestamp",
                        "ant_signal",
                        "ant_p_up",
                        "reference_price",
                        "partition",
                        "momentum_signal",
                        "raw_signal",
                        "raw_p_up",
                        "target",
                        "label_end_timestamp",
                        "future_return",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        st.subheader("Prediction statistics")
        st.caption(
            f"{partition.title()} · Full saved partition; independent of the chart date range and marker filters."
        )
        st.dataframe(comparison_table(research.metrics, partition), width="stretch")
        with st.expander("Metric definitions and undefined values"):
            st.write(
                "Overall classification accuracy uses P(up) >= 0.5 across all eligible "
                "model predictions, including HOLD. BUY/SELL directional accuracy excludes HOLD. "
                "BUY predicts a positive four-hour label; SELL predicts a zero or negative label. "
                "Momentum accuracy uses non-HOLD observations only. "
                "Statistics evaluate retrospective labels, not trading performance."
            )
            for strategy, name in STRATEGY_NAMES.items():
                metrics = research.metrics["partitions"][partition][strategy][
                    "predictive"
                ]
                st.write(f"**{name}**: {metrics['accuracy_scope']}.")
                for note in (
                    metrics["roc_auc_note"],
                    metrics["actionable_accuracy_note"],
                ):
                    if note:
                        st.write(note)


if __name__ == "__main__":
    main()
