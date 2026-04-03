import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates
import math

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Progressive Carries Dashboard")
st.title("Progressive Carries Dashboard")
st.caption("Click on the start dot of a carry to view the corresponding video (if available).")

# ==========================
# Configuration
# ==========================
FINAL_THIRD_LINE_X = 80

# ==========================
# Data
# Each carry: (x_start, y_start, x_end, y_end, video_path or None)
# ==========================
coords_by_match = {
    "Vs Los Angeles": [
        (75.96, 2.26, 111.03, 20.88, None),
        (53.02, 73.41, 99.90, 76.07, None),
    ],
    "Vs Slavia Praha": [
        (97.57, 3.93, 115.36, 19.22, None),
        (92.91, 10.41, 105.05, 22.88, None),
        (98.23, 26.37, 116.69, 24.54, None),
    ],
    "Vs Sockers": [
        (62.99, 70.42, 112.36, 69.09, None),
    ],
}

MATCHES = list(coords_by_match.keys()) + ["All Matches"]


# ==========================
# Helpers
# ==========================
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def build_df(carries: list) -> pd.DataFrame:
    rows = []
    for i, carry in enumerate(carries):
        x_start, y_start, x_end, y_end, video = carry
        dist = calculate_distance(x_start, y_start, x_end, y_end)
        rows.append(
            {
                "numero": i + 1,
                "x_start": float(x_start),
                "y_start": float(y_start),
                "x_end": float(x_end),
                "y_end": float(y_end),
                "distancia": dist,
                "video": video,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df["in_final_third"] = df["x_end"] >= FINAL_THIRD_LINE_X
        df["to_box"] = df["x_end"] >= 100
    else:
        df = pd.DataFrame(
            columns=[
                "numero", "x_start", "y_start", "x_end", "y_end",
                "distancia", "video", "in_final_third", "to_box",
            ]
        )

    return df


def compute_stats(df: pd.DataFrame) -> dict:
    total_carries = len(df)
    total_distance = round(df["distancia"].sum(), 1) if not df.empty else 0.0
    final_third_total = int(df["in_final_third"].sum()) if not df.empty else 0
    box_total = int(df["to_box"].sum()) if not df.empty else 0

    return {
        "total_carries": total_carries,
        "total_distance": total_distance,
        "final_third_total": final_third_total,
        "box_total": box_total,
    }


def draw_carry_map(df: pd.DataFrame, title: str):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(7.9, 5.3))
    fig.set_dpi(110)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    purple_color = (0.5, 0.0, 0.5, 0.75)

    for _, row in df.iterrows():
        has_vid = has_video_value(row["video"])

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=purple_color,
            width=1.55,
            headwidth=2.25,
            headlength=2.25,
            ax=ax,
            zorder=3,
        )

        # Golden ring if video is available
        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"],
                s=95,
                marker="o",
                facecolors="none",
                edgecolors="#FFD54F",
                linewidths=2.0,
                ax=ax,
                zorder=4,
            )

        # Start dot
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=55,
            marker="o",
            color=purple_color,
            edgecolors="white",
            linewidths=0.8,
            ax=ax,
            zorder=5,
        )

    ax.set_title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=purple_color, lw=2.5, label="Progressive Carry"),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=(0.5, 0.0, 0.5, 0.75),
            markeredgecolor="white",
            markersize=6,
            label="Start point (click)",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=(0.5, 0.0, 0.5, 0.75),
            markeredgecolor="#FFD54F",
            markeredgewidth=2,
            markersize=7,
            label="Has video",
        ),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        shadow=False,
        fontsize="x-small",
        labelspacing=0.5,
        borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center",
             fontsize=9, color="#333333")

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig


# ==========================
# Sidebar
# ==========================
st.sidebar.header("Match Selection")
selected_match = st.sidebar.radio("Choose a match", MATCHES, index=0)

# ==========================
# Build selected DataFrame
# ==========================
if selected_match == "All Matches":
    all_carries = []
    for match in MATCHES[:-1]:
        all_carries.extend(coords_by_match[match])
    carries = all_carries
else:
    carries = coords_by_match[selected_match]

df = build_df(carries)
stats = compute_stats(df)

# ==========================
# Dashboard Layout
# ==========================
col_stats, col_map = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Statistics")

    c1, c2 = st.columns(2)
    c1.metric("Total Carries", stats["total_carries"])
    c2.metric("Total Distance", f"{stats['total_distance']}m")

    st.divider()

    st.subheader("Progression")
    c3, c4 = st.columns(2)
    c3.metric("Into Final Third", stats["final_third_total"])
    c4.metric("Into the Box", stats["box_total"])

    if stats["total_carries"] > 0:
        avg = round(stats["total_distance"] / stats["total_carries"], 1)
        st.info(f"Average of {avg} metres per carry.")

with col_map:
    st.subheader("Carry Map (click on a start dot)")

    img_obj, ax, fig = draw_carry_map(df, title=f"Progressive Carries - {selected_match}")
    click = streamlit_image_coordinates(img_obj, width=780)

    selected_carry = None

    if click is not None:
        real_w, real_h = img_obj.size
        disp_w, disp_h = click["width"], click["height"]

        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)

        mpl_pixel_y = real_h - pixel_y
        coords_clicked = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
        field_x, field_y = coords_clicked[0], coords_clicked[1]

        df_sel = df.copy()
        df_sel["dist"] = np.sqrt(
            (df_sel["x_start"] - field_x) ** 2 +
            (df_sel["y_start"] - field_y) ** 2
        )

        RADIUS = 7.0
        candidates = df_sel[df_sel["dist"] < RADIUS]

        if not candidates.empty:
            selected_carry = candidates.loc[candidates["dist"].idxmin()]

    plt.close(fig)

    st.divider()
    st.subheader("Video")

    if selected_carry is None:
        st.info("Click on a carry's start dot to view the video (if available).")
    else:
        st.success(
            f"Selected carry #{int(selected_carry['numero'])}"
        )
        st.write(
            f"Start: ({selected_carry['x_start']:.2f}, {selected_carry['y_start']:.2f})  \n"
            f"End: ({selected_carry['x_end']:.2f}, {selected_carry['y_end']:.2f})  \n"
            f"Distance: {selected_carry['distancia']:.1f} m"
        )

        if has_video_value(selected_carry["video"]):
            try:
                st.video(selected_carry["video"])
            except Exception:
                st.error(f"Video file not found: {selected_carry['video']}")
        else:
            st.warning("No video available for this carry.")
