import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main():
    # --- Data (simplified names) ---
    data = {
        "CRPS": {
            "1-week": {
                "IFS det.": 0.568996,
                "IFS ens.": 0.467610,
                "Diff. det.": 0.537628,
                "Diff. ens.": 0.464930,
                "WRF": 0.556656,
            },
            "2-week": {
                "IFS det.": 0.598525,
                "IFS ens.": 0.479447,
                "Diff. det.": 0.602082,
                "Diff. ens.": 0.482350,
                "WRF": 0.565567,
            },
            "3-week": {
                "IFS det.": 0.610963,
                "IFS ens.": 0.471282,
                "Diff. det.": 0.566846,
                "Diff. ens.": 0.473097,
                "WRF": 0.543068,
            },
        },
        "W1": {
            "1-week": {
                "IFS det.": 0.467057,
                "IFS ens.": 0.389072,
                "Diff. det.": 0.435351,
                "Diff. ens.": 0.362243,
                "WRF": 0.330953,
            },
            "2-week": {
                "IFS det.": 0.330483,
                "IFS ens.": 0.387931,
                "Diff. det.": 0.276688,
                "Diff. ens.": 0.368501,
                "WRF": 0.468432,
            },
            "3-week": {
                "IFS det.": 0.309317,
                "IFS ens.": 0.386233,
                "Diff. det.": 0.259699,
                "Diff. ens.": 0.372740,
                "WRF": 0.448403,
            },
        },
        "PSD": {
            "1-week": {
                "IFS det.": 1.2577e-05,
                "IFS ens.": 1.1650e-05,
                "Diff. det.": 9.0887e-06,
                "Diff. ens.": 9.4543e-06,
                "WRF": 6.4962e-06,
            },
            "2-week": {
                "IFS det.": 1.1278e-05,
                "IFS ens.": 1.2253e-05,
                "Diff. det.": 1.0168e-05,
                "Diff. ens.": 1.0060e-05,
                "WRF": 7.5337e-06,
            },
            "3-week": {
                "IFS det.": 1.3223e-05,
                "IFS ens.": 1.2978e-05,
                "Diff. det.": 9.5474e-06,
                "Diff. ens.": 1.0857e-05,
                "WRF": 9.6346e-06,
            },
        },
    }

    lead_times = ["1-week", "2-week", "3-week"]
    metrics = list(data.keys())
    models = list(data["CRPS"]["1-week"].keys())

    # --- Convert to skill scores ---
    skill_data = {}
    for metric, leads in data.items():
        all_values = np.array([v for lead in leads.values() for v in lead.values()])
        best = all_values.min()  # lower is better
        worst = all_values.max()
        skill_data[metric] = {
            lt: {m: (worst - leads[lt][m]) / (worst - best) for m in leads[lt]}
            for lt in leads
        }

    # --- Radar setup ---
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # --- Color mapping ---
    color_map = {"det": "C0", "ens": "C1", "WRF": "C2"}

    # --- Subplots ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), subplot_kw=dict(polar=True))

    for i, lead_time in enumerate(lead_times):
        ax = axs[i]

        for model in models:
            values = [skill_data[m][lead_time][model] for m in metrics]
            values += values[:1]

            if "ifs det" in model.lower():
                c = color_map["det"]
                ax.plot(angles, values, color=c, linewidth=2, linestyle="--")
                ax.fill(
                    angles,
                    values,
                    facecolor="none",
                    hatch="//",
                    linestyle="--",
                    edgecolor=c,
                    linewidth=1.5,
                    zorder=2,
                )
            elif "ifs ens" in model.lower():
                c = color_map["ens"]
                ax.plot(angles, values, color=c, linewidth=2, linestyle="--")
                ax.fill(
                    angles,
                    values,
                    facecolor="none",
                    hatch="\\",
                    linestyle="--",
                    edgecolor=c,
                    linewidth=1.5,
                    zorder=2,
                )
            else:
                # Diff. det., Diff. ens., WRF: filled with alpha
                if "wrf" in model.lower():
                    c = color_map["WRF"]
                elif "diff. det." in model.lower():
                    c = color_map["det"]
                else:
                    c = color_map["ens"]
                ax.plot(angles, values, color=c, linewidth=2)
                ax.fill(
                    angles,
                    values,
                    facecolor=c,
                    alpha=0.3,
                    edgecolor=c,
                    linewidth=1.5,
                    zorder=1,
                )

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        ax.set_title(lead_time, pad=10)
        ax.grid(alpha=0.3)

    # Save plot
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/analysis/radar")
    plt.savefig(os.path.join(figs_dir, "radar.png"), dpi=300)
    plt.close()

    # --- Create manual legend ---
    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor=color_map["det"],
            hatch="//",
            label="IFS det.",
            linestyle="--",
        ),
        Patch(
            facecolor="none",
            edgecolor=color_map["ens"],
            hatch="\\",
            label="IFS ens.",
            linestyle="--",
        ),
        Patch(
            facecolor=color_map["det"],
            alpha=0.5,
            edgecolor=color_map["det"],
            linewidth=1.5,
            label="Diff. det.",
        ),
        Patch(
            facecolor=color_map["ens"],
            linewidth=1.5,
            edgecolor=color_map["ens"],
            alpha=0.5,
            label="Diff. ens.",
        ),
        Patch(
            facecolor=color_map["WRF"],
            linewidth=1.5,
            edgecolor=color_map["WRF"],
            alpha=0.5,
            label="WRF",
        ),
    ]

    # --- Save legend only ---
    fig_legend = plt.figure(figsize=(2, 3))
    fig_legend.legend(handles=legend_handles, loc="center")
    fig_legend.tight_layout()
    fig_legend.savefig(os.path.join(figs_dir, "legend.png"), dpi=300)
    plt.close(fig_legend)


if __name__ == "__main__":
    main()
