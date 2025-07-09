import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def irb_rw(pd, lgd, corr):
    g_pd, g_999 = norm.ppf(pd), norm.ppf(0.999)
    k = lgd * norm.cdf((1 / np.sqrt(1 - corr)) * g_pd + np.sqrt(corr / (1 - corr)) * g_999) - pd * lgd
    rw = k * 12.5
    return rw

def sa_rw(ltv):
    if ltv < 0.55:
        return 0.2
    else:
        return 0.75 - (0.55 / ltv) * (0.75 - 0.2)

def generate_impact_matrix(pd_value, lgd_range, ltv_range, output_floor):
    impact = np.zeros((len(ltv_range), len(lgd_range)))
    for i, ltv in enumerate(ltv_range):
        sa_val = sa_rw(ltv)
        for j, lgd in enumerate(lgd_range):
            irb_val = irb_rw(pd_value, lgd, corr)
            floored_irb = max(irb_val, output_floor * sa_val)
            impact[i, j] = floored_irb - sa_val
    return impact

def plot_impact_heatmap(impact_matrix, pd_value, output_floor_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    dx = (lgd_range[1] - lgd_range[0]) * 100
    dy = (ltv_range[1] - ltv_range[0]) * 100
    extent = [lgd_range[0]*100 - dx/2, lgd_range[-1]*100 + dx/2,
              ltv_range[0]*100 - dy/2, ltv_range[-1]*100 + dy/2]

    vmin = -0.4
    vmax = 0.4
    cmap = plt.get_cmap("RdYlGn_r")  # Red for positive, green for negative
    im = ax.imshow(impact_matrix, cmap=cmap, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(lgd_range * 100)
    ax.set_yticks(ltv_range * 100)
    ax.set_xlabel("LGD (%)")
    ax.set_ylabel("LTV (%)")
    ax.set_title(f"Capital Impact (IRB - SA)\nPD = {pd_value*100:.2f}%, Output Floor = {output_floor_label}", fontsize=14)
    fig.colorbar(im, ax=ax, label="Impact")

    for i in range(len(ltv_range)):
        for j in range(len(lgd_range)):
            ax.text(lgd_range[j]*100, ltv_range[i]*100, f"{impact_matrix[i,j]:.2f}", ha='center', va='center', fontsize=8)

    st.pyplot(fig)

# --- Streamlit Interface ---
st.title("IRB vs SA Capital Impact Heatmap (with Output Floor)")

corr = 0.15  # Fixed for mortgages
pd_input = st.number_input("Enter PD (%)", min_value=0.001, max_value=100.0, value=0.69, step=0.001, format="%.3f")
pd_val = pd_input / 100  # Convert to decimal

# Output floor dropdown
floor_label = st.selectbox("Select Output Floor", options=["0%", "50%", "55%", "60%", "65%", "70%", "72.5%"], index=5)
output_floor = float(floor_label.strip('%')) / 100

ltv_range = np.arange(0.0, 1.51, 0.20)
lgd_range = np.arange(0.0, 0.81, 0.10)

impact_matrix = generate_impact_matrix(pd_val, lgd_range, ltv_range, output_floor)
plot_impact_heatmap(impact_matrix, pd_val, floor_label)
