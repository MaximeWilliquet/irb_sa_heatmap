# rwa_streamlit_app.py
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
        rw = 0.75-(0.55/ltv)*(0.75-0.2)
    return rw

def generate_impact_matrix(pd_value, lgd_range, ltv_range):
    impact = np.zeros((len(ltv_range), len(lgd_range)))
    for i, ltv in enumerate(ltv_range):
        sa_val = sa_rw(ltv)
        for j, lgd in enumerate(lgd_range):
            irb_val = irb_rw(pd_value, lgd, corr)
            impact[i, j] = irb_val - sa_val
    return impact

def plot_impact_heatmap(impact_matrix, pd_value):
    fig, ax = plt.subplots(figsize=(8, 6))
    dx = (lgd_range[1] - lgd_range[0]) * 100
    dy = (ltv_range[1] - ltv_range[0]) * 100
    extent = [lgd_range[0]*100 - dx/2, lgd_range[-1]*100 + dx/2,
              ltv_range[0]*100 - dy/2, ltv_range[-1]*100 + dy/2]

    im = ax.imshow(impact_matrix, cmap='RdYlGn_r', origin='lower', extent=extent, aspect='auto')
    ax.set_xticks(lgd_range * 100)
    ax.set_yticks(ltv_range * 100)
    ax.set_xlabel("LGD (%)")
    ax.set_ylabel("LTV (%)")
    ax.set_title(f"Capital Impact (IRB - SA)\nPD = {pd_value*100:.2f}%", fontsize=14)
    fig.colorbar(im, ax=ax, label="Impact")

    for i in range(len(ltv_range)):
        for j in range(len(lgd_range)):
            ax.text(lgd_range[j]*100, ltv_range[i]*100, f"{impact_matrix[i,j]:.2f}", ha='center', va='center', fontsize=8)

    st.pyplot(fig)

# Streamlit interface
st.title("IRB vs SA Capital Impact Heatmap")

corr = 0.15 # Correlation is fixed for mortgage portfolios
pd_val = st.slider("Select PD (%)", min_value=0.01, max_value=3.0, value=0.69, step=0.01) / 100
ltv_range = np.arange(0.0, 1.51, 0.20)
lgd_range = np.arange(0.0, 0.81, 0.10)
impact_matrix = generate_impact_matrix(pd_val, lgd_range, ltv_range)
plot_impact_heatmap(impact_matrix, pd_val)