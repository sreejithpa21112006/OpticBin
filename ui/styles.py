"""
OpticBin — Premium Dark Theme Styling
=========================================
Glassmorphism-inspired dark UI with vibrant 3-bin color coding,
smooth animations, and modern visual hierarchy.
"""

import streamlit as st

IMAGE_MODE = "Image File Upload"
WEBCAM_MODE = "Camera Viewfinder"


def apply_styles() -> None:
    """Inject premium dark-mode CSS with glassmorphism, gradients, and micro-animations."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

            /* ───── Global Theme ───── */
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1320px;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            /* ───── Header ───── */
            .ob-header {
                text-align: center;
                padding: 2rem 1rem 1.5rem;
                margin-bottom: 1.5rem;
                background: linear-gradient(135deg, rgba(34,197,94,0.06) 0%, rgba(59,130,246,0.06) 50%, rgba(249,115,22,0.06) 100%);
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.06);
                position: relative;
                overflow: hidden;
            }

            .ob-header::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: radial-gradient(ellipse at 30% 0%, rgba(34,197,94,0.08) 0%, transparent 50%),
                            radial-gradient(ellipse at 70% 100%, rgba(59,130,246,0.08) 0%, transparent 50%);
                pointer-events: none;
            }

            .ob-logo-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
            }

            .ob-logo-icon {
                font-size: 2.4rem;
                line-height: 1;
            }

            .ob-title {
                font-size: 2.8rem;
                font-weight: 900;
                letter-spacing: -0.03em;
                background: linear-gradient(135deg, #22C55E 0%, #3B82F6 50%, #F97316 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0;
                line-height: 1.1;
            }

            .ob-subtitle {
                font-size: 1.05rem;
                font-weight: 400;
                opacity: 0.65;
                margin: 0.4rem 0 0;
                letter-spacing: 0.01em;
            }

            .ob-bin-legend {
                display: flex;
                justify-content: center;
                gap: 1.5rem;
                margin-top: 1rem;
                flex-wrap: wrap;
            }

            .ob-legend-item {
                display: flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.82rem;
                font-weight: 600;
                opacity: 0.8;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
            }

            /* ───── Glassmorphism Hero Card ───── */
            .ob-hero {
                border-radius: 20px;
                padding: 1.8rem 2rem;
                margin-bottom: 1.2rem;
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                animation: heroSlideIn 0.4s ease-out;
            }

            @keyframes heroSlideIn {
                from { opacity: 0; transform: translateY(12px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            .ob-hero-bio {
                background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(74,222,128,0.06) 100%);
                border: 1.5px solid rgba(34,197,94,0.35);
                box-shadow: 0 0 40px rgba(34,197,94,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
            }

            .ob-hero-nonbio {
                background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(96,165,250,0.06) 100%);
                border: 1.5px solid rgba(59,130,246,0.35);
                box-shadow: 0 0 40px rgba(59,130,246,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
            }

            .ob-hero-ewaste {
                background: linear-gradient(135deg, rgba(249,115,22,0.12) 0%, rgba(251,146,60,0.06) 100%);
                border: 1.5px solid rgba(249,115,22,0.35);
                box-shadow: 0 0 40px rgba(249,115,22,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
            }

            .ob-hero-kicker {
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                opacity: 0.7;
                margin-bottom: 0.5rem;
            }

            .ob-hero-bin-row {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
            }

            .ob-hero-bin-icon {
                font-size: 2.2rem;
                line-height: 1;
                filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
            }

            .ob-hero-bin-name {
                font-size: 1.6rem;
                font-weight: 800;
                letter-spacing: -0.01em;
                line-height: 1.2;
            }

            .ob-hero-disposal {
                font-size: 0.95rem;
                font-weight: 500;
                opacity: 0.85;
                margin: 0.3rem 0 0.8rem;
            }

            .ob-pills-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
            }

            .ob-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.3rem 0.75rem;
                border-radius: 20px;
                font-size: 0.78rem;
                font-weight: 600;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                backdrop-filter: blur(4px);
                transition: all 0.2s ease;
            }

            .ob-pill:hover {
                background: rgba(255,255,255,0.1);
                transform: translateY(-1px);
            }

            .ob-pill-verified {
                border-color: rgba(34,197,94,0.5) !important;
                color: #4ADE80 !important;
            }

            /* ───── Info Cards ───── */
            .ob-card {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 1.2rem 1.4rem;
                margin-bottom: 1rem;
                backdrop-filter: blur(8px);
                transition: all 0.25s ease;
                animation: cardFadeIn 0.5s ease-out;
            }

            @keyframes cardFadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            .ob-card:hover {
                border-color: rgba(255,255,255,0.15);
                background: rgba(255,255,255,0.05);
            }

            .ob-card-title {
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.6;
                margin-bottom: 0.8rem;
            }

            /* ───── Checklist ───── */
            .ob-checklist-item {
                display: flex;
                align-items: flex-start;
                margin: 0.55rem 0;
                font-size: 0.9rem;
                line-height: 1.5;
                padding: 0.4rem 0.6rem;
                border-radius: 8px;
                transition: background 0.2s ease;
            }

            .ob-checklist-item:hover {
                background: rgba(255,255,255,0.03);
            }

            .ob-check-icon {
                margin-right: 0.6rem;
                font-size: 0.9rem;
                flex-shrink: 0;
                margin-top: 0.15rem;
            }

            .ob-check-icon-bio { color: #22C55E; }
            .ob-check-icon-nonbio { color: #3B82F6; }
            .ob-check-icon-ewaste { color: #F97316; }

            /* ───── Examples Strip ───── */
            .ob-examples {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin-top: 0.5rem;
            }

            .ob-example-tag {
                padding: 0.2rem 0.6rem;
                border-radius: 12px;
                font-size: 0.72rem;
                font-weight: 600;
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                opacity: 0.75;
            }

            /* ───── Empty State ───── */
            .ob-empty-state {
                border: 2px dashed rgba(255,255,255,0.12);
                border-radius: 20px;
                padding: 3.5rem 2rem;
                text-align: center;
                background: linear-gradient(135deg, rgba(34,197,94,0.02) 0%, rgba(59,130,246,0.02) 50%, rgba(249,115,22,0.02) 100%);
                animation: emptyPulse 3s ease-in-out infinite;
            }

            @keyframes emptyPulse {
                0%, 100% { border-color: rgba(255,255,255,0.08); }
                50%      { border-color: rgba(255,255,255,0.18); }
            }

            .ob-empty-state h3 {
                margin: 0 0 0.5rem;
                font-size: 1.3rem;
                font-weight: 700;
            }

            .ob-empty-state p {
                font-size: 0.95rem;
                opacity: 0.55;
                margin: 0;
                max-width: 500px;
                margin: 0 auto;
                line-height: 1.6;
            }

            /* ───── Active Learning Badge ───── */
            .ob-al-badge {
                border-radius: 12px;
                padding: 0.7rem 1rem;
                margin-bottom: 1rem;
                font-size: 0.85rem;
                line-height: 1.5;
                backdrop-filter: blur(8px);
                animation: badgeSlide 0.3s ease-out;
            }

            @keyframes badgeSlide {
                from { opacity: 0; transform: translateX(-8px); }
                to   { opacity: 1; transform: translateX(0); }
            }

            .ob-al-verified {
                background: rgba(34,197,94,0.08);
                border: 1px solid rgba(34,197,94,0.25);
                border-left: 4px solid #22C55E;
            }

            .ob-al-corrected {
                background: rgba(249,115,22,0.08);
                border: 1px solid rgba(249,115,22,0.25);
                border-left: 4px solid #F97316;
            }

            /* ───── Section Headers ───── */
            .ob-section-title {
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            /* ───── Probability Bars ───── */
            .stProgress > div > div > div > div {
                border-radius: 8px !important;
            }

            /* ───── Sidebar ───── */
            [data-testid="stSidebar"] {
                background: rgba(0,0,0,0.15);
                border-right: 1px solid rgba(255,255,255,0.06);
            }

            [data-testid="stSidebar"] .stRadio > label {
                font-weight: 600;
            }

            /* ───── Hide default footer ───── */
            footer { visibility: hidden; }

            /* ───── Impact Stat ───── */
            .ob-impact-stat {
                text-align: center;
                padding: 0.8rem;
                border-radius: 12px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
            }

            .ob-impact-value {
                font-size: 1.5rem;
                font-weight: 800;
                line-height: 1.2;
            }

            .ob-impact-label {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                opacity: 0.5;
                margin-top: 0.2rem;
            }

            /* ───── Footer ───── */
            .ob-footer {
                text-align: center;
                padding: 1rem;
                margin-top: 2rem;
                opacity: 0.4;
                font-size: 0.8rem;
                border-top: 1px solid rgba(255,255,255,0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
