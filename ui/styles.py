"""
Modern, streamlined styling for OpticBin dashboard.
Focuses on clear visual hierarchy, high readability, color-coded bin guidance,
and zero clutter.
"""

import streamlit as st

IMAGE_MODE = "Image File Upload"
WEBCAM_MODE = "Camera Viewfinder"


def apply_styles() -> None:
    """Inject clean, modern CSS with high contrast and intuitive visual hierarchy."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1240px;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            /* Header styling */
            .ob-header-container {
                margin-bottom: 1.2rem;
                padding-bottom: 0.8rem;
                border-bottom: 1px solid rgba(127, 127, 127, 0.2);
            }

            .ob-title {
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin: 0;
            }

            .ob-subtitle {
                font-size: 1rem;
                opacity: 0.75;
                margin: 0.2rem 0 0 0;
            }

            /* Hero Bin Recommendation Card */
            .ob-bin-hero {
                border-radius: 12px;
                padding: 1.4rem 1.6rem;
                margin-bottom: 1.2rem;
                border: 2px solid;
                position: relative;
            }

            .ob-bin-hero-plastic {
                background: rgba(245, 158, 11, 0.08);
                border-color: #F59E0B;
            }

            .ob-bin-hero-paper {
                background: rgba(59, 130, 246, 0.08);
                border-color: #3B82F6;
            }

            .ob-bin-hero-cardboard {
                background: rgba(180, 83, 9, 0.08);
                border-color: #B45309;
            }

            .ob-bin-hero-metal {
                background: rgba(14, 165, 233, 0.08);
                border-color: #0EA5E9;
            }

            .ob-bin-hero-glass {
                background: rgba(16, 185, 129, 0.08);
                border-color: #10B981;
            }

            .ob-bin-hero-biodegradable {
                background: rgba(22, 163, 74, 0.08);
                border-color: #16A34A;
            }

            .ob-bin-kicker {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }

            .ob-bin-title {
                font-size: 1.85rem;
                font-weight: 800;
                margin: 0 0 0.6rem 0;
                line-height: 1.2;
            }

            .ob-bin-meta-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
                margin-top: 0.6rem;
            }

            .ob-pill {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                background: rgba(127, 127, 127, 0.15);
                border: 1px solid rgba(127, 127, 127, 0.3);
            }

            /* Checklist items */
            .ob-checklist {
                background: var(--secondary-background-color);
                border-radius: 10px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(127, 127, 127, 0.2);
            }

            .ob-checklist-item {
                display: flex;
                align-items: flex-start;
                margin: 0.45rem 0;
                font-size: 0.92rem;
                line-height: 1.4;
            }

            .ob-checklist-bullet {
                margin-right: 0.6rem;
                font-weight: bold;
                color: #10B981;
            }

            /* Empty state placeholder */
            .ob-empty-state {
                border: 2px dashed rgba(127, 127, 127, 0.35);
                border-radius: 12px;
                padding: 3rem 1.5rem;
                text-align: center;
                background: rgba(127, 127, 127, 0.04);
            }

            .ob-empty-state h3 {
                margin: 0 0 0.5rem 0;
                font-size: 1.25rem;
            }

            .ob-empty-state p {
                font-size: 0.95rem;
                opacity: 0.7;
                margin: 0;
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
