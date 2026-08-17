"""
Theme-adaptive dashboard styles for OpticBin.

Uses Streamlit theme variables so the interface follows light or dark
user settings without hard-coded palette overrides.
"""

import streamlit as st

IMAGE_MODE = "Image Upload"
WEBCAM_MODE = "Live Webcam"


def apply_styles() -> None:
    """Inject restrained CSS that tracks the active Streamlit theme."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .ob-kicker {
                margin: 0;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.62;
            }

            .ob-hero {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.22);
                border-radius: 14px;
                padding: 1.15rem 1.25rem 1.05rem;
                margin-bottom: 0.75rem;
            }

            .ob-hero h2 {
                margin: 0.15rem 0 0.55rem;
                font-size: 1.85rem;
                line-height: 1.2;
            }

            .ob-badge {
                display: inline-block;
                padding: 0.22rem 0.7rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                border: 1px solid rgba(127, 127, 127, 0.28);
            }

            .ob-muted {
                opacity: 0.72;
                font-size: 0.9rem;
                margin-top: 0.55rem;
            }

            .ob-tip {
                background: var(--secondary-background-color);
                border-left: 3px solid var(--primary-color);
                border-radius: 0 8px 8px 0;
                padding: 0.5rem 0.8rem;
                margin: 0.4rem 0;
                font-size: 0.9rem;
            }

            .ob-empty {
                background: var(--secondary-background-color);
                border: 1px dashed rgba(127, 127, 127, 0.35);
                border-radius: 14px;
                padding: 2.4rem 1.5rem;
                text-align: center;
            }

            .ob-empty h3 {
                margin: 0.35rem 0 0.4rem;
            }

            .ob-legend-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
                padding: 0.28rem 0;
                font-size: 0.92rem;
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
