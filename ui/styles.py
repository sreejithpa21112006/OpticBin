"""
Functional & theme-adaptive dashboard styles for OpticBin.

Focuses on high contrast, readability, clear visual hierarchy,
and zero flashy animations or distractors.
"""

import streamlit as st

IMAGE_MODE = "Image Upload"
WEBCAM_MODE = "Live Webcam"


def apply_styles() -> None:
    """Inject restrained, functional CSS tracking active Streamlit theme."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.5rem;
                max-width: 1200px;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .ob-kicker {
                margin: 0 0 0.25rem 0;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.7;
            }

            .ob-action-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.3);
                border-radius: 8px;
                padding: 1rem 1.25rem;
                margin-bottom: 0.85rem;
            }

            .ob-action-card h3 {
                margin: 0.2rem 0 0.5rem 0;
                font-size: 1.5rem;
                font-weight: 700;
            }

            .ob-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 4px;
                font-size: 0.85rem;
                font-weight: 600;
                border: 1px solid rgba(127, 127, 127, 0.3);
                margin-right: 0.4rem;
            }

            .ob-badge-blue {
                background-color: rgba(59, 130, 246, 0.15);
                color: #2563EB;
                border-color: rgba(59, 130, 246, 0.4);
            }

            .ob-badge-green {
                background-color: rgba(16, 185, 129, 0.15);
                color: #059669;
                border-color: rgba(16, 185, 129, 0.4);
            }

            .ob-badge-gray {
                background-color: rgba(107, 114, 128, 0.15);
                color: #4B5563;
                border-color: rgba(107, 114, 128, 0.4);
            }

            .ob-tip {
                background: var(--secondary-background-color);
                border-left: 4px solid var(--primary-color);
                border-radius: 0 4px 4px 0;
                padding: 0.55rem 0.85rem;
                margin: 0.4rem 0;
                font-size: 0.9rem;
                line-height: 1.4;
            }

            .ob-empty {
                background: var(--secondary-background-color);
                border: 1px dashed rgba(127, 127, 127, 0.4);
                border-radius: 8px;
                padding: 2rem 1.25rem;
                text-align: center;
            }

            .ob-empty h4 {
                margin: 0.3rem 0 0.3rem 0;
                font-size: 1.15rem;
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
