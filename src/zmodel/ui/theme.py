"""Refined light design tokens and restrained custom styling.

Provides a calm, human-crafted engineering interface using warm off-whites,
clean surfaces, dark charcoal typography, and fine neutral borders.
"""

import streamlit as st


LIGHT_THEME_CSS = """
<style>
/* Developer Tool Light Design Tokens */
:root {
    --zm-bg-app: #F8FAFC;
    --zm-bg-surface: #FFFFFF;
    --zm-bg-subtle: #F1F5F9;
    --zm-border-subtle: #E2E8F0;
    --zm-border-strong: #CBD5E1;
    --zm-text-primary: #0F172A;
    --zm-text-secondary: #475569;
    --zm-text-muted: #64748B;
    --zm-accent-blue: #2563EB;
    --zm-accent-indigo: #4F46E5;
    --zm-status-green: #059669;
    --zm-status-amber: #D97706;
    --zm-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

/* Restrained App Header Spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1140px !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--zm-border-subtle) !important;
}

.sidebar-brand-box {
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid var(--zm-border-subtle);
    margin-bottom: 1.25rem;
}
.sidebar-wordmark {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--zm-text-primary);
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: var(--zm-text-muted);
    margin-top: 0.2rem;
    line-height: 1.35;
}
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--zm-text-muted);
    margin: 1.25rem 0 0.5rem 0;
}

/* Top Page Header */
.workspace-header {
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--zm-border-subtle);
}
.workspace-header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}
.workspace-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--zm-text-primary);
    letter-spacing: -0.025em;
    margin: 0;
}
.workspace-desc {
    font-size: 0.9rem;
    color: var(--zm-text-secondary);
    line-height: 1.5;
    max-width: 780px;
    margin: 0;
}

/* Engineering Content Surfaces */
.surface-card {
    background-color: var(--zm-bg-surface);
    border: 1px solid var(--zm-border-subtle);
    border-radius: 8px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
    margin-bottom: 1.25rem;
}
.surface-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--zm-text-primary);
    margin-bottom: 0.25rem;
}
.surface-card-desc {
    font-size: 0.82rem;
    color: var(--zm-text-secondary);
    line-height: 1.45;
    margin-bottom: 1rem;
}

/* Restrained Metric Box */
.metric-box {
    background-color: var(--zm-bg-surface);
    border: 1px solid var(--zm-border-subtle);
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
}
.metric-box-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--zm-text-muted);
    margin-bottom: 0.25rem;
}
.metric-box-val {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--zm-text-primary);
    font-family: var(--zm-font-mono);
    letter-spacing: -0.02em;
}
.metric-box-sub {
    font-size: 0.74rem;
    color: var(--zm-text-muted);
    margin-top: 0.2rem;
}

/* Subtle Badges and Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: var(--zm-font-mono);
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    line-height: 1.1;
}
.pill-local {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
}
.pill-active {
    background-color: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}
.pill-unverified {
    background-color: #F8FAFC;
    color: #64748B;
    border: 1px solid #E2E8F0;
}
.pill-success {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

/* Restrained Empty State Box */
.empty-state-container {
    background-color: var(--zm-bg-surface);
    border: 1px dashed var(--zm-border-strong);
    border-radius: 8px;
    padding: 2.25rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.empty-state-heading {
    font-size: 1rem;
    font-weight: 600;
    color: var(--zm-text-primary);
    margin-bottom: 0.35rem;
}
.empty-state-copy {
    font-size: 0.85rem;
    color: var(--zm-text-secondary);
    max-width: 520px;
    margin: 0 auto 1.25rem auto;
    line-height: 1.5;
}

/* Monospace Code Callout */
.code-inline {
    font-family: var(--zm-font-mono);
    font-size: 0.78rem;
    background-color: #F1F5F9;
    color: #0F172A;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
}

/* Roadmap Notice */
.roadmap-box {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #64748B;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
</style>
"""


def apply_theme() -> None:
    """Inject refined light developer styling."""
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
