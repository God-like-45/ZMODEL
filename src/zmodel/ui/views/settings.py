"""Settings and System Configuration View for Z-ModelPilot."""

import streamlit as st
from zmodel.api.config import get_api_settings
from zmodel.core.config import (
    APP_NAME,
    APP_VERSION,
    ENVIRONMENT_TYPE,
    MAX_UPLOAD_SIZE_MB,
)
from zmodel.core.profiles import load_builtin_profiles
from zmodel.core.system_info import get_system_environment
from zmodel.db.config import check_database_connection
from zmodel.ui.components import render_workspace_header


def render_settings_view() -> None:
    """Render the active system settings, REST API endpoints, and architectural boundaries."""
    render_workspace_header(
        title="Settings & System Diagnostics",
        description="Review FastAPI REST API status, database persistence configuration, resource limits, and architectural execution boundaries.",
        environment_badge=f"v{APP_VERSION} · Release Active",
    )

    api_settings = get_api_settings()
    env = get_system_environment()
    db_status = check_database_connection(timeout_seconds=1)

    tab_api, tab_db, tab_bounds, tab_arch = st.tabs([
        "FastAPI REST API",
        "Database & Persistence",
        "Resource Limits & Ceilings",
        "IBM Z & Architecture Disclosure",
    ])

    with tab_api:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A;">FastAPI REST Service (`/api/v1`)</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Versioned REST API for model inspection, capability matching, benchmarking, and optimization.</div>
                    </div>
                    <span class="status-pill pill-active">API v{api_settings.api_version}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; border-top: 1px solid #F1F5F9; padding-top: 0.75rem; font-size: 0.82rem;">
                    <div><span style="color: #64748B;">API Host:</span> <code style="font-size: 0.8rem;">{api_settings.api_host}:{api_settings.api_port}</code></div>
                    <div><span style="color: #64748B;">Environment:</span> <strong style="color: #0F172A;">{api_settings.environment}</strong></div>
                    <div><span style="color: #64748B;">CORS Allowed:</span> <code style="font-size: 0.8rem;">{api_settings.allowed_cors_origins}</code></div>
                    <div><span style="color: #64748B;">Log Level:</span> <strong style="color: #0F172A;">{api_settings.log_level}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <div style="font-weight: 600; font-size: 0.9rem; color: #0F172A; margin-bottom: 0.35rem;">Interactive API Documentation</div>
                <div style="font-size: 0.82rem; color: #475569;">
                    Access real-time OpenAPI schema and interactive test consoles generated directly from Pydantic schemas:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.link_button("Open Swagger UI (/docs)", f"http://localhost:{api_settings.api_port}/docs", use_container_width=True)
        with c2:
            st.link_button("Open ReDoc (/redoc)", f"http://localhost:{api_settings.api_port}/redoc", use_container_width=True)
        with c3:
            st.link_button("View OpenAPI Schema (/openapi.json)", f"http://localhost:{api_settings.api_port}/openapi.json", use_container_width=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Core Versioned REST Endpoints")
        endpoints_data = [
            {"Method": "GET", "Path": "/health/live", "Description": "Process liveness probe."},
            {"Method": "GET", "Path": "/health/ready", "Description": "Dependency readiness check (PostgreSQL connectivity)."},
            {"Method": "POST", "Path": "/api/v1/models/inspect", "Description": "Multipart upload for ONNX parsing and SHA-256 fingerprinting."},
            {"Method": "GET", "Path": "/api/v1/models", "Description": "List registered models from PostgreSQL."},
            {"Method": "GET", "Path": "/api/v1/compatibility/profiles", "Description": "List versioned target capability profiles."},
            {"Method": "POST", "Path": "/api/v1/compatibility/analyze", "Description": "Evaluate model against a capability profile with explainable evidence."},
            {"Method": "POST", "Path": "/api/v1/benchmarks", "Description": "Submit bounded benchmark job (HTTP 202 Accepted)."},
            {"Method": "GET", "Path": "/api/v1/benchmarks/{id}/status", "Description": "Poll benchmark job lifecycle progress."},
            {"Method": "POST", "Path": "/api/v1/optimizations", "Description": "Submit multi-candidate optimization experiment (HTTP 202 Accepted)."},
            {"Method": "GET", "Path": "/api/v1/experiments", "Description": "List persisted optimization experiments from PostgreSQL."},
            {"Method": "POST", "Path": "/api/v1/experiments/compare", "Description": "Scientifically verify comparability between two runs."},
        ]
        st.dataframe(endpoints_data, use_container_width=True, hide_index=True)

    with tab_db:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A;">PostgreSQL Persistence (Phase 6)</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Durable relational storage managed via Alembic migrations.</div>
                    </div>
                    <span class="status-pill {'pill-active' if db_status.get('connected') else 'pill-error'}">
                        {'Connected' if db_status.get('connected') else 'Disconnected / Fallback'}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; border-top: 1px solid #F1F5F9; padding-top: 0.75rem; font-size: 0.82rem;">
                    <div><span style="color: #64748B;">Dialect:</span> <strong style="color: #0F172A;">{db_status.get('dialect', 'N/A')}</strong></div>
                    <div><span style="color: #64748B;">URL (Masked):</span> <code style="font-size: 0.78rem;">{db_status.get('database_url_masked', 'N/A')}</code></div>
                    <div><span style="color: #64748B;">Tables Count:</span> <strong style="color: #0F172A;">{db_status.get('table_count', 0)}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Migration Commands")
        st.code(
            "# Apply migrations to latest schema\n"
            "alembic upgrade head\n\n"
            "# Check current database revision\n"
            "alembic current\n\n"
            "# Docker Compose migration\n"
            "docker compose run --rm api alembic upgrade head",
            language="bash",
        )

    with tab_bounds:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 0.35rem;">System Ceilings & Concurrency Controls</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;">Hardened bounds protecting system resources from unbounded jobs or memory exhaustion:</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; font-size: 0.85rem;">
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <div style="color: #64748B; font-size: 0.72rem; text-transform: uppercase;">Max Model Upload Size</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">50 MB</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Prevents unbounded memory parsing.</div>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <div style="color: #64748B; font-size: 0.72rem; text-transform: uppercase;">Benchmark Iteration Ceiling</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">1,000 Iterations</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Prevents long-running thread lockups.</div>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <div style="color: #64748B; font-size: 0.72rem; text-transform: uppercase;">Max Optimization Candidates</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">50 Candidates</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Bounded search space per experiment.</div>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <div style="color: #64748B; font-size: 0.72rem; text-transform: uppercase;">Job Concurrency Limit</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">2 Active Workers</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Prevents thread pool thrashing.</div>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <div style="color: #64748B; font-size: 0.72rem; text-transform: uppercase;">Job Queue Capacity</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">20 Queued Jobs</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Excess returns HTTP 429 Too Many Requests.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_arch:
        st.markdown(
            """
            <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 600; color: #92400E; font-size: 0.95rem; margin-bottom: 0.4rem;">
                    IBM Z Alignment & Technical Honesty Guidelines
                </div>
                <div style="font-size: 0.82rem; color: #78350F; line-height: 1.5;">
                    Z-ModelPilot models enterprise AI deployment architectures aligned with the IBM AI on Z ecosystem. 
                    However, the platform maintains strict empirical honesty:
                </div>
                <ul style="font-size: 0.8rem; color: #78350F; margin-top: 0.5rem; padding-left: 1.25rem; line-height: 1.5;">
                    <li><strong>Host Execution Boundary:</strong> All empirical benchmark measurements (latency, throughput, peak memory RSS) reflect execution on the local host processor (x86_64 / ARM64) and memory bus. Local CPU execution is never presented as evidence of s390x or IBM Telum performance.</li>
                    <li><strong>Illustrative Acceleration Profiles:</strong> Hardware acceleration via IBM Integrated Accelerator for AI (NNPA) is provided as an illustrative ruleset (<code>ibm_z_onnx_illustrative.json</code>) clearly tagged with <code>is_illustrative_profile=True</code>. Hardware speedups are never claimed as measured without verified access to physical IBM Z hardware.</li>
                    <li><strong>Preservation of Uncertainty:</strong> The compatibility engine preserves <code>PASS</code>, <code>FAIL</code>, <code>WARNING</code>, and <code>UNKNOWN</code> distinctions. Missing rules are never treated as proof of compatibility.</li>
                    <li><strong>No Fabricated Metrics:</strong> Unmeasured fields or skipped candidate evaluations are explicitly preserved as <code>None</code> / SQL <code>NULL</code> rather than substituted with synthetic zeros.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Loaded Capability Profiles")
        profiles = load_builtin_profiles()
        profile_rows = []
        for pid, p in sorted(profiles.items()):
            profile_rows.append({
                "Profile ID": p.profile_id,
                "Display Name": p.display_name,
                "Version": p.version,
                "Ruleset": p.ruleset_version,
                "Target Device": f"{p.target_architecture} / {p.target_runtime}",
                "Classification": p.confidence_classification,
                "Illustrative": "Yes (Draft)" if p.is_illustrative else "No (Verified)",
                "Operator Rules": len(p.operator_rules),
            })
        st.dataframe(profile_rows, use_container_width=True, hide_index=True)
