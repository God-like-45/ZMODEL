# IBM AI on Z Validation & Deployment Readiness Guide

**Platform:** Z-ModelPilot — Architecture-Aware AI Deployment Optimizer  
**Release:** `v1.3.0` (Phase 11: IBM Z / Linux on IBM Z Validation and Deployment Readiness)  
**Target Environments:**
- Host Development: Local x86_64 / AMD64 (Windows / Linux / macOS)
- Target Hardware: IBM z16 / LinuxONE 4 (`s390x` architecture with Integrated Accelerator for AI / NNPA)
- Target Standard: Linux on IBM Z (`s390x` general-purpose CPU SIMD)

---

## 1. What IBM Z Validation Means in Z-ModelPilot

In enterprise mainframe systems engineering, credibility is rooted in technical honesty and verifiable reproducibility. A system must never treat an x86_64 development host, CI runner, or Kubernetes cluster as proof of IBM Z compatibility or performance.

Z-ModelPilot introduces an unambiguous **5-Tier Validation Status Classification**:

| Validation Tier | Identifier | Definition & Criteria |
|---|---|---|
| **Tier 1: Target Verified** | `TARGET_VERIFIED` | Tested directly on an explicitly identified, verified Linux on IBM Z (`s390x`) physical LPAR or z/VM guest. All mandatory hardware, OS, runtime, and numerical tests passed on target silicon. |
| **Tier 2: Locally Verified** | `LOCALLY_VERIFIED` | Evaluated on local development host (e.g., `x86_64` / AMD64). Model loadability, ONNX operator coverage, graph validity, and local reference execution passed, but **IBM Z target execution remains pending**. |
| **Tier 3: Partially Verified** | `PARTIALLY_VERIFIED` | Some target requirements are confirmed, but one or more non-critical checks produced `WARNING` or `UNKNOWN` results (e.g., optional accelerator unconfirmed or dynamic shape warnings). |
| **Tier 4: Not Verified** | `NOT_VERIFIED` | Target hardware, remote runner credentials, or target runtime dependencies are inaccessible or unconfigured. |
| **Tier 5: Failed Validation** | `FAILED_VALIDATION` | A real test was executed against a target profile or environment and failed (e.g., unsupported operator, load failure, or numerical divergence exceeding tolerance). |

> **Fundamental Principle:**
> *"IBM Z target validation remains pending; only the listed local checks have been verified."*  
> If target hardware is not attached, Z-ModelPilot reports local facts truthfully, marks target-specific checks as `NOT_RUN`, and provides actionable remediation guidance.

---

## 2. Environment Fact Detection

Z-ModelPilot inspects the runtime environment safely using read-only standard system facilities (`zmodel.core.environment.EnvironmentInventory`). Root privileges are never required.

### Facts Captured
- **Machine Architecture:** Detected via `platform.machine()` and `os.uname()`. Distinguishes `s390x` from `x86_64`, `AMD64`, `aarch64`.
- **Operating System:** OS system name, kernel release, and distribution identification (via `/etc/os-release` on Linux).
- **Python Runtime:** Python implementation, major/minor/patch versions, and system architecture bitness.
- **CPU Model & Facilities:**
  - On Linux (`/proc/cpuinfo`): inspects machine type, processor model, and hardware facility bits. Specifically looks for **Facility Bit 193** (NNPA — Neural Network Processing Assist).
  - Non-Linux / Windows: safe fallback using `platform.processor()` and environment variables.
- **ONNX Runtime:** Installed ORT version and list of available Execution Providers (`get_available_providers()`).
- **Container Environment:** Checks for `/.dockerenv` or container cgroup signatures.
- **Environment Fingerprint:** Computes a deterministic SHA-256 hash over sorted environment facts for immutable audit logging.

Missing, restricted, or inaccessible properties are explicitly recorded as `UNKNOWN` rather than guessed or fabricated.

---

## 3. Target Profiles & Versioning

Target profiles reside in versioned, declarative JSON files under `profiles_data/` and are encapsulated by `zmodel.core.target_profiles.TargetValidationProfile`.

### Included Profiles
1. **`ibm_z16_linux_nnpa` (`v1.3.0`):**
   - Target Architecture: `s390x`
   - Required OS: Linux (`rhel >= 8.6`, `sles >= 15`, `ubuntu >= 22.04`)
   - Required Execution Provider: `NNPAExecutionProvider`
   - Requires Accelerator: `True` (IBM Telum Integrated Accelerator for AI, Facility 193)
   - Tolerances: `atol=1e-3`, `rtol=1e-2` (accounting for DLFloat16 / FP32 rounding)
2. **`ibm_z_linux_cpu` (`v1.3.0`):**
   - Target Architecture: `s390x`
   - Required OS: Linux on IBM Z
   - Required Execution Provider: `CPUExecutionProvider`
   - Requires Accelerator: `False` (General s390x SIMD / Vector Facility)
   - Tolerances: `atol=1e-4`, `rtol=1e-3`
3. **`local_host_development` (`v1.3.0`):**
   - Target Architecture: Current host (`x86_64` / `AMD64` / `aarch64`)
   - Purpose: Baseline local verification prior to mainframe deployment.

---

## 4. The 12-Check Architecture & Runtime Compatibility Battery

The `ValidationEngine` evaluates an ONNX model against a target profile across 12 distinct checks:

| Check ID | Check Name | Target-Only? | Evaluation Criteria & Logic |
|---|---|---|---|
| `CHK-01-ARCH` | Machine Architecture | Yes | Compares detected `machine_architecture` against profile requirement (`s390x`). |
| `CHK-02-OS` | Operating System Compatibility | Yes | Verifies OS family matches target requirements (e.g. Linux). |
| `CHK-03-PYVER` | Python Runtime Version | No | Checks Python version against target version constraint (e.g., `>=3.9, <3.13`). |
| `CHK-04-ORTVER` | ONNX Runtime Version | No | Checks installed ORT version against minimum required version. |
| `CHK-05-PROV` | Execution Provider Availability | Yes | Checks if required execution provider (e.g. `NNPAExecutionProvider`) is available without silent fallback. |
| `CHK-06-LOAD` | Model Loadability | No | Instantiates an ONNX Runtime inference session with target/fallback providers. |
| `CHK-07-IOSIG` | Input/Output Signature | No | Validates tensor shapes and data types against target profile constraints. |
| `CHK-08-OPSET` | Operator Set & Rules | No | Inspects model operators against target architecture compatibility rules. |
| `CHK-09-SHLIB` | Shared Library Dependencies | Yes | Checks for presence of target runtime libraries (e.g., `libnnpa.so`). |
| `CHK-10-NUMER` | Numerical Correctness | No | Compares inference outputs across runs or against reference data (`all_finite`, repeatability, `atol`/`rtol`). |
| `CHK-11-BENCH` | Benchmark Eligibility | No | Assesses whether the model meets correctness and latency prerequisites for benchmarking. |
| `CHK-12-ACCEL` | Accelerator Availability | Yes | On s390x, checks for Telum NNPA hardware facility bit 193. |

### Check Status Taxonomy
- **`PASS`**: Condition evaluated and verified satisfied with concrete evidence.
- **`FAIL`**: Condition evaluated and explicitly violated.
- **`WARNING`**: Non-fatal condition detected (e.g. dynamic shapes or minor version discrepancy).
- **`UNKNOWN`**: Metric could not be inspected or determined without elevated privilege or target access.
- **`NOT_RUN`**: Check requires physical IBM Z target execution that was not available in current environment.

---

## 5. Connecting an Authorized IBM Z / Linux on IBM Z Target

When physical or virtual Linux on IBM Z hardware is available, Z-ModelPilot can orchestrate remote target validation via `RemoteSSHTargetRunner`.

### Configuration via Environment Variables
Remote target connection is strictly opt-in and gated behind explicit configuration:

```bash
# Target connection configuration (never hardcoded in source)
export ZMODEL_TARGET_HOST="s390x-lpar01.corp.ibm.com"
export ZMODEL_TARGET_PORT="22"
export ZMODEL_TARGET_USER="zmodel_runner"
export ZMODEL_TARGET_SSH_KEY_PATH="/home/user/.ssh/id_rsa_ibmz"
export ZMODEL_TARGET_PYTHON_PATH="/opt/ibm/venv/bin/python"
```

### Security Architecture & Precautions
1. **Strict Command Whitelist:** Z-ModelPilot never executes arbitrary user-supplied shell scripts. Remote runners only execute an explicit, documented set of read-only probe commands (`uname -m`, `cat /proc/cpuinfo`, `python -c "import onnxruntime..."`).
2. **Secret Redaction:** All SSH keys, passwords, and authorization tokens are redacted (`***REDACTED***`) before recording in logs, database tables, API responses, or UI dashboards.
3. **Non-Destructive & Unprivileged:** Runs entirely under standard user privileges. Does not invoke `sudo` or modify system files.
4. **Timeouts & Safe Failure:** All network operations apply strict 30-second timeouts to prevent blocking or hanging worker processes.

---

## 6. Real Target Validation Execution Flow

```
+-----------------------------------------------------------------------------------+
|                           Target Validation Workflow                              |
+-----------------------------------------------------------------------------------+
                                          |
  1. Target Environment Inspection       v
     Safe probe: architecture (s390x), OS, CPU facilities (NNPA bit 193), ORT version
                                          |
  2. Artifact Integrity Verification     v
     Compute SHA-256 hash of model artifact prior to transfer & verify on target
                                          |
  3. Prerequisite Evaluation             v
     Execute 12-check battery; verify provider, loadability, and operator coverage
                                          |
  4. Numerical Verification              v
     Run target inference; evaluate output finiteness, repeatability, and tolerances
                                          |
  5. Controlled Benchmark Execution      v
     Execute warmup (>=5 iterations) and measured runs (>=50 iterations)
     Record pure inference kernel time using target s390x time-of-day clock
                                          |
  6. Provenance Tagging & Persistence    v
     Tag benchmark record with IBM_Z_NATIVE or IBM_Z_CONTAINER provenance
     Persist complete validation report in PostgreSQL validation_runs table
```

---

## 7. Numerical Correctness Evaluation

Numerical correctness is evaluated under `CHK-10-NUMER` using deterministic reference inputs:

1. **Finiteness:** Verifies that no output elements are `NaN` or `+/- Inf`.
2. **Repeatability:** Executes multiple consecutive iterations on identical inputs; checks that max absolute discrepancy is zero or within numerical noise (`< 1e-7`).
3. **Reference Comparison:** When reference outputs are provided, computes:
   $$\text{Max Abs Error} = \max |y_{\text{target}} - y_{\text{ref}}|$$
   $$\text{Max Rel Error} = \max \left| \frac{y_{\text{target}} - y_{\text{ref}}}{y_{\text{ref}} + \epsilon} \right|$$
   Ensures:
   $$\text{Max Abs Error} \le \text{atol} \quad \text{and} \quad \text{Max Rel Error} \le \text{rtol}$$

---

## 8. Benchmark Provenance & Environment Separation

Every benchmark record in Z-ModelPilot is tagged with its immutable execution provenance:

- `LOCAL_HOST`: Measured on the local development workstation (x86_64).
- `IBM_Z_NATIVE`: Measured directly on physical Linux on IBM Z bare-metal / LPAR.
- `IBM_Z_CONTAINER`: Measured inside a container running on Linux on IBM Z (`s390x`).
- `REMOTE_TARGET`: Measured on an authorized remote execution target.
- `SIMULATED_ILLUSTRATIVE`: Clearly designated illustrative capability model.

Local x86_64 performance figures are never directly compared to IBM Z benchmarks or presented as equivalent.

---

## 9. What Cannot Be Verified Without Physical Target Access

In the absence of an authorized s390x target:
1. **Hardware NNPA Acceleration:** Actual dispatch of the unprivileged machine instruction `NNPA` (facility bit 193) cannot be executed on x86_64.
2. **True Latency & Throughput:** Inference latency and throughput on the IBM Telum coprocessor cannot be measured on non-s390x silicon.
3. **Shared Dynamic Libraries:** Target-specific shared libraries (`libnnpa.so`, IBM AI Toolkit binaries) cannot be dynamically linked or resolved on x86_64 hosts.
4. **Endianness Effects:** Potential numerical deviations due to s390x big-endian architecture versus x86_64 little-endian architecture cannot be verified without physical execution.

---

## 10. Verification of Current Development Environment

In the current development environment:
- **Host Architecture:** AMD64 / x86_64 (Windows 11)
- **Local Checks Run:** `CHK-03-PYVER`, `CHK-04-ORTVER`, `CHK-06-LOAD`, `CHK-07-IOSIG`, `CHK-08-OPSET`, `CHK-10-NUMER`, `CHK-11-BENCH` executed and passed on local host.
- **Target Checks:** `CHK-01-ARCH`, `CHK-02-OS`, `CHK-05-PROV`, `CHK-09-SHLIB`, `CHK-12-ACCEL` are marked `NOT_RUN` with status rationale: *Target s390x access is required to evaluate this check.*
- **Overall Status:** `LOCALLY_VERIFIED` (or `PARTIALLY_VERIFIED` against IBM Z target profile).

**Summary Statement:**
*“IBM Z target validation remains pending; only the listed local checks have been verified.”*
