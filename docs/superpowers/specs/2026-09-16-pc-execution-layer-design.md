# Chemistry PC Execution Layer Design

## Purpose

Extend the existing `TheLitis/Chemistry` repository from a file-delivery bridge into a controlled execution system for the user's Windows PC. The existing fast-forward sync bridge remains unchanged in responsibility: it delivers repository changes to `C:\Users\loval\Chemistry`. A separate execution layer will run approved GitHub Actions jobs on that PC, expose the machine's local CPU/GPU/RAM/disk environment to those jobs, and return logs/status/artifacts through GitHub.

The target machine is Windows 11 with Git and GitHub CLI already authenticated as `TheLitis`. The known hardware target is an Intel Core i9-10900KF, NVIDIA RTX 3070 8 GB, and 64 GB system RAM. The design must not assume all workloads should pin every resource to 100%; instead it must ensure there are no artificial repository-level limits and must report actual bottlenecks.

## Goals

1. Keep GitHub -> PC file synchronization working independently of execution.
2. Add a repository-scoped GitHub Actions self-hosted runner named `ChemistryPC`.
3. Run the runner as a Windows service so it starts automatically after reboot without an interactive terminal.
4. Restrict workflows intended for the PC to the labels `self-hosted`, `Windows`, `X64`, and `ChemistryPC`.
5. Provide a hardware smoke test that records CPU/thread count, total/free RAM, local disks/free space, NVIDIA GPU details, driver/CUDA visibility, Python version, PyTorch availability, and PyTorch CUDA visibility when installed.
6. Provide a real compute smoke test that exercises CPU threads, allocates a bounded amount of RAM, and, when PyTorch+CUDA are available, performs a GPU matrix-multiplication benchmark without exhausting VRAM.
7. Store diagnostic results as workflow artifacts and surface them in the Actions logs.
8. Provide a reusable job-launch mechanism for approved repository tasks rather than automatically executing arbitrary files merely because they arrive through Git.
9. Ensure jobs run with normal user/service permissions by default, not as an unrestricted administrator shell.
10. Preserve local datasets, model weights, caches, credentials, and generated outputs outside Git tracking; workflows may use them but must not commit or upload them unless explicitly configured.
11. Make installation idempotent enough that rerunning the installer repairs/reconfigures the same repository-scoped runner rather than creating uncontrolled duplicates.
12. Provide clear health diagnostics for runner service state and recent job capability.

## Non-goals

- Do not replace the existing `bridge/sync.ps1` fast-forward synchronizer.
- Do not execute every new commit automatically as a shell command.
- Do not expose SSH, RDP, WinRM, or another inbound remote-control port.
- Do not run the self-hosted runner with permanent administrator privileges solely for convenience.
- Do not promise that every workload will use 100% CPU, GPU, and RAM simultaneously.
- Do not automatically upload local private datasets, model weights, secrets, or arbitrary files to GitHub.
- Do not install or repair NVIDIA display drivers automatically.
- Do not accept Kaggle rules, authenticate third-party services, or submit competition entries without a separately authorized workflow.

## Architecture

### 1. Existing file bridge

The existing `bridge/sync.ps1` continues polling `origin/main` and applying only safe fast-forward updates to the local repository. It remains a delivery mechanism only. Execution-system files are delivered by this bridge like any other repository files, but the bridge itself never evaluates or invokes them.

### 2. Repository-scoped self-hosted runner

A new installer under `runner/install.ps1` configures a GitHub Actions runner specifically for `TheLitis/Chemistry`.

The installer will:

- require an elevated PowerShell only for installation/service-registration operations;
- use the already authenticated GitHub CLI to request a short-lived repository runner registration token at runtime;
- resolve the current GitHub Actions runner release dynamically from GitHub's public release metadata rather than hard-coding a permanent registration token or secret in the repository;
- download the x64 Windows runner archive into a local state directory such as `%LOCALAPPDATA%\ChemistryRunner` or another non-Git directory;
- configure the runner with repository URL `https://github.com/TheLitis/Chemistry`, runner name `ChemistryPC`, and custom label `ChemistryPC`;
- install it as a Windows service using the runner's supported service installation mechanism;
- make reruns detect an existing installation and avoid producing multiple active runners for the same machine.

The resulting GitHub labels used by PC workflows are:

```yaml
runs-on: [self-hosted, Windows, X64, ChemistryPC]
```

No workflow that lacks the `ChemistryPC` label should be treated as targeting this machine.

### 3. Runner workspace and repository workspace

GitHub Actions will use the runner's own `_work` checkout as the ephemeral job workspace. It must not use the live synchronized `C:\Users\loval\Chemistry` checkout as its normal checkout because Actions routinely cleans and mutates its working directory.

When a job needs persistent local resources, workflows will refer to explicit host paths through a small machine configuration file outside Git, for example `%LOCALAPPDATA%\ChemistryRunner\machine.json`. The initial configuration records the synchronized repository path and optional persistent data/cache roots. This keeps GitHub's temporary checkout isolated from user data and from the bridge's clean-working-tree requirement.

### 4. Machine configuration

A local JSON configuration generated by the installer defines machine-specific paths and limits, for example:

```json
{
  "repoPath": "C:\\Users\\loval\\Chemistry",
  "dataRoot": "C:\\Users\\loval\\ChemistryData",
  "cacheRoot": "C:\\Users\\loval\\AppData\\Local\\ChemistryRunner\\cache",
  "maxRamSmokeGiB": 8,
  "runnerName": "ChemistryPC"
}
```

The repository contains a schema/reader but not the user's private data. Workflows must fail with a clear diagnostic if a required configured path does not exist.

### 5. Health and hardware diagnostics

A script `runner/diagnostics.ps1` produces both human-readable output and a JSON report. It collects:

- Windows version and hostname;
- logical processor count and CPU model;
- total and available physical memory;
- fixed disks and free space;
- `nvidia-smi` presence, GPU name, VRAM total/free, driver version, reported CUDA version;
- Python launcher and interpreter versions;
- whether `torch` imports successfully;
- `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`, detected CUDA device count and names;
- runner service status when invoked interactively outside a job.

Missing optional software is reported as `available: false`; it is not silently treated as success.

### 6. Compute smoke test

A Python helper `runner/compute_smoke.py` runs bounded verification workloads.

CPU phase:

- use up to the detected logical CPU count;
- perform independent numeric work long enough to confirm parallel execution without becoming a long stress test;
- report wall time and worker count.

RAM phase:

- allocate and touch a bounded configurable amount of memory, defaulting to the smaller of 8 GiB or 25% of currently available RAM;
- immediately release it after verification;
- never attempt to fill all 64 GB merely to show utilization.

GPU phase:

- only run if PyTorch imports and CUDA is available;
- perform repeated FP16/FP32 matrix multiplications sized to fit comfortably within detected free VRAM;
- synchronize CUDA before timing;
- report device, peak PyTorch allocated/reserved memory and elapsed time;
- fail the GPU capability check if CUDA was expected by a workflow flag but unavailable.

The smoke test validates access to hardware; it does not benchmark the machine as a formal performance suite.

### 7. Workflows

#### `pc-health.yml`

Supports `workflow_dispatch` and runs only on `[self-hosted, Windows, X64, ChemistryPC]`. It runs diagnostics and the compute smoke test, then uploads reports as an artifact. It is the canonical end-to-end proof that GitHub can dispatch a job to the actual PC and receive results.

#### `pc-task.yml`

Provides a controlled reusable execution surface via `workflow_dispatch`. Inputs select from known task types instead of accepting an unrestricted shell string. Initial task types are:

- `tests`: run repository tests;
- `casmi26`: run the repository CASMI26 entry point against configured local data paths;
- `command-file`: execute a reviewed script file committed under an approved directory such as `tasks/`.

The workflow does not take an arbitrary command input. Adding a new task means adding or reviewing a repository script, which creates an auditable Git history.

### 8. CASMI26 integration

The existing CASMI26 prototype remains scientific application code. The execution layer only supplies the ability to run it on the real PC.

A PC task wrapper will:

- locate the configured local data directory;
- create/reuse a task-specific virtual environment outside the Git checkout or in the runner workspace;
- install dependencies only from the committed requirements file;
- print Python/PyTorch/CUDA diagnostic information before inference;
- invoke `predict.py` with explicit local paths;
- upload only the intended small result files/reports unless a workflow explicitly opts into larger artifacts.

The execution layer must not imply that successfully running CASMI26 proves molecular-identification accuracy or an official Kaggle score.

## Security model

The private repository is the control plane. Jobs on the PC can execute repository code, so anyone with write/workflow-dispatch permission to `TheLitis/Chemistry` is effectively trusted to run code as the runner service account. For that reason:

- the runner is repository-scoped, not organization-wide;
- PC workflows use the `ChemistryPC` label;
- arbitrary shell command inputs are not exposed;
- pull-request events from untrusted forks are not used to trigger PC jobs;
- PC workflows default to `workflow_dispatch` or trusted `push` events only when explicitly designed;
- secrets are supplied through GitHub Actions secrets/environment mechanisms when needed, never committed;
- the service account should not be a permanent local administrator;
- installation may require an elevated one-time PowerShell session, but normal jobs should run without elevation.

## Reliability and self-healing

The runner is installed as a Windows service with automatic startup. The installation/repair script can be run again after runner corruption or migration. A health script reports service state and GitHub-runner directory integrity.

The workflow layer uses:

- explicit job timeouts;
- concurrency groups for PC workflows so multiple GPU-heavy tasks do not accidentally contend on one GPU;
- fail-fast diagnostics when required local paths or dependencies are unavailable;
- artifact upload with `if: always()` for diagnostic reports;
- no mutation of the synchronized repository checkout during normal Action jobs.

GitHub Actions itself provides job logs and final exit status. Failed jobs are not silently retried unless the specific workflow defines a bounded retry for a known transient operation such as dependency download.

## Resource-utilization policy

The execution layer does not impose small hard limits on CPU/RAM/GPU. Individual workloads can use the machine's available resources according to their own libraries and settings.

For ML workloads, wrappers should expose and log relevant controls rather than forcing universal values: DataLoader worker count, batch size, mixed precision, CUDA device selection, thread environment variables, cache paths and temporary directories. Diagnostics record utilization capability so a later workload can distinguish compute, memory, storage, or data-pipeline bottlenecks.

The system explicitly avoids a policy of forcing CPU, GPU and RAM to 100% simultaneously, because that can reduce throughput and cause paging/OOM failures.

## Installation and upgrade flow

1. Repository changes arrive through the already working bridge.
2. The user performs one elevated installation command to register the repository-scoped runner service. This is the only unavoidable manual bootstrap because Windows service registration and GitHub runner registration require local authorization.
3. After installation, workflows can be dispatched through GitHub without an interactive terminal on the PC.
4. Runner/project scripts are updated through `main` by the existing bridge and by fresh Action checkouts.
5. A repair command can reinstall/reconfigure the runner if service state becomes unhealthy.

## Acceptance criteria

The execution layer is complete when all of the following are demonstrated:

1. Existing bridge tests still pass.
2. New execution-layer parser/unit tests pass in hosted Windows CI.
3. `runner/install.ps1` is syntactically valid, idempotency-safe, and contains no committed registration token.
4. After the one-time local install, GitHub shows an online repository-scoped runner named `ChemistryPC` with the expected labels.
5. Dispatching `pc-health.yml` schedules on `ChemistryPC`, not on a GitHub-hosted runner.
6. The health job returns a report from the actual machine showing the real CPU/RAM/GPU inventory.
7. The GPU smoke phase either performs CUDA work successfully or returns an explicit actionable reason CUDA is unavailable; no false success is permitted.
8. The job uploads diagnostic JSON/text artifacts and returns a correct exit code.
9. A controlled task workflow can run repository tests and the CASMI26 wrapper on configured local inputs without modifying the live bridge checkout.
10. Reboot/login is no longer required to start the runner interactively: the Windows service starts automatically.
11. No workflow exposes a generic arbitrary-command text box or automatically executes every synchronized commit.
12. Documentation describes installation, health checking, task dispatch, repair/uninstall and the trust model.

## Testing strategy

- Static PowerShell parser tests for every new `.ps1` file.
- Unit tests for configuration validation and Python smoke-test sizing logic.
- Hosted Windows CI to ensure scripts parse and tests run without depending on the user's hardware.
- Negative tests for missing `nvidia-smi`, missing PyTorch, missing CUDA, missing configured data paths and unsupported task names.
- The decisive integration test is a manual `workflow_dispatch` of `pc-health.yml` after runner installation; only this proves execution on the physical PC.
- CASMI26 remains separately validated for scientific correctness; execution success is not accuracy evidence.

## Operational boundary

Once the one-time runner service is installed, routine repository jobs should not require the user to copy commands into PowerShell. The user may still need to intervene for failures outside repository control, including Windows credential loss, driver corruption, disk exhaustion, internet outages, security software blocks, or third-party reauthentication. The execution layer should surface those failures precisely rather than masking them.
