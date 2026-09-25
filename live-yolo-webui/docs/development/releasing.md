# Release Process

This document describes how to create and publish a new release of `live-vlm-webui` to PyPI.

## Overview

We use **GitHub Releases** to trigger automated publishing to PyPI. This approach:
- ✅ Creates a tagged release with release notes
- ✅ Automatically builds and publishes to PyPI
- ✅ Attaches wheel artifacts to the GitHub Release
- ✅ Uses PyPI Trusted Publishing (no API tokens needed)

## Prerequisites

### One-time Setup: PyPI Trusted Publishing

Configure PyPI to trust GitHub Actions (more secure than API tokens):

1. Go to [PyPI Publishing Settings](https://pypi.org/manage/account/publishing/)
2. Add a new "pending publisher":
   - **PyPI Project Name**: `live-vlm-webui`
   - **Owner**: `NVIDIA-AI-IOT`
   - **Repository**: `live-vlm-webui`
   - **Workflow**: `build-wheel.yml`
   - **Environment**: (leave blank or use `release`)

3. Click "Add"

Once configured, GitHub Actions can publish without API tokens!

## Version Numbering

Follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **Patch** (`v0.1.1`): Bug fixes, no API changes
- **Minor** (`v0.2.0`): New features, backwards compatible
- **Major** (`v1.0.0`): Breaking changes, incompatible API changes

**Pre-releases:**
- Alpha: `v0.2.0-alpha.1` (early testing, unstable)
- Beta: `v0.2.0-beta.1` (feature complete, testing)
- Release Candidate: `v0.2.0-rc.1` (final testing before release)

## Pre-Release Checklist

Before starting the release process, ensure:

- [ ] All planned features and bug fixes are merged to `main`
- [ ] All tests pass (run individually to avoid event loop conflicts):
  ```bash
  # Unit tests (fast, run in CI)
  pytest tests/unit/ -v

  # Integration tests (component interactions, run in CI)
  pytest tests/integration/ -v

  # Performance tests (requires GPU, local only)
  pytest tests/performance/ -v

  # E2E tests (requires server + Ollama, run separately)
  pytest tests/e2e/test_real_workflow.py::test_full_video_analysis_workflow -v -s
  ```
  **Note:** Running `pytest tests/` all together may cause event loop conflicts between e2e and integration tests. Run them separately as shown above.
- [ ] Code quality checks pass: `./scripts/pre_commit_check.sh`
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated with all changes since last release
- [ ] **CRITICAL: Version files updated and verified** (both must match exactly):
  - `pyproject.toml` - `version = "0.2.0"`
  - `src/live_vlm_webui/__init__.py` - `__version__ = "0.2.0"` ⚠️ **This is what `--version` displays!**
  ```bash
  # Verify both files have matching versions:
  grep '^version =' pyproject.toml
  grep '__version__' src/live_vlm_webui/__init__.py

  # After installation, verify it works:
  live-vlm-webui --version  # Should show correct version
  ```
- [ ] Version number follows [Semantic Versioning](https://semver.org/)
  - **MAJOR**: Breaking changes
  - **MINOR**: New features (backwards compatible)
  - **PATCH**: Bug fixes (backwards compatible)

## Release Workflow

### 1. Prepare the Release

On the `main` branch:

```bash
# 1. Pull latest changes
git checkout main
git pull origin main

# 2. Update version in pyproject.toml
vim pyproject.toml
# Change: version = "0.2.0"

# 3. Update version in __init__.py (CRITICAL: This is what --version displays!)
# ⚠️  MUST MATCH pyproject.toml exactly!
vim src/live_vlm_webui/__init__.py
# Change: __version__ = "0.2.0"

# 3a. Verify both versions match:
grep '^version =' pyproject.toml
grep '__version__' src/live_vlm_webui/__init__.py
# Both should show: 0.2.0

# 4. Update CHANGELOG.md
vim CHANGELOG.md
# Move [Unreleased] items to [0.2.0] section
# Add release date: ## [0.2.0] - 2025-11-08

# 5. Commit version bump
git add pyproject.toml src/live_vlm_webui/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"

# 6. Push to main
git push origin main
```

### 2. Create Git Tag

Git tags trigger the Docker image builds with version-specific tags.

```bash
# Create annotated tag (recommended for releases)
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag to GitHub
git push origin v0.2.0
```

**Important**: The tag format must be `vX.Y.Z` (with the `v` prefix) for the Docker workflow to recognize it as a semver tag.

### 3. Create GitHub Release

1. Go to [Releases](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/releases)
2. Click **"Draft a new release"**
3. Fill in:
   - **Tag**: `v0.2.0` (select the tag you just pushed)
   - **Target**: `main`
   - **Title**: `v0.2.0 - Brief description`
   - **Description**:
     ```markdown
     ## What's New

     - Added configurable GPU monitoring interval
     - Fixed Docker build issues
     - Improved system stats display

     ## Installation

     ```bash
     pip install live-vlm-webui==0.2.0
     ```

     See [CHANGELOG.md](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/CHANGELOG.md) for full details.
     ```

4. Check **"Set as the latest release"** (for production releases)
5. Check **"Create a discussion for this release"** (optional)
6. Click **"Publish release"**

### 4. Automated Publishing

GitHub Actions will automatically:

1. ✅ Build the wheel
2. ✅ Run tests (if configured)
3. ✅ Publish to PyPI (via trusted publishing)
4. ✅ Attach wheel to GitHub Release
5. ✅ Build and push Docker images

**Monitor the workflow:**
- Go to [Actions](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/actions)
- Check the `build-wheel.yml` workflow run
- Check the "Build and Push Docker Images" workflow
- Verify all steps complete successfully

### 5. Docker Image Verification

After pushing the git tag, the GitHub Actions workflow automatically builds and publishes Docker images.

#### Verify Published Images

Check that the following tags exist at:
https://github.com/NVIDIA-AI-IOT/live-vlm-webui/pkgs/container/live-vlm-webui

**Base multi-arch images:**
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.0`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest`

**Platform-specific images:**
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.0-jetson-orin`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.0-jetson-thor`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.0-mac`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-orin`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-thor`
- `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-mac`

#### Test Docker Images

Test the version selection feature:

```bash
# List available versions
./scripts/start_container.sh --list-versions

# Test interactive version picker
./scripts/start_container.sh

# Test specific version
./scripts/start_container.sh --version 0.2.0

# Test latest
./scripts/start_container.sh --version latest
```

### 6. Verify the Release

```bash
# 1. Wait ~5-10 minutes for PyPI to update

# 2. Install from PyPI
pip install --upgrade live-vlm-webui==0.2.0

# 3. Verify version (CRITICAL: Check both!)
python -c "import live_vlm_webui; print(live_vlm_webui.__version__)"
live-vlm-webui --version  # ⚠️ This reads from __init__.py - must match!
# Both should display: 0.2.0

# 4. Test basic functionality
live-vlm-webui --help
```

### 7. Post-Release Tasks

- [ ] Update documentation if needed
- [ ] Announce on relevant channels
- [ ] Close related milestone (if using GitHub Milestones)
- [ ] Update any downstream projects
- [ ] Mark completed items in ROADMAP.md or TODO.md
- [ ] Plan next release milestones
- [ ] Watch for issues related to the new release
- [ ] Respond promptly to installation/upgrade problems

## Release Checklist

Quick reference for releases:

**Pre-release:**
- [ ] All tests passing on main (run individually):
  - `pytest tests/unit/ -v`
  - `pytest tests/integration/ -v`
  - `pytest tests/performance/ -v`
  - `pytest tests/e2e/test_real_workflow.py::test_full_video_analysis_workflow -v -s` (if server/Ollama available)
- [ ] **CRITICAL: Update and verify version in BOTH files:**
  - `pyproject.toml` - `version = "0.2.0"`
  - `src/live_vlm_webui/__init__.py` - `__version__ = "0.2.0"` ⚠️ **Must match!**
  - Verify: `grep '^version =' pyproject.toml && grep '__version__' src/live_vlm_webui/__init__.py`
  - Test: `live-vlm-webui --version` (should show correct version)
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Review open issues/PRs for blockers
- [ ] Test Docker builds locally (optional)

**Release:**
- [ ] Commit and push version bump
- [ ] Create and push git tag `vX.Y.Z`
- [ ] Create GitHub Release with tag `vX.Y.Z`
- [ ] Monitor GitHub Actions workflows
- [ ] Verify PyPI upload successful
- [ ] Verify Docker images published

**Post-release:**
- [ ] Test PyPI installation
- [ ] **Verify version is correct:**
  - `live-vlm-webui --version` (should match release version)
  - `python -c "import live_vlm_webui; print(live_vlm_webui.__version__)"`
- [ ] Verify wheel functionality
- [ ] Test Docker images
- [ ] Update documentation
- [ ] Announce release

## Manual PyPI Upload (Alternative)

If you need to upload manually instead of using GitHub Actions:

### 1. Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build
```

### 2. Test Upload (TestPyPI)

```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ live-vlm-webui==0.2.0
```

### 3. Production Upload

```bash
# Upload to PyPI
python -m twine upload dist/*
```

### 4. Verify PyPI Page

Visit: https://pypi.org/project/live-vlm-webui/

Ensure:
- Correct version is displayed
- README renders correctly
- Project links work
- Classifiers are correct

## Emergency: Yanking a Release

If you need to remove a broken release from PyPI:

```bash
# Install twine
pip install twine

# Yank the release (requires PyPI credentials)
twine yank live-vlm-webui 0.2.0 --reason "Critical bug, use 0.2.1 instead"
```

Then immediately:
1. Fix the issue
2. Release a patch version (e.g., `v0.2.1`)

## Rollback Procedure

If a critical issue is discovered after release:

### 1. Delete Git Tag (if not widely used)

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag
git push origin :refs/tags/v0.2.0
```

### 2. Delete GitHub Release

Go to the release page and click "Delete release"

### 3. Yank PyPI Release

Mark as yanked through PyPI web interface (users can still install explicitly)

### 4. Issue Patch Release

Fix the issue and release v0.2.1 immediately.

## Troubleshooting

### PyPI Upload Fails

**Error: "Project name not found on PyPI"**
- Solution: First release must be uploaded manually or project must exist on PyPI
- Create project on PyPI first, or use TestPyPI for testing

**Error: "Invalid or non-existent authentication"**
- Solution: Verify PyPI Trusted Publishing is configured correctly
- Check repository, workflow name, and owner match exactly

**Error: "File already exists"**
- Solution: You cannot re-upload the same version
- Increment version and create a new release

### GitHub Actions Workflow Fails

1. Check [Actions logs](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/actions)
2. Look for specific error messages
3. Common issues:
   - Build failures: Check dependencies in `pyproject.toml`
   - Test failures: Fix tests before releasing
   - Upload failures: Check PyPI Trusted Publishing config

### Docker Workflow Fails

1. Check GitHub Actions logs
2. Verify Dockerfile syntax
3. Ensure GITHUB_TOKEN has correct permissions
4. Test Docker build locally:
   ```bash
   docker build -f docker/Dockerfile -t test:latest .
   ```

### Version Tags Not Appearing

1. Ensure tag follows `vX.Y.Z` format
2. Check workflow triggers in `.github/workflows/docker-publish.yml`
3. Verify workflow completed successfully

## Testing Releases

Use [TestPyPI](https://test.pypi.org/) for testing the release process:

1. Configure TestPyPI trusted publishing (separate from PyPI)
2. Modify workflow to upload to TestPyPI
3. Test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ live-vlm-webui
   ```

## Quick Reference Commands

```bash
# Complete release workflow
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# Monitor workflows
gh workflow view "Build and Push Docker Images"
gh workflow view "build-wheel"

# Verify Docker images
docker pull ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.0
./scripts/start_container.sh --version 0.2.0

# Build and upload to PyPI (manual)
python -m build
python -m twine upload dist/*
```

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [Docker Metadata Action](https://github.com/docker/metadata-action)
- [Python Packaging Guide](https://packaging.python.org/)
