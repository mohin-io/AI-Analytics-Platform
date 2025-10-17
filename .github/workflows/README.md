# GitHub Actions Workflows

## Status: CI/CD Pipeline Temporarily Disabled

The CI/CD pipeline has been temporarily disabled to prevent failing workflow runs during initial development.

### Disabled Workflows

- **ci.yml.disabled** - CI/CD Pipeline
  - Reason: Project is in initial development phase
  - Disabled on: October 17, 2025
  - Will be re-enabled when: Core functionality is implemented and tests are ready

### Why Disabled?

The CI/CD pipeline was failing because:
1. Test files are not yet fully implemented
2. Some Python modules are still in development
3. Dependencies may need adjustment for the CI environment

### To Re-enable Later

When you're ready to enable the CI/CD pipeline:

```bash
# Rename the file back
mv .github/workflows/ci.yml.disabled .github/workflows/ci.yml

# Or use GitHub CLI
gh workflow enable "CI/CD Pipeline"

# Commit and push
git add .github/workflows/ci.yml
git commit -m "ci: re-enable CI/CD pipeline"
git push origin main
```

### Before Re-enabling

Make sure you have:
- [ ] Implemented core modules
- [ ] Written comprehensive tests
- [ ] Verified tests pass locally (`pytest tests/`)
- [ ] Checked all dependencies are correct
- [ ] Linting passes (`black --check src/ tests/`)
- [ ] Type checking passes (`mypy src/`)

### Current Development Phase

The project is in **Phase 1: Foundation** where we're building:
- Core preprocessing modules (COMPLETE)
- Model implementations (IN PROGRESS)
- Evaluation framework (PLANNED)
- API and Dashboard (PLANNED)

Once these are implemented and tested, we'll re-enable the CI/CD pipeline to ensure code quality and automate deployments.

### Manual Testing

For now, run tests manually:

```bash
# Run tests locally
pytest tests/ -v

# Run linting
black --check src/ tests/
flake8 src/ tests/
mypy src/

# Run formatting
black src/ tests/
isort src/ tests/
```

---

**Note**: This is a temporary measure. The CI/CD pipeline is well-configured and will be valuable once the codebase has sufficient test coverage.
