# Branch Protection Configuration

This document describes the branch protection rules configured for this repository.

## Protected Branches

### Main Branch (`main`)

The `main` branch is the primary production branch and has the following protections enabled:

#### Active Protections

✅ **Prevent Force Pushes**
- Force pushes to `main` are blocked
- Protects against accidental history rewrites
- Ensures commit history integrity

✅ **Prevent Branch Deletion**
- The `main` branch cannot be deleted
- Safeguards production codebase

✅ **Require Linear History**
- Enforces clean, linear commit history
- Merge commits are prevented
- Requires rebase or squash merge for pull requests

✅ **Pull Request Reviews**
- Structure in place for code review requirements
- Can be configured to require N approvals before merging
- Supports dismissing stale reviews
- Supports code owner reviews

## Working with Protected Branches

### Recommended Workflow

1. **Create Feature Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Commit**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   ```bash
   gh pr create --base main --head feature/your-feature-name \
     --title "Feature: Your Feature" \
     --body "Description of changes"
   ```

4. **Merge Pull Request**
   ```bash
   # After review and approval
   gh pr merge <PR-number> --squash
   ```

### Direct Push to Main

Direct pushes are allowed for fast-forward commits only:

```bash
git checkout main
git pull origin main
# Make changes
git add .
git commit -m "fix: urgent hotfix"
git push origin main  # Only works if fast-forward
```

⚠️ **Note:** Due to linear history requirement, you cannot push merge commits directly.

## Verification

Check current branch protection settings:

```bash
gh api repos/mohin-io/AI-Analytics-Platform/branches/main/protection
```

Or view in GitHub UI:
https://github.com/mohin-io/AI-Analytics-Platform/settings/branches

## Configuration History

- **2025-10-18**: Initial branch protection configured
  - Enabled force push prevention
  - Enabled deletion prevention
  - Enabled linear history requirement
  - Configured pull request review structure

## Modifying Protection Rules

To update branch protection settings, use the GitHub API:

```bash
gh api repos/mohin-io/AI-Analytics-Platform/branches/main/protection -X PUT \
  --input protection-config.json
```

Example `protection-config.json`:
```json
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

## Best Practices

1. **Always use feature branches** for development
2. **Create pull requests** for code review
3. **Write clear commit messages** following conventional commits
4. **Keep commits atomic** - one logical change per commit
5. **Rebase feature branches** on main before merging
6. **Use squash merge** to maintain clean history

## Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Git Workflow Best Practices](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)
