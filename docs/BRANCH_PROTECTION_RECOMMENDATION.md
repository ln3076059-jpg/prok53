# Main Branch Protection Recommendation

Status: **BLOCKED_BY_GITHUB_SETTINGS**.

GitHub's public branch metadata reported `main.protected=false` on 2026-09-02. Repository settings
cannot be changed by source-code commits, and this audit does not claim that protection was
enabled.

Configure the `main` ruleset in GitHub repository settings to:

- require a pull request before merge;
- require the `CI / verify` status check;
- require the branch to be up to date before merge;
- block force pushes and branch deletion;
- apply the rules to administrators, with emergency bypass recorded and narrowly assigned.

After enabling it, verify with GitHub branch metadata and retain a screenshot or exported ruleset
as repository-governance evidence. Until then, direct pushes remain technically possible even
though CI itself is implemented and passing.
