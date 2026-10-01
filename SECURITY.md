# Security Policy

## Supported Versions

Only the latest release on the `main` branch is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Prohibited Data & Secrets Policy

To preserve academic integrity, personal privacy, and infrastructure security, contributors and users must **NEVER** commit, upload, or expose the following items in this repository:

1. **Credentials & Secrets:**
   - Passwords, passphrases, or private keys
   - Personal Access Tokens (PATs) or OAuth tokens
   - API keys, cloud provider secrets, or service account credentials
   - GitHub credentials or CI/CD secret values

2. **Personally Identifiable Information (PII):**
   - Real student names, email addresses, phone numbers, or residential addresses
   - Permanent Registration Numbers (PRNs), roll numbers, or government ID numbers (except the author's public academic PRN in credit blocks)
   - Disciplinary records or non-anonymized academic transcripts

3. **Confidential Environments:**
   - Local `.env` files containing environment secrets
   - `.streamlit/secrets.toml` with private server configurations

---

## Reporting a Vulnerability

If you discover a security vulnerability or accidental credential leakage within this repository, please do **NOT** open a public GitHub issue.

Instead, please report the vulnerability privately:

* **Maintainer:** Bhedheer Bhushan Jain (PRN: 25030422033)
* **Email:** `bhedheerbhushanjain@gmail.com`
* **Subject Line:** `[SECURITY] Student Performance Dashboard Vulnerability Report`

Please include:
- A description of the vulnerability and its potential impact
- Detailed steps to reproduce the issue or proof-of-concept
- Any remediation suggestions

We will acknowledge receipt within 48 hours and coordinate a coordinated fix and security disclosure.
