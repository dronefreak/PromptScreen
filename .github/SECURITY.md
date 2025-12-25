# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **kumaar324@gmail.com**

Include the following information:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

You should receive a response within **48 hours**. If for some reason you do not, please follow up via email to ensure we received your original message.

## Security Update Process

1. **Report received** - We acknowledge receipt within 48 hours
2. **Investigation** - We investigate and confirm the vulnerability
3. **Fix development** - We develop a fix in a private repository
4. **Release** - We release a patched version
5. **Disclosure** - We publicly disclose the vulnerability after the patch is released

## Security Best Practices for Users

### 1. Keep PromptScreen Updated

```bash
# Check current version
pip show promptscreen

# Upgrade to latest
pip install --upgrade promptscreen
```

### 2. Use Virtual Environments

Always install in isolated environments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install promptscreen
```

### 3. Validate User Input

When using PromptScreen in production:

```python
from promptscreen import HeuristicVectorAnalyzer, Scanner

# Use multiple guards for defense in depth
guards = [
    HeuristicVectorAnalyzer(threshold=2, pm_shot_lim=3),
    Scanner(),  # YARA rules
]

def validate_prompt(prompt: str) -> bool:
    """Validate prompt against all guards."""
    for guard in guards:
        result = guard.analyse(prompt)
        if not result.get_verdict():
            # Log blocked attempt
            print(f"Blocked: {result.get_type()}")
            return False
    return True
```

### 4. ML Model Security

**Note:** ML guards (ShieldGemma, ClassifierCluster) download models from HuggingFace Hub without revision pinning.

**Mitigation:**

- Only use models from trusted, verified sources
- Review model cards before deployment
- Consider pinning to specific model commits in production

**Future improvement:** We plan to add revision pinning in v0.2.0.

### 5. API Security

If using the FastAPI server:

```python
from promptscreen.api import create_app

app = create_app(guards)

# Add authentication middleware
# Add rate limiting
# Run behind reverse proxy (nginx/traefik)
```

**Recommendations:**

- Enable HTTPS in production
- Implement authentication (OAuth2, API keys)
- Add rate limiting to prevent abuse
- Monitor for unusual patterns

### 6. Dependency Security

We use:

- **Bandit** for security linting
- **Dependabot** (GitHub) for dependency updates
- **Pre-commit hooks** for code quality

Check for vulnerable dependencies:

```bash
pip install safety
safety check
```

## Known Security Considerations

### 1. Guard Bypass Risks

**No guard is 100% effective.** Attackers may:

- Use encoding tricks (ROT13, base64, unicode)
- Employ adversarial prompts designed to bypass detection
- Use context manipulation techniques

**Mitigation:** Use multiple guards in combination (defense in depth).

### 2. Model Poisoning

ML-based guards rely on pre-trained models that could be compromised at the source.

**Mitigation:**

- Only use models from verified HuggingFace authors
- Monitor model behavior for anomalies
- Consider hosting models locally in critical applications

### 3. Resource Exhaustion

Large prompts or many concurrent requests could exhaust system resources.

**Mitigation:**

- Implement prompt length limits
- Add request rate limiting
- Monitor resource usage
- Use timeouts

### 4. Data Privacy

Prompts analyzed by guards may contain sensitive information.

**Mitigation:**

- Do not log full prompts in production
- Implement data retention policies
- Consider local-only deployment for sensitive data
- Review privacy implications of ML model usage

## Security Tooling

We use the following tools to maintain security:

- **Bandit** - Python security linter
- **MyPy** - Static type checking (catches some bugs)
- **Ruff** - Fast linting with security rules
- **Pre-commit hooks** - Automated checks before commit
- **GitHub Actions** - CI/CD security scanning

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patched versions as quickly as possible

We aim to:

- Acknowledge reports within 48 hours
- Provide initial assessment within 1 week
- Release patches within 30 days for critical vulnerabilities

## Hall of Fame

We appreciate security researchers who responsibly disclose vulnerabilities:

_None yet - be the first!_

## Contact

Security contact: **kumaar324@gmail.com**

For general questions: Open a GitHub issue

## Attribution

This security policy is inspired by the [Contributor Covenant](https://www.contributor-covenant.org/) and follows industry best practices.

---

_Last updated: December 25, 2024_
