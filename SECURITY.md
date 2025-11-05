# Security Policy

## Supported Versions

We release patches to fix security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the IB Forex Trading Setup seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### **Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to our security team:

**Email:** security@quantinsti.com

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

### Information to Include

1. **Type of issue** (buffer overflow, SQL injection, cross-site scripting, etc.)
2. **Full paths of source file(s) related to the vulnerability**
3. **The location of the affected source code (tag/branch/commit or direct URL)**
4. **Any special configuration required to reproduce the issue**
5. **Step-by-step instructions to reproduce the issue**
6. **Proof-of-concept or exploit code (if possible)**
7. **Impact of the issue, including how an attacker might exploit it**

This information will help us triage your report more quickly.

### Preferred Languages

We prefer all communications to be in English.

## What to Expect

### When You Report a Vulnerability

1. **Acknowledgment**: You will receive an acknowledgment within 48 hours.
2. **Initial Assessment**: We will conduct an initial assessment of the report.
3. **Timeline**: We will provide a timeline for addressing the vulnerability.
4. **Updates**: You will receive regular updates on our progress.

### Our Response Process

1. **Confirmation**: We will confirm receipt of your vulnerability report.
2. **Investigation**: Our security team will investigate the reported issue.
3. **Assessment**: We will assess the severity and impact of the vulnerability.
4. **Fix Development**: If confirmed, we will develop a fix.
5. **Testing**: The fix will be thoroughly tested.
6. **Release**: A security update will be released.
7. **Disclosure**: We will publicly disclose the vulnerability (with appropriate details).

### Disclosure Policy

When we receive a security bug report, we will:

1. **Acknowledge** the receipt of the vulnerability report.
2. **Investigate** the issue and determine its severity.
3. **Fix** the issue and test the fix.
4. **Release** a security update.
5. **Disclose** the vulnerability publicly.

### Timeline

- **Critical vulnerabilities**: 7 days or less
- **High severity vulnerabilities**: 14 days or less
- **Medium severity vulnerabilities**: 30 days or less
- **Low severity vulnerabilities**: 90 days or less

## Security Best Practices

### For You

1. **You should keep your installation updated** to the latest version.
2. **You should use strong, unique passwords** for your Interactive Brokers account.
3. **You should enable two-factor authentication** on your IB account.
4. **You should never share your API credentials** or account information.
5. **You should use a dedicated trading account** with limited funds for testing.
6. **You should monitor your account activity** regularly.
7. **You should use secure connections** (HTTPS) when accessing your account.

### For Developers

1. **You should never commit sensitive information** (API keys, passwords, etc.) to version control.
2. **You should use environment variables** for configuration.
3. **You should validate all inputs** from external sources.
4. **You should use parameterized queries** to prevent SQL injection.
5. **You should implement proper error handling** without exposing sensitive information.
6. **You should keep dependencies updated** to patch known vulnerabilities.
7. **You should use HTTPS** for all external API calls.

### For Contributors

1. **You should review code changes** for potential security issues.
2. **You should test your changes** thoroughly before submitting.
3. **You should follow secure coding practices** and guidelines.
4. **You should report any security concerns** immediately.
5. **You should not include test credentials** in your contributions.

## Security Features

The IB Forex Trading Setup includes several security features:

1. **Input Validation**: All your inputs are validated to prevent injection attacks.
2. **Error Handling**: Comprehensive error handling without exposing sensitive information.
3. **Secure Connections**: It uses secure connections for all API communications.
4. **Configuration Management**: Secure handling of configuration and credentials.
5. **Logging**: Comprehensive logging for security monitoring.

## Security Updates

Security updates will be released as patch versions (e.g., 1.0.1, 1.0.2) and will be clearly marked as security updates in the changelog.

## Contact Information

For security-related questions or concerns:

- **Security Team**: security@quantinsti.com
- **General Support**: support@quantinsti.com
- **Emergency Contact**: +1-XXX-XXX-XXXX (for critical issues only)

## Acknowledgments

We would like to thank all security researchers and users who responsibly report vulnerabilities to us. Your contributions help make the IB Forex Trading Setup more secure for everyone.

## Legal

By reporting a vulnerability, you agree that:

1. You will not publicly disclose the vulnerability until we have had a reasonable time to address it.
2. You will not use the vulnerability for malicious purposes.
3. You will not attempt to access or modify data without authorization.
4. You will comply with all applicable laws and regulations.

Thank you for helping keep the IB Forex Trading Setup secure!