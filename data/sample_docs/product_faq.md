# Acme Platform — Technical FAQ

## Authentication & Security

**Q: What authentication methods does Acme Platform support?**

Acme Platform supports OAuth 2.0, SAML 2.0, and API key-based authentication. For enterprise customers, SSO integration via SAML 2.0 with identity providers such as Okta, Azure AD, and Google Workspace is available. Multi-factor authentication (MFA) is enforced by default for all admin accounts and can be enforced organisation-wide via the security settings panel.

**Q: How are API keys managed?**

API keys can be generated from the Developer Settings page. Each key can be scoped to specific permissions (read-only, read-write, admin). Keys should be rotated every 90 days as a security best practice. Compromised keys can be revoked immediately from the dashboard. All API key usage is logged and available in the Audit Log.

**Q: What encryption standards are used?**

All data in transit is encrypted using TLS 1.3. Data at rest is encrypted using AES-256. Database backups are encrypted using customer-managed keys (CMK) for Enterprise plan customers. The platform is SOC 2 Type II and ISO 27001 certified.

## Performance & Scalability

**Q: What are the rate limits?**

Rate limits vary by plan:
- Starter: 100 API requests/minute
- Professional: 1,000 API requests/minute
- Enterprise: 10,000 API requests/minute (custom limits available)

Rate limit headers are included in all API responses: X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset.

**Q: What is the SLA for uptime?**

Acme Platform guarantees 99.9% uptime for Professional plans and 99.99% uptime for Enterprise plans, measured monthly. Planned maintenance windows are announced 72 hours in advance and scheduled during low-traffic hours (2–4 AM UTC on Sundays). Compensation for SLA breaches is provided as service credits.

**Q: How does the platform handle large file uploads?**

Files up to 100MB can be uploaded directly. Files between 100MB and 5GB must use the multipart upload API, which supports resumable uploads. Files over 5GB require the chunked upload endpoint. All uploads are virus-scanned before processing. Supported formats include PDF, DOCX, XLSX, CSV, JSON, and most common image formats.

## Billing & Subscriptions

**Q: What happens if I exceed my plan limits?**

For API calls, requests exceeding the rate limit will receive a 429 Too Many Requests response. Storage overages are billed at $0.10/GB/month for Starter and Professional plans. Enterprise plans have custom overage pricing negotiated at contract time. You will receive email alerts at 80% and 95% of your storage limit.

**Q: How do I cancel my subscription?**

Subscriptions can be cancelled from the Billing section of the Account Settings page. Cancellations take effect at the end of the current billing period. No partial refunds are issued for the remaining period. Data is retained for 30 days after cancellation, after which it is permanently deleted. You can export all your data before cancellation using the Data Export tool.

**Q: What payment methods are accepted?**

Visa, Mastercard, American Express, and PayPal are accepted for Starter and Professional plans. Enterprise customers may be invoiced monthly or annually (annual invoicing requires a minimum 12-month contract). Wire transfers are available for annual contracts over $10,000.

## Data & Privacy

**Q: Where is data stored?**

By default, all data is stored in US-East (AWS us-east-1). Enterprise customers can request data residency in EU (Frankfurt, eu-central-1) or APAC (Singapore, ap-southeast-1) at no additional cost. Data residency requests must be made before onboarding, as migrating existing data between regions requires a service window.

**Q: How long is data retained?**

Active account data is retained indefinitely while the subscription is active. Deleted data is purged within 30 days. Audit logs are retained for 1 year on Professional plans and 7 years on Enterprise plans (to meet compliance requirements). Backups are retained for 30 days.

**Q: Is Acme Platform GDPR compliant?**

Yes. Acme Corp acts as a Data Processor under GDPR. A Data Processing Agreement (DPA) is available for all customers and is automatically included in Enterprise contracts. Right to erasure requests are processed within 30 days. Data portability exports are available in JSON and CSV formats. The Data Protection Officer can be contacted at dpo@acmecorp.com.

## Integrations

**Q: What third-party integrations are available?**

Native integrations are available for: Slack, Microsoft Teams, Salesforce, HubSpot, Jira, Confluence, GitHub, GitLab, Zapier, and Webhook-based custom integrations. An iPaaS integration via Make (formerly Integromat) and Workato is also available for Enterprise customers. The full list of integrations is available at integrations.acmecorp.com.

**Q: How do webhooks work?**

Webhooks can be configured from the Integrations page. You can subscribe to specific event types (e.g., document.created, user.invited, payment.failed). Webhook payloads are signed with HMAC-SHA256 using your webhook secret. Failed deliveries are retried up to 5 times with exponential backoff. The webhook delivery log is available for the past 7 days.
