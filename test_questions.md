# Test Questions for /chat Endpoint

Use these to validate the RAG pipeline against the ingested insurance documents.

## Usage

```bash
curl -X POST https://<SERVICE_URL>/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "<QUESTION>"}'
```

---

## Q&A

### 1. Policy coverage limits (PLY-FL-001)

**Question:** What is the hurricane deductible for Maria Gonzalez's Florida homeowners policy?

**Expected Answer:** 2% of Coverage A dwelling limit, which is $7,000.

---

### 2. Claim details (CLM-FL-2024-001)

**Question:** What was the total estimated repair cost for the burst pipe claim CLM-FL-2024-001 and what caused the damage?

**Expected Answer:** $28,500. A copper supply line failed at a solder joint in the second-floor bathroom, causing water to flow for approximately 4 hours before discovery.

---

### 3. State regulation (REG-CA)

**Question:** Under California insurance regulations, how many days does an insurer have to accept or deny a claim after receiving proof of loss?

**Expected Answer:** 40 calendar days, per CCR § 2695.7(b). If more time is needed, the insurer must notify the claimant every 30 days.

---

### 4. Endorsement coverage (PLY-CA-002)

**Question:** Does James Chen's California policy cover mold remediation, and if so what is the limit?

**Expected Answer:** Yes, up to $5,000 per occurrence under endorsement HO-MOLD-CA-2024, but only for mold resulting from a covered peril (not long-term moisture or maintenance neglect).

---

### 5. Adjuster recommendation (NOTE-001)

**Question:** What was adjuster Thomas Rivera's recommendation on claim CLM-FL-2024-001, and were any red flags identified?

**Expected Answer:** Rivera recommended approving the claim. No red flags were identified. The damage was consistent with a sudden pipe burst (covered peril), and the mold was a direct result of the covered event, falling within the $10,000 mold endorsement limit.
