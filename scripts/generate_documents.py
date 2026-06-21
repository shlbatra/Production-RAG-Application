"""Generate realistic insurance documents for RAG pipeline testing."""

import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
DIRS = ["policies", "claims", "regulations", "adjuster_notes"]

metadata = {}


def register(filename: str, **meta):
    metadata[filename] = meta


# ── Policies ───────────────────────────────────────────────────────────────

POLICIES = {
    "PLY-FL-001": {
        "state": "FL",
        "insured": "Maria Gonzalez",
        "address": "4521 Palm Beach Blvd, Fort Lauderdale, FL 33301",
        "effective": "01/01/2024",
        "expiry": "01/01/2025",
        "coverage_a": 350000,
        "coverage_b": 35000,
        "coverage_c": 175000,
        "coverage_d": 70000,
        "deductible": 2500,
        "hurricane_deductible": "2% of Coverage A ($7,000)",
        "mold_limit": 10000,
        "water_backup_limit": 5000,
        "state_specific": """FLORIDA-SPECIFIC PROVISIONS:
- Hurricane deductible: 2% of Coverage A dwelling limit ($7,000)
- Sinkhole coverage: Available by endorsement (not included)
- Assignment of Benefits (AOB): Subject to FL § 627.7152 restrictions
- Managed repair program: Policyholder may choose contractor from approved list""",
    },
    "PLY-CA-002": {
        "state": "CA",
        "insured": "James Chen",
        "address": "892 Hillside Dr, Los Angeles, CA 90046",
        "effective": "03/15/2024",
        "expiry": "03/15/2025",
        "coverage_a": 750000,
        "coverage_b": 75000,
        "coverage_c": 375000,
        "coverage_d": 150000,
        "deductible": 5000,
        "hurricane_deductible": "N/A",
        "mold_limit": 5000,
        "water_backup_limit": 10000,
        "state_specific": """CALIFORNIA-SPECIFIC PROVISIONS:
- Earthquake coverage: EXCLUDED. Available separately through CEA (California Earthquake Authority)
- Brush fire zone: Property is in a designated wildfire risk area
- Fair Plan: This policy is NOT a FAIR Plan policy
- CA § 790.03: Unfair claims settlement practices apply
- Mandatory offer of earthquake coverage was made and declined by insured""",
    },
    "PLY-TX-003": {
        "state": "TX",
        "insured": "Robert Williams",
        "address": "1205 Lone Star Pkwy, Houston, TX 77002",
        "effective": "06/01/2024",
        "expiry": "06/01/2025",
        "coverage_a": 425000,
        "coverage_b": 42500,
        "coverage_c": 212500,
        "coverage_d": 85000,
        "deductible": 3500,
        "hurricane_deductible": "N/A",
        "mold_limit": 0,
        "water_backup_limit": 5000,
        "state_specific": """TEXAS-SPECIFIC PROVISIONS:
- Wind/hail deductible: 1% of Coverage A ($4,250)
- Mold coverage: NOT INCLUDED. Mold remediation endorsement not purchased.
- TX Insurance Code § 542.056: Insurer must accept or reject claim within 15 business days
- Texas Windstorm Insurance Association (TWIA): This policy is NOT a TWIA policy
- Foundation coverage: Included for sudden and accidental foundation damage""",
    },
    "PLY-NY-004": {
        "state": "NY",
        "insured": "Sarah Patel",
        "address": "315 Park Ave, Brooklyn, NY 11205",
        "effective": "09/01/2024",
        "expiry": "09/01/2025",
        "coverage_a": 550000,
        "coverage_b": 55000,
        "coverage_c": 275000,
        "coverage_d": 110000,
        "deductible": 2500,
        "hurricane_deductible": "5% of Coverage A ($27,500) for named storms",
        "mold_limit": 15000,
        "water_backup_limit": 10000,
        "state_specific": """NEW YORK-SPECIFIC PROVISIONS:
- Named storm deductible: 5% of Coverage A ($27,500)
- NY Insurance Law § 3408: Prompt payment requirements
- Exterior maintenance requirement: Insured must maintain exterior to prevent water intrusion
- NY Regulation 64: Minimum standards for claim settlement
- Water damage sublimit for units below grade: $25,000""",
    },
    "PLY-IL-005": {
        "state": "IL",
        "insured": "David Thompson",
        "address": "780 Michigan Ave, Chicago, IL 60611",
        "effective": "02/15/2024",
        "expiry": "02/15/2025",
        "coverage_a": 480000,
        "coverage_b": 48000,
        "coverage_c": 240000,
        "coverage_d": 96000,
        "deductible": 2000,
        "hurricane_deductible": "N/A",
        "mold_limit": 10000,
        "water_backup_limit": 7500,
        "state_specific": """ILLINOIS-SPECIFIC PROVISIONS:
- 215 ILCS 5/154.6: Claims must be paid within 30 days of proof of loss
- Frozen pipe coverage: Included (common winter peril)
- Ice dam coverage: Covered under water damage provisions
- Water backup of sewers and drains endorsement: INCLUDED ($7,500 limit)
- Wind/hail: Standard deductible applies (no separate wind deductible)""",
    },
}

SEP = "=" * 70
DASH = "─" * 70


def generate_policy(policy_id: str, p: dict) -> str:
    mold_section = (
        f"""
MOLD REMEDIATION ENDORSEMENT
Endorsement Number: HO-MOLD-{p['state']}-2024
We will pay up to ${p['mold_limit']:,} per occurrence for the cost of mold
remediation that results from a covered peril. This includes the cost of
testing, removal, and restoration necessitated by mold resulting from
a covered water damage event. This endorsement does not cover mold
resulting from long-term moisture, condensation, or maintenance neglect.
Mold resulting from flood is excluded per the flood exclusion.
"""
        if p["mold_limit"] > 0
        else """
MOLD REMEDIATION: NOT INCLUDED
The insured has not purchased the optional mold remediation endorsement.
Any mold damage, testing, removal, or remediation is excluded from coverage
under this policy regardless of cause.
"""
    )

    return f"""{SEP}
HOMEOWNERS POLICY — HO-3 SPECIAL FORM
{SEP}

POLICY NUMBER: {policy_id}
CARRIER: Guidewire Mutual Insurance Company
CARRIER ID: GWMI-001

DECLARATIONS PAGE
{DASH}
Named Insured:     {p['insured']}
Property Address:  {p['address']}
Policy Period:     {p['effective']} to {p['expiry']}
Form Type:         HO-3 (Special Form — Open Perils on Dwelling)
Line of Business:  Homeowners
State:             {p['state']}

COVERAGE SUMMARY
{DASH}
Coverage A — Dwelling:                      ${p['coverage_a']:,}
Coverage B — Other Structures:              ${p['coverage_b']:,}
Coverage C — Personal Property:             ${p['coverage_c']:,}
Coverage D — Loss of Use:                   ${p['coverage_d']:,}
All-Peril Deductible:                       ${p['deductible']:,}
Hurricane/Named Storm Deductible:           {p['hurricane_deductible']}

{SEP}
SECTION I — COVERAGES
{SEP}

COVERAGE A — DWELLING
We cover the dwelling on the residence premises shown in the Declarations,
including structures attached to the dwelling. We also cover materials and
supplies located on or next to the residence premises used to construct,
alter, or repair the dwelling. Coverage A limit: ${p['coverage_a']:,}.

COVERAGE B — OTHER STRUCTURES
We cover other structures on the residence premises set apart from the
dwelling by clear space, including structures connected to the dwelling
only by a fence, utility line, or similar connection. Limit: ${p['coverage_b']:,}.

COVERAGE C — PERSONAL PROPERTY
We cover personal property owned or used by an insured while it is anywhere
in the world. At your request, we will cover personal property owned by
others while the property is on the part of the residence premises occupied
by an insured. Limit: ${p['coverage_c']:,}.

COVERAGE D — LOSS OF USE
If a covered peril makes the residence premises not fit to live in, we
cover additional living expenses incurred by you so that your household
can maintain its normal standard of living. Limit: ${p['coverage_d']:,}.

{SEP}
SECTION I — EXCLUSIONS
{SEP}

We do not insure for loss caused directly or indirectly by any of the
following. Such loss is excluded regardless of any other cause or event
contributing concurrently or in any sequence to the loss.

FLOOD EXCLUSION
Flood, surface water, waves, tidal water, overflow of a body of water,
or spray from any of these, whether or not driven by wind, is EXCLUDED.
This includes flood damage caused by hurricane storm surge. Flood
insurance is available separately through the National Flood Insurance
Program (NFIP) or private flood insurers.

EARTHQUAKE EXCLUSION
Earthquake, including land shock waves or tremors before, during, or
after a volcanic eruption, is EXCLUDED. Earth movement of any kind,
including but not limited to earthquake, landslide, mudflow, mudslide,
sinkhole, subsidence, or erosion is excluded.

WEAR AND TEAR EXCLUSION
Wear and tear, marring, deterioration, inherent vice, latent defect,
mechanical breakdown, rust, mold (except as provided by endorsement),
wet or dry rot is EXCLUDED.

INTENTIONAL LOSS
Any loss arising out of any act an insured commits or conspires to
commit with the intent to cause a loss is EXCLUDED.

NEGLECT
Neglect meaning neglect of an insured to use all reasonable means to
save and preserve property at and after the time of a loss is EXCLUDED.

ORDINANCE OR LAW
The enforcement of any ordinance or law regulating the construction,
repair, or demolition of a building or other structure, unless
specifically provided under this policy, is EXCLUDED.

{SEP}
ENDORSEMENTS
{SEP}
{mold_section}
WATER BACKUP AND SUMP DISCHARGE/OVERFLOW ENDORSEMENT
Endorsement Number: HO-WBSD-{p['state']}-2024
We will pay up to ${p['water_backup_limit']:,} for direct physical loss to
covered property caused by water which backs up through sewers or drains,
or which overflows or is discharged from a sump, sump pump, or related
equipment. This endorsement does not cover flood as defined in the
flood exclusion above.

{SEP}
{p['state_specific']}
{SEP}
END OF POLICY — {policy_id}
"""


# ── Claims ─────────────────────────────────────────────────────────────────

CLAIMS = [
    {
        "id": "CLM-FL-2024-001",
        "policy": "PLY-FL-001",
        "state": "FL",
        "date_of_loss": "03/15/2024",
        "peril": "water_damage",
        "reported": "03/16/2024",
        "estimated_amount": 28500,
        "description": """CLAIM REPORT — WATER DAMAGE (BURST PIPE)

Claim ID: CLM-FL-2024-001
Policy: PLY-FL-001
Date of Loss: 03/15/2024
Date Reported: 03/16/2024
Peril: Water Damage — Burst Pipe

LOSS DESCRIPTION:
Insured Maria Gonzalez reported a burst pipe in the second-floor bathroom
on the evening of March 15, 2024. The pipe (copper supply line) failed at
a joint connection, causing water to flow for approximately 4 hours before
discovery. Water damage affected the second-floor bathroom, hallway, and
the first-floor ceiling and living room below.

DAMAGE DETAILS:
- Second floor bathroom: destroyed drywall, warped subfloor, damaged vanity
- Hallway: water-stained carpet (15 ft), baseboards buckled
- First floor ceiling: collapsed drywall section (8x10 ft area)
- First floor living room: water-damaged hardwood flooring (200 sq ft)
- Mold discovered behind bathroom wall during inspection (estimated 30 sq ft)

ESTIMATED REPAIR COST: $28,500
- Plumbing repair: $1,200
- Drywall and ceiling: $8,500
- Flooring replacement: $9,800
- Mold remediation: $4,500
- Contents damage: $4,500

ADJUSTER ASSIGNED: Thomas Rivera
STATUS: Under Investigation""",
    },
    {
        "id": "CLM-FL-2024-002",
        "policy": "PLY-FL-001",
        "state": "FL",
        "date_of_loss": "06/10/2024",
        "peril": "wind",
        "reported": "06/11/2024",
        "estimated_amount": 15200,
        "description": """CLAIM REPORT — WIND DAMAGE

Claim ID: CLM-FL-2024-002
Policy: PLY-FL-001
Date of Loss: 06/10/2024
Date Reported: 06/11/2024
Peril: Wind Damage — Tropical Storm

LOSS DESCRIPTION:
Tropical Storm Alberto caused wind damage to the insured property on
June 10, 2024. Sustained winds of approximately 55 mph with gusts to
70 mph impacted the Fort Lauderdale area. Named storm was declared by
the National Hurricane Center.

DAMAGE DETAILS:
- Roof: 12 shingles displaced, 3 sections of ridge cap damaged
- Soffit: detached along north-facing eave (20 linear feet)
- Fence: wooden privacy fence (40 ft section) blown down
- Screen enclosure: aluminum frame bent, 4 panels torn
- Tree limb fell on detached garage (minor roof dent)

ESTIMATED REPAIR COST: $15,200
- Roof repairs: $6,500
- Soffit repair: $2,200
- Fence replacement: $3,500
- Screen enclosure: $2,000
- Garage repair: $1,000

NOTE: This is a named storm event. Hurricane deductible of 2% of
Coverage A ($7,000) applies per FL policy provisions.

ADJUSTER ASSIGNED: Thomas Rivera
STATUS: Open""",
    },
    {
        "id": "CLM-CA-2024-003",
        "policy": "PLY-CA-002",
        "state": "CA",
        "date_of_loss": "04/22/2024",
        "peril": "fire",
        "reported": "04/22/2024",
        "estimated_amount": 185000,
        "description": """CLAIM REPORT — FIRE DAMAGE

Claim ID: CLM-CA-2024-003
Policy: PLY-CA-002
Date of Loss: 04/22/2024
Date Reported: 04/22/2024
Peril: Fire — Kitchen Origin

LOSS DESCRIPTION:
Fire originated in the kitchen at approximately 6:30 PM on April 22, 2024.
The Los Angeles Fire Department responded and extinguished the fire.
Cause determined to be an unattended cooking fire (grease fire from stovetop).
Fire spread from kitchen to adjacent dining room before suppression.

DAMAGE DETAILS:
- Kitchen: total loss — cabinets, countertops, appliances destroyed
- Dining room: severe smoke and heat damage, partial wall destruction
- Living room: heavy smoke damage throughout
- Second floor: moderate smoke damage to all rooms
- Contents: significant loss of kitchen items, dining furniture
- Smoke remediation needed throughout entire dwelling

ESTIMATED REPAIR COST: $185,000
- Kitchen rebuild: $75,000
- Dining room restoration: $35,000
- Smoke remediation: $25,000
- Living room restoration: $20,000
- Contents replacement: $30,000

ADJUSTER ASSIGNED: Linda Park
STATUS: Under Investigation""",
    },
    {
        "id": "CLM-CA-2024-004",
        "policy": "PLY-CA-002",
        "state": "CA",
        "date_of_loss": "07/30/2024",
        "peril": "theft",
        "reported": "07/31/2024",
        "estimated_amount": 22000,
        "description": """CLAIM REPORT — THEFT / BURGLARY

Claim ID: CLM-CA-2024-004
Policy: PLY-CA-002
Date of Loss: 07/30/2024
Date Reported: 07/31/2024
Peril: Theft — Residential Burglary

LOSS DESCRIPTION:
Insured James Chen returned from a weekend trip on July 31, 2024 to find
the residence had been burglarized. Entry was made through a rear sliding
door (forced open). LAPD report filed (Report #LA-2024-0730-4521).

STOLEN ITEMS:
- 65" Samsung TV: $2,200
- MacBook Pro laptop: $3,500
- Jewelry (itemized list provided): $8,500
- Camera equipment (Canon EOS R5 + lenses): $5,800
- Cash: $500 (subject to $200 cash limit per policy)
- Miscellaneous electronics: $1,500

PROPERTY DAMAGE:
- Rear sliding door: frame damaged, glass cracked
- Interior: ransacked, minor damage to furniture

ESTIMATED TOTAL: $22,000 (subject to Coverage C limit and sublimits)

NOTE: Cash claim limited to $200 per policy terms. Jewelry claim may
be subject to $2,500 sublimit unless scheduled.

ADJUSTER ASSIGNED: Linda Park
STATUS: Open""",
    },
    {
        "id": "CLM-TX-2024-005",
        "policy": "PLY-TX-003",
        "state": "TX",
        "date_of_loss": "05/18/2024",
        "peril": "water_damage",
        "reported": "05/19/2024",
        "estimated_amount": 42000,
        "description": """CLAIM REPORT — WATER DAMAGE (PIPE BURST)

Claim ID: CLM-TX-2024-005
Policy: PLY-TX-003
Date of Loss: 05/18/2024
Date Reported: 05/19/2024
Peril: Water Damage — Supply Line Failure

LOSS DESCRIPTION:
A hot water supply line in the master bathroom wall failed on May 18, 2024.
The insured was away at work and the leak continued for approximately
8 hours. Extensive water damage to the master suite, hallway, and guest
bedroom on the first floor. Water penetrated to the foundation slab.

DAMAGE DETAILS:
- Master bathroom: destroyed — drywall, tile, vanity, fixtures
- Master bedroom: flooring ruined (hardwood, 300 sq ft), baseboards
- Hallway: carpet and pad saturated, drywall damage
- Guest bedroom: carpet damage, lower 2 ft of drywall affected
- Foundation: possible moisture under slab (engineering assessment needed)
- Mold growth detected in wall cavity during demolition

ESTIMATED REPAIR COST: $42,000
- Plumbing repair: $2,500
- Master bath rebuild: $15,000
- Bedroom restoration: $12,000
- Hallway restoration: $5,000
- Foundation assessment and repair: $4,500
- Mold remediation: $3,000

IMPORTANT NOTE: Policy PLY-TX-003 does NOT include mold remediation
endorsement. Mold remediation costs ($3,000) may not be covered.

ADJUSTER ASSIGNED: Michael Torres
STATUS: Under Investigation""",
    },
    {
        "id": "CLM-TX-2024-006",
        "policy": "PLY-TX-003",
        "state": "TX",
        "date_of_loss": "08/05/2024",
        "peril": "wind",
        "reported": "08/05/2024",
        "estimated_amount": 31500,
        "description": """CLAIM REPORT — WIND/HAIL DAMAGE

Claim ID: CLM-TX-2024-006
Policy: PLY-TX-003
Date of Loss: 08/05/2024
Date Reported: 08/05/2024
Peril: Wind and Hail — Severe Thunderstorm

LOSS DESCRIPTION:
A severe thunderstorm with large hail (golf ball size, 1.75" diameter)
and straight-line winds of 65+ mph struck the Houston area on August 5.
Multiple neighboring properties also sustained damage.

DAMAGE DETAILS:
- Roof: extensive hail damage, 47 impacts per 10x10 test square
- Siding: 15 panels cracked/broken from hail impact (vinyl)
- Windows: 2 windows cracked from hail
- AC condenser unit: fins damaged, unit may need replacement
- Gutters: dented and pulled away from fascia (60 linear feet)
- Vehicle damage: reported separately under auto policy

ESTIMATED REPAIR COST: $31,500
- Full roof replacement: $18,000
- Siding replacement: $6,500
- Window replacement: $3,000
- AC condenser: $2,500
- Gutters: $1,500

NOTE: Wind/hail deductible of 1% of Coverage A ($4,250) applies per
TX policy provisions.

ADJUSTER ASSIGNED: Michael Torres
STATUS: Open""",
    },
    {
        "id": "CLM-NY-2024-007",
        "policy": "PLY-NY-004",
        "state": "NY",
        "date_of_loss": "10/12/2024",
        "peril": "water_damage",
        "reported": "10/12/2024",
        "estimated_amount": 18500,
        "description": """CLAIM REPORT — WATER DAMAGE (BACKUP)

Claim ID: CLM-NY-2024-007
Policy: PLY-NY-004
Date of Loss: 10/12/2024
Date Reported: 10/12/2024
Peril: Water Backup — Sewer/Drain

LOSS DESCRIPTION:
Heavy rainfall on October 12, 2024 caused the building's main sewer line
to back up. Sewage water entered the insured's basement unit through floor
drains. The basement level (below grade) sustained significant water damage
from approximately 6 inches of standing sewage water.

DAMAGE DETAILS:
- Basement flooring: laminate destroyed throughout (400 sq ft)
- Drywall: lower 3 ft damaged throughout basement (contaminated)
- Stored personal property: boxes of clothing, books, electronics damaged
- Furniture: couch, bookshelf, desk damaged beyond repair
- Sanitation: professional biohazard cleanup required

ESTIMATED REPAIR COST: $18,500
- Biohazard cleanup/sanitation: $4,500
- Flooring replacement: $5,000
- Drywall replacement: $4,000
- Contents loss: $5,000

NOTE: Water backup endorsement applies ($10,000 limit). Additionally,
below-grade sublimit of $25,000 may apply per NY policy provisions.
Claim amount ($18,500) is within both limits.

ADJUSTER ASSIGNED: Jennifer Walsh
STATUS: Open""",
    },
    {
        "id": "CLM-NY-2024-008",
        "policy": "PLY-NY-004",
        "state": "NY",
        "date_of_loss": "11/28/2024",
        "peril": "theft",
        "reported": "11/29/2024",
        "estimated_amount": 12800,
        "description": """CLAIM REPORT — THEFT / PACKAGE THEFT + BREAK-IN

Claim ID: CLM-NY-2024-008
Policy: PLY-NY-004
Date of Loss: 11/28/2024
Date Reported: 11/29/2024
Peril: Theft — Break-in

LOSS DESCRIPTION:
The insured's apartment was broken into while the insured was traveling
for Thanksgiving. Entry was gained through a fire escape window. NYPD
report filed (Report #NY-2024-1128-8823).

STOLEN ITEMS:
- Laptop (Dell XPS 15): $1,800
- iPad Pro: $1,200
- Watches (2): $4,500
- Designer handbag: $2,300
- Small electronics/chargers: $800
- Coat collection (3 designer coats): $2,200

PROPERTY DAMAGE:
- Window lock: forced open, frame damaged
- Bedroom door: kicked in

ESTIMATED TOTAL: $12,800

ADJUSTER ASSIGNED: Jennifer Walsh
STATUS: Under Investigation""",
    },
    {
        "id": "CLM-IL-2024-009",
        "policy": "PLY-IL-005",
        "state": "IL",
        "date_of_loss": "01/22/2024",
        "peril": "fire",
        "reported": "01/22/2024",
        "estimated_amount": 95000,
        "description": """CLAIM REPORT — FIRE DAMAGE

Claim ID: CLM-IL-2024-009
Policy: PLY-IL-005
Date of Loss: 01/22/2024
Date Reported: 01/22/2024
Peril: Fire — Electrical Origin

LOSS DESCRIPTION:
Fire originated from a faulty electrical outlet in the second-floor
bedroom at approximately 2:15 AM on January 22, 2024. Chicago Fire
Department responded. Fire was contained to the second floor but smoke
damage affected the entire dwelling. The insured and family evacuated
safely. The home is temporarily uninhabitable.

DAMAGE DETAILS:
- Second floor bedroom (origin): total loss
- Second floor hallway: severe fire and smoke damage
- Second floor bathroom: heat and smoke damage
- First floor: heavy smoke damage throughout all rooms
- Attic: heat damage to insulation and roof deck
- Contents: significant loss on second floor
- Temporary housing needed: estimated 3 months

ESTIMATED REPAIR COST: $95,000
- Second floor rebuild: $45,000
- First floor smoke remediation and repainting: $15,000
- Attic repairs: $8,000
- Contents replacement: $17,000
- Loss of use (3 months temporary housing): $10,000

ADJUSTER ASSIGNED: Kevin O'Brien
STATUS: Under Investigation""",
    },
    {
        "id": "CLM-IL-2024-010",
        "policy": "PLY-IL-005",
        "state": "IL",
        "date_of_loss": "12/05/2024",
        "peril": "liability",
        "reported": "12/06/2024",
        "estimated_amount": 35000,
        "description": """CLAIM REPORT — LIABILITY

Claim ID: CLM-IL-2024-010
Policy: PLY-IL-005
Date of Loss: 12/05/2024
Date Reported: 12/06/2024
Peril: Liability — Slip and Fall

LOSS DESCRIPTION:
A visitor (neighbor Margaret Foster, age 67) slipped on an icy patch on
the insured's front walkway on December 5, 2024. The visitor sustained
a fractured wrist and bruised hip. She was transported to Northwestern
Memorial Hospital. The insured reports that he had salted the walkway
that morning but freezing rain occurred in the afternoon.

INJURY DETAILS:
- Fractured right wrist (distal radius fracture)
- Bruised left hip
- Emergency room visit and X-rays
- Orthopedic follow-up required, possible surgery for wrist
- Estimated 6-8 weeks recovery

ESTIMATED COSTS:
- Medical expenses (current): $8,500
- Projected medical (surgery if needed): $15,000
- Pain and suffering: $10,000
- Lost wages (visitor is a part-time consultant): $1,500

ESTIMATED TOTAL: $35,000

NOTE: This is a liability claim under Section II of the policy.
Coverage E (Personal Liability) limit: $100,000.

ADJUSTER ASSIGNED: Kevin O'Brien
STATUS: Under Investigation""",
    },
]


# ── Regulations ────────────────────────────────────────────────────────────

REGULATIONS = {
    "REG-FL": {
        "state": "FL",
        "content": """STATE OF FLORIDA — INSURANCE CLAIMS REGULATIONS SUMMARY
============================================================

APPLICABLE STATUTES:
Florida Statutes Title XXXVII — Insurance, Chapter 627

CLAIM ACKNOWLEDGMENT:
Per FL § 627.426(2), the insurer must acknowledge receipt of a claim
communication within 14 calendar days.

CLAIM INVESTIGATION:
Per FL § 627.426(3), the insurer must begin investigation of a claim
within 10 business days of receiving proof of loss.

CLAIM DECISION:
Per FL § 627.426(4), the insurer must pay or deny a claim within
90 calendar days after receiving the proof of loss, unless the failure
to pay is caused by factors beyond the insurer's control.

PROMPT PAYMENT:
Per FL § 627.4265, when an insurer makes a payment on a claim, it must
be paid within 20 calendar days after the date the insurer received
the proof of loss.

BAD FAITH:
Per FL § 624.155, an insured may bring a civil action against an insurer
for bad faith failure to settle claims. The insured must file a Civil
Remedy Notice with the Department of Financial Services and allow
60 days for the insurer to cure the violation.

HURRICANE CLAIMS:
Per FL § 627.70132, for claims arising from a hurricane:
- Insurer must provide acknowledgment within 14 days
- Must begin investigation within 7 days of assignment
- Must provide written decision within 90 days

ASSIGNMENT OF BENEFITS:
Per FL § 627.7152, an AOB agreement must be in writing, may not exceed
the coverage limits, and the assignee must provide a copy to the insurer
within 3 business days.

MEDIATION:
Per FL § 627.7015, either party may request mediation through the
Department of Financial Services for disputed claims.
""",
    },
    "REG-CA": {
        "state": "CA",
        "content": """STATE OF CALIFORNIA — INSURANCE CLAIMS REGULATIONS SUMMARY
============================================================

APPLICABLE STATUTES:
California Insurance Code and California Code of Regulations Title 10

CLAIM ACKNOWLEDGMENT:
Per CA Insurance Code § 790.03(h) and CCR § 2695.5(e), the insurer must
acknowledge receipt of a claim within 15 calendar days.

CLAIM INVESTIGATION:
Per CCR § 2695.7(b), the insurer must accept or deny the claim within
40 calendar days after receiving proof of loss. If more time is needed,
the insurer must notify the claimant every 30 days of the status.

CLAIM DECISION:
Per CCR § 2695.7(b), the insurer must accept or deny the claim, in
whole or in part, within 40 calendar days of receiving proof of claim.

PROMPT PAYMENT:
Per CA Insurance Code § 790.03(h)(5), once a claim is accepted, payment
must be made within 30 calendar days.

BAD FAITH:
Per CA Civil Code § 3294, punitive damages may be available for bad faith
claims handling. The insured may also bring action under CA Insurance
Code § 790.03 for unfair claims settlement practices.

WILDFIRE CLAIMS:
Per CA Insurance Code § 10103.7:
- Advance payment of no less than 4 months of additional living expenses
- Contents claim: insured may provide a lump sum estimate
- Insurer cannot require room-by-room inventory for initial payment

EARTHQUAKE COVERAGE:
Per CA Insurance Code § 10081, every residential property insurer must
offer earthquake coverage. This offer must be made at policy inception
and renewal. The insured may decline in writing.

UNFAIR SETTLEMENT PRACTICES (CA § 790.03(h)):
Includes: misrepresenting policy provisions, failing to acknowledge
claims promptly, not attempting fair settlement when liability is clear,
compelling litigation by offering substantially less than recovery amount.
""",
    },
    "REG-TX": {
        "state": "TX",
        "content": """STATE OF TEXAS — INSURANCE CLAIMS REGULATIONS SUMMARY
============================================================

APPLICABLE STATUTES:
Texas Insurance Code and Texas Administrative Code Title 28

CLAIM ACKNOWLEDGMENT:
Per TX Insurance Code § 542.055, the insurer must acknowledge receipt
of a claim within 15 calendar days. This acknowledgment must be in writing.

CLAIM INVESTIGATION:
Per TX Insurance Code § 542.055(b), the insurer must commence
investigation of the claim no later than 15 days after receiving
notice of the claim.

CLAIM DECISION:
Per TX Insurance Code § 542.056, the insurer must accept or reject
the claim no later than 15 business days after receiving all items,
statements, and forms reasonably requested.

PROMPT PAYMENT:
Per TX Insurance Code § 542.057, if the insurer notifies the claimant
that the claim is accepted (in whole or part), the insurer must pay
the claim within 5 business days of notification.

PENALTIES FOR DELAY:
Per TX Insurance Code § 542.060, if an insurer violates the prompt
payment statutes, the insurer is liable for:
- The amount of the claim
- 18% annual interest on the claim amount
- Reasonable attorney's fees

BAD FAITH:
Per TX Insurance Code § 541.060, an insured may bring action for
unfair settlement practices. Damages may include actual damages,
court costs, attorney's fees, and up to three times actual damages
if the violation was committed knowingly.

HAIL/WIND CLAIMS:
Per TX Insurance Code § 4002.052 (TWIA):
- TWIA policies have specific wind/hail provisions
- Non-TWIA policies: standard homeowner provisions apply
- Separate wind/hail deductible is common in Texas policies

MOLD COVERAGE:
Mold coverage in Texas is typically offered as an optional endorsement.
Per the Texas Department of Insurance, mold liability caps were
established to control costs. Insurers are not required to include
mold coverage in standard homeowner policies.
""",
    },
    "REG-NY": {
        "state": "NY",
        "content": """STATE OF NEW YORK — INSURANCE CLAIMS REGULATIONS SUMMARY
============================================================

APPLICABLE STATUTES:
New York Insurance Law and NY Regulation 64 (11 NYCRR 216)

CLAIM ACKNOWLEDGMENT:
Per NY Regulation 64 § 216.4(a), the insurer must acknowledge receipt
of a claim within 15 business days. The acknowledgment must provide
the name of the claims examiner handling the matter.

CLAIM INVESTIGATION:
Per NY Regulation 64 § 216.6(a), the insurer must complete its
investigation within 30 calendar days of receiving proof of loss,
unless the investigation cannot reasonably be completed within that time.

CLAIM DECISION:
Per NY Insurance Law § 3408(a), the insurer must make a determination
on the claim within 30 business days after proof of loss is received.

PROMPT PAYMENT:
Per NY Regulation 64 § 216.6(d), payment must be made within 35
business days of proof of loss being received, provided the claim
has been approved.

BAD FAITH:
New York does not have a specific bad faith statute for first-party
claims. However, courts have recognized an implied covenant of good
faith and fair dealing. Claimants may bring action for consequential
damages under certain circumstances.

NAMED STORM PROVISIONS:
Per NY Insurance Law § 3425:
- Named storm deductible may not exceed 5% of Coverage A
- Named storm deductible applies only when the National Weather Service
  declares a hurricane warning or watch for the area
- Standard deductible applies to all other wind losses

WATER DAMAGE:
Per NY Regulation 64, insurers must clearly disclose any sublimits
for water damage, particularly for below-grade (basement) units.
The policy must clearly state whether sewer backup is covered
and any applicable sublimits.
""",
    },
    "REG-IL": {
        "state": "IL",
        "content": """STATE OF ILLINOIS — INSURANCE CLAIMS REGULATIONS SUMMARY
============================================================

APPLICABLE STATUTES:
215 ILCS 5/ Illinois Insurance Code and 50 Ill. Adm. Code 919

CLAIM ACKNOWLEDGMENT:
Per 50 Ill. Adm. Code 919.50(a), the insurer must acknowledge receipt
of a claim within 15 business days. The acknowledgment must provide
necessary claim forms and instructions.

CLAIM INVESTIGATION:
Per 50 Ill. Adm. Code 919.50(b), the insurer must begin a reasonable
investigation within 15 business days of receiving a claim.

CLAIM DECISION:
Per 215 ILCS 5/154.6, the insurer must approve or deny a claim
within 30 days after receipt of proof of loss.

PROMPT PAYMENT:
Per 215 ILCS 5/154.6, once approved, the insurer must pay the claim
within 30 days after receiving satisfactory proof of loss.

PENALTIES FOR DELAY:
Per 215 ILCS 5/154.6(i), if payment is not made within the time
specified, the insurer must pay interest at the rate of 9% per annum
from the date the payment should have been made.

BAD FAITH:
Per 215 ILCS 5/155, an insured may bring action for unreasonable and
vexatious delay in settling a claim. The court may award:
- Reasonable attorney's fees
- Costs of the action
- An additional amount not to exceed 60% of the amount recovered

WINTER WEATHER CLAIMS:
Illinois experiences significant winter weather claims (frozen pipes,
ice dams, weight of snow). Per standard HO-3 provisions:
- Frozen pipes: covered if insured maintained heat or shut off water
- Ice dam damage: covered for resulting water damage
- Weight of snow: covered for structural damage

LOSS OF USE:
Per Illinois practice, loss of use (Coverage D) claims must be
supported by documentation of actual additional living expenses.
The insured must mitigate costs by choosing reasonable accommodations.
""",
    },
}


# ── Adjuster Notes ─────────────────────────────────────────────────────────

ADJUSTER_NOTES = [
    {
        "id": "NOTE-001",
        "claim": "CLM-FL-2024-001",
        "policy": "PLY-FL-001",
        "state": "FL",
        "adjuster": "Thomas Rivera",
        "content": """ADJUSTER FIELD NOTES — CLM-FL-2024-001
Date of Inspection: 03/18/2024
Adjuster: Thomas Rivera

INSPECTION SUMMARY:
Arrived at property at 9:00 AM. Met with insured Maria Gonzalez who
walked me through the damage. The burst pipe is in the second-floor
bathroom wall — a copper supply line with a failed solder joint.
Plumber on-site confirmed this is a sudden failure, not a slow leak.

OBSERVATIONS:
1. Water damage is consistent with a sudden pipe burst event.
2. Mold was found behind the bathroom wall during demolition.
   The mold appears to be recent growth (consistent with 3-day timeline
   since the loss). No evidence of pre-existing mold or long-term moisture.
3. The insured had the water shut off within 4 hours — reasonable response.
4. Mitigation company (ServPro) was called same day — drying equipment
   deployed March 16.

RED FLAGS: None identified. This appears to be a straightforward
covered water damage claim.

MOLD NOTE: Mold remediation should be covered under the mold endorsement
(HO-MOLD-FL-2024, $10,000 limit). The estimated mold cost of $4,500
is within the endorsement limit.

RECOMMENDATION: Approve claim. All damage is consistent with covered
peril (sudden pipe burst). Mold is a direct result of the covered event.
""",
    },
    {
        "id": "NOTE-002",
        "claim": "CLM-CA-2024-003",
        "policy": "PLY-CA-002",
        "state": "CA",
        "adjuster": "Linda Park",
        "content": """ADJUSTER FIELD NOTES — CLM-CA-2024-003
Date of Inspection: 04/24/2024
Adjuster: Linda Park

INSPECTION SUMMARY:
Inspected the fire-damaged property at 892 Hillside Dr, Los Angeles.
Fire department report confirms origin in the kitchen (unattended cooking).
The insured, James Chen, was present and cooperative.

OBSERVATIONS:
1. Kitchen is a total loss — fire damage is severe and consistent with
   a grease fire that spread to cabinets and ceiling.
2. Dining room has extensive heat and smoke damage. The shared wall with
   the kitchen partially collapsed.
3. Smoke damage is heavy throughout the first floor and moderate on the
   second floor. Professional smoke remediation will be needed.
4. Structural engineer assessment may be needed for the kitchen/dining
   room wall. I've requested one.

RED FLAGS:
- The insured mentioned recent financial difficulties (job loss 2 months
  ago). This alone is not indicative of fraud but should be noted.
- Fire origin is consistent with accidental kitchen fire. No accelerant
  patterns observed. Fire marshal report supports accidental cause.
- No prior fire claims on this policy.

SUBROGATION: None identified. This is an accidental fire with no
third-party involvement.

RECOMMENDATION: Proceed with claim. Request structural engineer report
before finalizing estimate. Current estimate of $185,000 appears
reasonable given scope of damage.
""",
    },
    {
        "id": "NOTE-003",
        "claim": "CLM-TX-2024-005",
        "policy": "PLY-TX-003",
        "state": "TX",
        "adjuster": "Michael Torres",
        "content": """ADJUSTER FIELD NOTES — CLM-TX-2024-005
Date of Inspection: 05/21/2024
Adjuster: Michael Torres

INSPECTION SUMMARY:
Inspected water damage at 1205 Lone Star Pkwy, Houston. The insured
Robert Williams was present. Significant water damage from a failed
hot water supply line in the master bathroom wall.

OBSERVATIONS:
1. Supply line failure point is clearly visible — the PEX line split at a
   connection fitting. This is a sudden and accidental failure.
2. Water damage is extensive due to the 8-hour duration of the leak.
   The insured was at work and did not discover the issue until returning.
3. Mold was discovered in the wall cavity during demolition. Based on
   growth patterns, the mold appears to have developed after the water
   event (consistent with 3-day timeline).

CRITICAL COVERAGE ISSUE:
Policy PLY-TX-003 does NOT include the optional mold remediation
endorsement. The estimated mold remediation cost is $3,000.
Per Texas regulations, mold coverage is optional and this insured
did not purchase it. The mold portion of this claim ($3,000) should
be DENIED. All other water damage repairs ($39,000) are covered.

FOUNDATION NOTE: Moisture detected under the slab. Foundation engineer
assessment has been ordered. If foundation damage is found to be
sudden and accidental (caused by this event), it would be covered.

RECOMMENDATION: Approve $39,000 for water damage repairs. Deny $3,000
for mold remediation (no endorsement). Await foundation assessment.
""",
    },
    {
        "id": "NOTE-004",
        "claim": "CLM-NY-2024-007",
        "policy": "PLY-NY-004",
        "state": "NY",
        "adjuster": "Jennifer Walsh",
        "content": """ADJUSTER FIELD NOTES — CLM-NY-2024-007
Date of Inspection: 10/14/2024
Adjuster: Jennifer Walsh

INSPECTION SUMMARY:
Inspected water backup damage at 315 Park Ave, Brooklyn basement unit.
The insured Sarah Patel was present. Building superintendent also
provided access to the building's main sewer line.

OBSERVATIONS:
1. The damage is clearly from sewer backup — contaminated water entered
   through the floor drains during the heavy rainfall event of 10/12.
2. Water mark on the walls confirms approximately 6 inches of standing
   water. The contamination level (Category 3 — black water) requires
   professional biohazard remediation.
3. All affected materials (drywall, flooring, contents) in contact with
   sewage water must be removed and replaced per health codes.

COVERAGE ANALYSIS:
- Water backup endorsement HO-WBSD-NY-2024 is in effect ($10,000 limit)
- Below-grade sublimit per NY provisions: $25,000
- Total claim estimate: $18,500
- Claim is within BOTH limits (water backup endorsement at $10,000 will
  be the controlling limit)

IMPORTANT: The total claim ($18,500) exceeds the water backup endorsement
limit of $10,000. Maximum payable under the endorsement is $10,000.
The remaining $8,500 would need to be reviewed for coverage under the
standard water damage provisions (if applicable).

RECOMMENDATION: Approve up to $10,000 under water backup endorsement.
Review whether additional $8,500 qualifies under standard water damage
provisions.
""",
    },
    {
        "id": "NOTE-005",
        "claim": "CLM-IL-2024-009",
        "policy": "PLY-IL-005",
        "state": "IL",
        "adjuster": "Kevin O'Brien",
        "content": """ADJUSTER FIELD NOTES — CLM-IL-2024-009
Date of Inspection: 01/24/2024
Adjuster: Kevin O'Brien

INSPECTION SUMMARY:
Inspected fire damage at 780 Michigan Ave, Chicago. The insured David
Thompson and his family have been displaced and are in temporary housing.
Chicago Fire Department report obtained — fire origin confirmed as
electrical (faulty outlet in second-floor bedroom).

OBSERVATIONS:
1. Second floor bedroom (fire origin) is a total loss. Fire burned through
   walls, ceiling, and floor in this room.
2. The fire was contained to the second floor but heat and smoke traveled
   throughout the structure. All rooms have at minimum smoke damage.
3. Attic insulation has heat damage and portions of the roof deck show
   charring. Full attic inspection needed.
4. The electrical panel was inspected — the outlet that started the fire
   was on a circuit with no AFCI protection (older installation).

SUBROGATION POTENTIAL:
The faulty outlet may have been a defective product. The insured states
the outlet was original to the building (built 2005). I've preserved the
outlet for potential product liability investigation. If the manufacturer
is identified, subrogation may be pursued to recover claim costs.

LOSS OF USE:
The home is uninhabitable. The insured family (2 adults, 2 children)
is staying at a nearby extended-stay hotel ($175/night). Estimated
repair timeline: 3 months. Loss of use estimate: $10,000 is reasonable
under Coverage D ($96,000 limit).

RECOMMENDATION: Approve claim. Clear covered peril (electrical fire).
Pursue subrogation investigation for the defective outlet.
""",
    },
]


# ── Write files ────────────────────────────────────────────────────────────


def main():
    for d in DIRS:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

    # Policies
    for pid, p in POLICIES.items():
        path = os.path.join(BASE_DIR, "policies", f"{pid}.txt")
        with open(path, "w") as f:
            f.write(generate_policy(pid, p))
        register(
            f"policies/{pid}.txt",
            doc_type="policy",
            state=p["state"],
            carrier_id="GWMI-001",
            lob="homeowners",
            policy_number=pid,
            claim_id=None,
        )
    print(f"  Generated {len(POLICIES)} policies")

    # Claims
    for c in CLAIMS:
        path = os.path.join(BASE_DIR, "claims", f"{c['id']}.txt")
        with open(path, "w") as f:
            f.write(c["description"])
        register(
            f"claims/{c['id']}.txt",
            doc_type="claim",
            state=c["state"],
            carrier_id="GWMI-001",
            lob="homeowners",
            policy_number=c["policy"],
            claim_id=c["id"],
            date_of_loss=c["date_of_loss"],
            peril=c["peril"],
            estimated_amount=c["estimated_amount"],
        )
    print(f"  Generated {len(CLAIMS)} claims")

    # Regulations
    for rid, r in REGULATIONS.items():
        path = os.path.join(BASE_DIR, "regulations", f"{rid}.txt")
        with open(path, "w") as f:
            f.write(r["content"])
        register(
            f"regulations/{rid}.txt",
            doc_type="regulation",
            state=r["state"],
            carrier_id=None,
            lob=None,
            policy_number=None,
            claim_id=None,
        )
    print(f"  Generated {len(REGULATIONS)} regulations")

    # Adjuster notes
    for n in ADJUSTER_NOTES:
        path = os.path.join(BASE_DIR, "adjuster_notes", f"{n['id']}.txt")
        with open(path, "w") as f:
            f.write(n["content"])
        register(
            f"adjuster_notes/{n['id']}.txt",
            doc_type="adjuster_note",
            state=n["state"],
            carrier_id="GWMI-001",
            lob="homeowners",
            policy_number=n["policy"],
            claim_id=n["claim"],
        )
    print(f"  Generated {len(ADJUSTER_NOTES)} adjuster notes")

    # Metadata
    meta_path = os.path.join(BASE_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata.json ({len(metadata)} entries)")

    total = len(POLICIES) + len(CLAIMS) + len(REGULATIONS) + len(ADJUSTER_NOTES)
    print(f"\nDone — generated {total} documents in {BASE_DIR}")


if __name__ == "__main__":
    main()
