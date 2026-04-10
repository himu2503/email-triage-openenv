"""
OpenEnv Email Triage — Task Definitions and Graders

3 tasks with increasing difficulty:
  - easy:   3 emails, classify only (single correct label)
  - medium: 5 emails, classify + route (label + action_type)
  - hard:   8 emails, classify + route + draft reply (full pipeline)

Each grader returns a Reward with score in [0.0, 1.0] and partial credits.
"""

import re
from typing import Dict, Any
from models import Reward


# ─────────────────────────────────────────────
# EASY TASK — 3 emails, obvious classification
# ─────────────────────────────────────────────

EASY_EMAILS = [
    {
        "email_id": "e1",
        "subject": "URGENT: Production server is DOWN",
        "body": (
            "Hi team,\n\n"
            "Our production server has been unreachable since 08:52 AM. "
            "Customers cannot access the platform. "
            "This is a P0 incident — all hands needed IMMEDIATELY.\n\n"
            "— DevOps"
        ),
        "sender": "devops@company.com",
        "timestamp": "2024-01-15T08:55:00",
    },
    {
        "email_id": "e2",
        "subject": "You've won a $1000 Amazon gift card!!!",
        "body": (
            "Congratulations! You have been selected as our lucky winner. "
            "Click here to claim your prize: http://totally-legit-prize.xyz/claim "
            "Limited time offer! Act now! Do not miss out!"
        ),
        "sender": "prizes@totally-legit-prize.xyz",
        "timestamp": "2024-01-15T09:10:00",
    },
    {
        "email_id": "e3",
        "subject": "Invoice #INV-2024-0042 from Acme Corp",
        "body": (
            "Dear Finance Team,\n\n"
            "Please find attached Invoice #INV-2024-0042 for $4,250.00 "
            "for software licensing services rendered in December 2023. "
            "Payment is due within 30 days.\n\n"
            "Thank you,\nAcme Corp Billing"
        ),
        "sender": "billing@acmecorp.com",
        "timestamp": "2024-01-15T09:20:00",
    },
]

EASY_LABELS = {
    "e1": "urgent",
    "e2": "spam",
    "e3": "billing",
}


def easy_grader(email: Dict, action, history: list) -> Reward:
    """
    Easy grader: only checks the label.
    Full credit (1.0) for correct label, 0.0 for wrong.
    """
    correct = EASY_LABELS.get(email["email_id"], "other")
    if action.label == correct:
        return Reward(
            score=1.0,
            partial_credits={"label": 1.0},
            reason=f"Correct label: '{correct}'",
            correct_label=correct,
        )
    return Reward(
        score=0.0,
        partial_credits={"label": 0.0},
        reason=f"Wrong label. Expected '{correct}', got '{action.label}'",
        correct_label=correct,
    )


# ─────────────────────────────────────────────
# MEDIUM TASK — 5 emails, label + action_type
# ─────────────────────────────────────────────

MEDIUM_EMAILS = [
    {
        "email_id": "m1",
        "subject": "Request for emergency leave",
        "body": (
            "Hi,\n\nI need to take emergency leave starting today due to a "
            "family medical situation. I'm not sure when I'll be back. "
            "Please let me know the process for emergency leave approval.\n\n"
            "— Sarah"
        ),
        "sender": "sarah.k@company.com",
        "timestamp": "2024-01-15T07:30:00",
    },
    {
        "email_id": "m2",
        "subject": "Monthly newsletter — January 2024",
        "body": (
            "Hello Subscriber,\n\n"
            "Welcome to the January edition of the TechDigest newsletter. "
            "This month: AI trends, productivity tips, and upcoming webinars. "
            "To unsubscribe, click here."
        ),
        "sender": "newsletter@techdigest.io",
        "timestamp": "2024-01-15T08:00:00",
    },
    {
        "email_id": "m3",
        "subject": "My laptop won't connect to VPN",
        "body": (
            "Hi IT,\n\n"
            "Since this morning I can't connect to the VPN. "
            "I've tried restarting but the error says 'Authentication failed'. "
            "I have a client presentation at 2 PM and need this fixed urgently.\n\n"
            "Thanks,\nMarcus"
        ),
        "sender": "marcus.l@company.com",
        "timestamp": "2024-01-15T09:45:00",
    },
    {
        "email_id": "m4",
        "subject": "Re: Project Phoenix — status update",
        "body": (
            "Hey team,\n\n"
            "Just a quick update on Project Phoenix: we're on track for the "
            "Q1 deadline. Design mockups are done, backend API is 80% complete. "
            "Next sync is Thursday 3 PM.\n\n"
            "— Priya"
        ),
        "sender": "priya.s@company.com",
        "timestamp": "2024-01-15T10:00:00",
    },
    {
        "email_id": "m5",
        "subject": "Overdue invoice — final notice",
        "body": (
            "Dear Account Holder,\n\n"
            "This is a final notice regarding Invoice #2023-1187 for $8,500 "
            "which is now 45 days overdue. Failure to pay within 7 days will "
            "result in service suspension and referral to collections.\n\n"
            "Regards,\nCloud Services Billing"
        ),
        "sender": "billing@cloudservices.com",
        "timestamp": "2024-01-15T10:30:00",
    },
]

MEDIUM_EXPECTED = {
    "m1": {"label": "hr",          "action_type": "escalate",  "priority": "high"},
    "m2": {"label": "newsletter",  "action_type": "archive",   "priority": "low"},
    "m3": {"label": "it_support",  "action_type": "escalate",  "priority": "high"},
    "m4": {"label": "internal",    "action_type": "classify",  "priority": "medium"},
    "m5": {"label": "billing",     "action_type": "escalate",  "priority": "high"},
}


def medium_grader(email: Dict, action, history: list) -> Reward:
    """
    Medium grader: checks label (50%), action_type (30%), priority (20%).
    Partial credit for each component.
    """
    eid = email["email_id"]
    expected = MEDIUM_EXPECTED.get(eid, {})
    credits = {}
    score = 0.0

    # Label — 50%
    if action.label == expected.get("label"):
        credits["label"] = 0.5
        score += 0.5
    else:
        credits["label"] = 0.0

    # Action type — 30%
    if action.action_type == expected.get("action_type"):
        credits["action_type"] = 0.3
        score += 0.3
    else:
        credits["action_type"] = 0.0

    # Priority — 20%
    if action.priority == expected.get("priority"):
        credits["priority"] = 0.2
        score += 0.2
    else:
        credits["priority"] = 0.0

    reasons = []
    if credits["label"] == 0:
        reasons.append(f"label: expected '{expected.get('label')}', got '{action.label}'")
    if credits["action_type"] == 0:
        reasons.append(f"action_type: expected '{expected.get('action_type')}', got '{action.action_type}'")
    if credits["priority"] == 0:
        reasons.append(f"priority: expected '{expected.get('priority')}', got '{action.priority}'")

    reason = "All correct!" if not reasons else "Issues: " + "; ".join(reasons)

    return Reward(
        score=round(score, 2),
        partial_credits=credits,
        reason=reason,
        correct_label=expected.get("label"),
        correct_action_type=expected.get("action_type"),
    )


# ─────────────────────────────────────────────
# HARD TASK — 8 emails, label + action + reply
# ─────────────────────────────────────────────

HARD_EMAILS = [
    {
        "email_id": "h1",
        "subject": "Data breach suspected — customer data may be exposed",
        "body": (
            "Team,\n\n"
            "Our security scan flagged unusual outbound traffic at 3 AM. "
            "We suspect a data breach affecting ~10,000 customer records "
            "including emails and hashed passwords. Legal and exec need to know NOW. "
            "This may trigger GDPR notification obligations within 72 hours.\n\n"
            "— Security Team"
        ),
        "sender": "security@company.com",
        "timestamp": "2024-01-15T06:00:00",
    },
    {
        "email_id": "h2",
        "subject": "Question about your refund policy",
        "body": (
            "Hello,\n\n"
            "I purchased your Pro plan 8 days ago but it doesn't meet my needs. "
            "Your website says 7-day refund policy. I'm only 1 day over — "
            "would you be able to make an exception and process a refund? "
            "Order #ORD-88421.\n\n"
            "Thanks,\nLena"
        ),
        "sender": "lena.m@gmail.com",
        "timestamp": "2024-01-15T08:20:00",
    },
    {
        "email_id": "h3",
        "subject": "Reminder: Team lunch today at noon",
        "body": (
            "Hi all,\n\n"
            "Just a reminder that team lunch is today at 12 PM at Noodle House "
            "on 5th Ave. Please RSVP if you haven't already so we can confirm the booking.\n\n"
            "— Events Committee"
        ),
        "sender": "events@company.com",
        "timestamp": "2024-01-15T09:00:00",
    },
    {
        "email_id": "h4",
        "subject": "Partnership proposal — AI integration",
        "body": (
            "Dear Team,\n\n"
            "We are a Series B startup specializing in AI-powered analytics. "
            "We believe a partnership with your platform could create significant value "
            "for both companies. We'd love to schedule a 30-minute intro call this week "
            "to explore synergies.\n\n"
            "Best,\nDavid Chen, CEO — DataSpark AI"
        ),
        "sender": "david.chen@dataspark.ai",
        "timestamp": "2024-01-15T09:30:00",
    },
    {
        "email_id": "h5",
        "subject": "Payroll issue — missing January salary",
        "body": (
            "Hi HR,\n\n"
            "My January salary has not been credited to my bank account as of today. "
            "The usual payment date is the 14th. This is causing me financial difficulty. "
            "Could you urgently check what happened? Employee ID: EMP-0421.\n\n"
            "— James"
        ),
        "sender": "james.r@company.com",
        "timestamp": "2024-01-15T10:00:00",
    },
    {
        "email_id": "h6",
        "subject": "Free SEO audit for your website!!!",
        "body": (
            "Hi there,\n\n"
            "We noticed your website is not ranking on Google. "
            "We offer FREE SEO audits and can get you to page 1 GUARANTEED! "
            "Reply now or click: http://seo-spam-link.biz\n\n"
            "Best,\nSEO Experts Team"
        ),
        "sender": "seo@seo-spam-link.biz",
        "timestamp": "2024-01-15T10:15:00",
    },
    {
        "email_id": "h7",
        "subject": "Bug report: payment processing fails for EU users",
        "body": (
            "Hi,\n\n"
            "We're seeing a critical bug where EU-based users get a 500 error "
            "during checkout. This started around 10 AM today. "
            "Approximately 15% of EU checkouts are failing. "
            "Revenue impact is estimated at $3,000/hour.\n\n"
            "— Backend Engineering"
        ),
        "sender": "backend@company.com",
        "timestamp": "2024-01-15T10:45:00",
    },
    {
        "email_id": "h8",
        "subject": "Can I schedule a 1:1 with you this week?",
        "body": (
            "Hi,\n\n"
            "I wanted to check in about my performance review and also discuss "
            "some ideas I have for the Q2 roadmap. Do you have 30 minutes "
            "available Thursday or Friday afternoon?\n\n"
            "— Alex"
        ),
        "sender": "alex.p@company.com",
        "timestamp": "2024-01-15T11:00:00",
    },
]

HARD_EXPECTED = {
    "h1": {
        "label": "urgent",
        "action_type": "escalate",
        "priority": "high",
        "reply_keywords": ["gdpr", "breach", "security", "legal", "notify", "72 hour"],
    },
    "h2": {
        "label": "inquiry",
        "action_type": "reply",
        "priority": "medium",
        "reply_keywords": ["refund", "exception", "order", "policy", "apolog", "understand"],
    },
    "h3": {
        "label": "internal",
        "action_type": "archive",
        "priority": "low",
        "reply_keywords": [],
    },
    "h4": {
        "label": "inquiry",
        "action_type": "reply",
        "priority": "medium",
        "reply_keywords": ["partnership", "call", "interest", "schedule", "discuss"],
    },
    "h5": {
        "label": "hr",
        "action_type": "escalate",
        "priority": "high",
        "reply_keywords": ["salary", "payroll", "urgent", "investigate", "apolog"],
    },
    "h6": {
        "label": "spam",
        "action_type": "delete",
        "priority": "low",
        "reply_keywords": [],
    },
    "h7": {
        "label": "urgent",
        "action_type": "escalate",
        "priority": "high",
        "reply_keywords": ["bug", "payment", "eu", "fix", "investigate", "critical"],
    },
    "h8": {
        "label": "internal",
        "action_type": "reply",
        "priority": "medium",
        "reply_keywords": ["thursday", "friday", "meeting", "schedule", "available", "1:1"],
    },
}


def _score_reply(reply_text: str, keywords: list) -> float:
    """Score a reply based on keyword presence. Returns 0.0–1.0."""
    if not keywords:
        return 0.99  # No reply expected, full credit
    if not reply_text:
        return 0.01
    reply_lower = reply_text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
    # Need at least 2 keywords for partial credit, 4+ for full
    if matched == 0:
        return 0.01
    elif matched == 1:
        return 0.2
    elif matched == 2:
        return 0.5
    elif matched == 3:
        return 0.7
    else:
        return min(1.0, 0.7 + 0.1 * (matched - 3))


def hard_grader(email: Dict, action, history: list) -> Reward:
    """
    Hard grader: label (30%), action_type (25%), priority (15%), reply quality (30%).
    Reply scoring uses keyword matching against expected response components.
    """
    eid = email["email_id"]
    expected = HARD_EXPECTED.get(eid, {})
    credits = {}
    score = 0.01

    # Label — 30%
    if action.label == expected.get("label"):
        credits["label"] = 0.3
        score += 0.3
    else:
        credits["label"] = 0.0

    # Action type — 25%
    if action.action_type == expected.get("action_type"):
        credits["action_type"] = 0.25
        score += 0.25
    else:
        credits["action_type"] = 0.0

    # Priority — 15%
    if action.priority == expected.get("priority"):
        credits["priority"] = 0.15
        score += 0.15
    else:
        credits["priority"] = 0.0

    # Reply quality — 30%
    keywords = expected.get("reply_keywords", [])
    needs_reply = expected.get("action_type") in ("reply", "escalate")

    if not needs_reply or not keywords:
        # No reply needed (archive/delete) — full credit for this component
        credits["reply_quality"] = 0.3
        score += 0.3
    else:
        reply_score = _score_reply(action.reply_text or "", keywords)
        credits["reply_quality"] = round(reply_score * 0.3, 3)
        score += credits["reply_quality"]

    reasons = []
    if credits["label"] == 0:
        reasons.append(f"label: expected '{expected.get('label')}', got '{action.label}'")
    if credits["action_type"] == 0:
        reasons.append(f"action_type: expected '{expected.get('action_type')}', got '{action.action_type}'")
    if credits["priority"] == 0:
        reasons.append(f"priority: expected '{expected.get('priority')}', got '{action.priority}'")
    if credits.get("reply_quality", 0.3) < 0.15:
        reasons.append("reply quality too low — missing key content")

    reason = "All correct!" if not reasons else "Issues: " + "; ".join(reasons)

    return Reward(
        score=round(score, 3),
        partial_credits=credits,
        reason=reason,
        correct_label=expected.get("label"),
        correct_action_type=expected.get("action_type"),
    )


# ─────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────

TASKS = {
    "easy": {
        "id": "easy",
        "description": "Classify 3 clearly labeled emails (spam, urgent, billing)",
        "difficulty": "easy",
        "emails": EASY_EMAILS,
        "grader": easy_grader,
        "total_steps": len(EASY_EMAILS),
    },
    "medium": {
        "id": "medium",
        "description": "Classify, route, and prioritize 5 ambiguous emails",
        "difficulty": "medium",
        "emails": MEDIUM_EMAILS,
        "grader": medium_grader,
        "total_steps": len(MEDIUM_EMAILS),
    },
    "hard": {
        "id": "hard",
        "description": "Classify, route, prioritize, and draft replies for 8 complex emails",
        "difficulty": "hard",
        "emails": HARD_EMAILS,
        "grader": hard_grader,
        "total_steps": len(HARD_EMAILS),
    },
}
