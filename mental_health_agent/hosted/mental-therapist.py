import os; os.environ.setdefault("AGENT_ADDRESSES_LOADED", "false")
# Agent addresses are set via environment so hosted runtime can wire startup config.

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import requests
from requests.auth import HTTPBasicAuth

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)


# ---------------------------------------------------------------------------
# Agent + protocol bootstrap
# ---------------------------------------------------------------------------

agent = Agent(
    name="mental-therapist",
    seed=os.environ.get("AGENT_SEED", "fallback_seed_change_in_prod"),
    mailbox=True,
)

# Keep local import compatibility for run_local.py naming.
orchestrator = agent
chat_proto = Protocol(spec=chat_protocol_spec)


# ---------------------------------------------------------------------------
# Clinical instruments (verbatim, public-domain question text)
# ---------------------------------------------------------------------------

WHO5_QUESTIONS = [
    "Over the past 2 weeks, how often have you felt cheerful and in good spirits?",
    "Over the past 2 weeks, how often have you felt calm and relaxed?",
    "Over the past 2 weeks, how often have you felt active and vigorous?",
    "Over the past 2 weeks, how often did you wake up feeling fresh and rested?",
    "Over the past 2 weeks, how often has your daily life been filled with things that interest you?",
]

WHO5_OPTIONS_HINT = (
    "Reply with one of: 'All of the time', 'Most of the time', 'More than half of the time', "
    "'Less than half of the time', 'Some of the time', 'At no time'."
)

PHQ9_QUESTIONS = [
    "Over the past 2 weeks, how often have you had little interest or pleasure in doing things?",
    "Over the past 2 weeks, how often have you been feeling down, depressed, or hopeless?",
    "Over the past 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "Over the past 2 weeks, how often have you been feeling tired or having little energy?",
    "Over the past 2 weeks, how often have you had poor appetite or been overeating?",
    "Over the past 2 weeks, how often have you felt bad about yourself, that you are a failure, or have let yourself or your family down?",
    "Over the past 2 weeks, how often have you had trouble concentrating on things, such as reading or watching television?",
    "Over the past 2 weeks, how often have you been moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you've been moving around a lot more than usual?",
    "Over the past 2 weeks, how often have you had thoughts that you would be better off dead or of hurting yourself in some way?",
]

GAD7_QUESTIONS = [
    "Over the past 2 weeks, how often have you been feeling nervous, anxious, or on edge?",
    "Over the past 2 weeks, how often have you not been able to stop or control worrying?",
    "Over the past 2 weeks, how often have you been worrying too much about different things?",
    "Over the past 2 weeks, how often have you had trouble relaxing?",
    "Over the past 2 weeks, how often have you been so restless that it is hard to sit still?",
    "Over the past 2 weeks, how often have you become easily annoyed or irritable?",
    "Over the past 2 weeks, how often have you been feeling afraid as if something awful might happen?",
]

PHQ_GAD_OPTIONS_HINT = (
    "Reply with one of: 'Not at all', 'Several days', 'More than half the days', 'Nearly every day'."
)


# ---------------------------------------------------------------------------
# Crisis resources
# ---------------------------------------------------------------------------

CRISIS_RESPONSE_TEXT = (
    "I'm really concerned about your safety right now. Please speak to someone right away.\n\n"
    "Immediate helplines:\n"
    "- iCall (India): 9152987821\n"
    "- Vandrevala Foundation (India, 24/7): 1860-2662-345\n"
    "- 988 Suicide & Crisis Lifeline (US): call or text 988\n\n"
    "If you are in immediate danger, please call your local emergency number or go to the nearest emergency room. "
    "You don't have to face this alone."
)


# ---------------------------------------------------------------------------
# NPI taxonomy mapping
# ---------------------------------------------------------------------------

# NPI Registry's `taxonomy_description` parameter is a text search over the
# taxonomy *description*, not the alphanumeric code. We therefore use search
# terms that match the published Health Care Provider Taxonomy descriptions.
TAXONOMY_MAP = {
    "Outpatient therapy": ["Psychologist", "Counselor", "Social Worker"],
    "Intensive outpatient": ["Psychiatry", "Psychologist"],
    "Inpatient evaluation": ["Psychiatry"],
    "Self-help resources": [],
}

CURATED_SELF_HELP_RESOURCES = [
    "988 Suicide & Crisis Lifeline (US): Call or text 988",
    "SAMHSA treatment locator: https://findtreatment.gov/",
    "NAMI helpline: 1-800-950-NAMI (6264)",
    "iCall (India): 9152987821",
]

DISCLAIMER = (
    "This report was generated by an AI-assisted assessment tool. "
    "It is not a clinical diagnosis. Please share with a licensed mental health professional."
)


# ---------------------------------------------------------------------------
# Free-text answer parsing
# ---------------------------------------------------------------------------

WHO5_KEYWORDS: list[tuple[str, int]] = [
    ("all of the time", 5),
    ("all the time", 5),
    ("every time", 5),
    ("always", 5),
    ("most of the time", 4),
    ("most of time", 4),
    ("usually", 4),
    ("often", 4),
    ("more than half of the time", 3),
    ("more than half the time", 3),
    ("more than half", 3),
    ("less than half of the time", 2),
    ("less than half the time", 2),
    ("less than half", 2),
    ("some of the time", 1),
    ("sometimes", 1),
    ("at no time", 0),
    ("none of the time", 0),
    ("never", 0),
    ("none", 0),
]

PHQ_GAD_KEYWORDS: list[tuple[str, int]] = [
    ("nearly every day", 3),
    ("every day", 3),
    ("daily", 3),
    ("more than half the days", 2),
    ("more than half", 2),
    ("several days", 1),
    ("a few days", 1),
    ("few days", 1),
    ("some days", 1),
    ("not at all", 0),
    ("never", 0),
    ("none", 0),
]


def _parse_who5_answer(text: str) -> int:
    """Map free-text WHO-5 answer to 0..5."""
    lower = text.lower().strip()
    digit_match = re.fullmatch(r"\s*([0-5])\s*", lower)
    if digit_match:
        return int(digit_match.group(1))
    for keyword, value in WHO5_KEYWORDS:
        if keyword in lower:
            return value
    return 2  # neutral fallback


def _parse_phq_gad_answer(text: str) -> int:
    """Map free-text PHQ-9/GAD-7 answer to 0..3."""
    lower = text.lower().strip()
    digit_match = re.fullmatch(r"\s*([0-3])\s*", lower)
    if digit_match:
        return int(digit_match.group(1))
    for keyword, value in PHQ_GAD_KEYWORDS:
        if keyword in lower:
            return value
    return 0  # safest fallback


def _is_yes(text: str) -> bool:
    lowered = text.lower().strip()
    return any(token in lowered for token in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "y"])


def _is_no(text: str) -> bool:
    lowered = text.lower().strip()
    return any(token in lowered for token in ["no", "not now", "nope", "skip", "later", "n"])


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _score_who5(answers: list[int]) -> dict[str, Any]:
    raw_25 = sum(int(a) for a in answers)
    raw = raw_25 * 4
    if raw <= 28:
        severity = "Poor wellbeing"
        crossed = True
    elif raw <= 50:
        severity = "Below average wellbeing"
        crossed = False
    else:
        severity = "Above average wellbeing"
        crossed = False
    return {
        "scale": "WHO-5",
        "raw_score": raw,
        "severity": severity,
        "threshold_crossed": crossed,
        "clinical_note": "WHO-5 raw 0-100. Scores at or below 28 indicate poor wellbeing and warrant follow-up screening.",
    }


def _score_phq9(answers: list[int]) -> dict[str, Any]:
    raw = sum(int(a) for a in answers)
    if raw <= 4:
        severity = "Minimal depression"
        crossed = False
    elif raw <= 9:
        severity = "Mild depression"
        crossed = False
    elif raw <= 14:
        severity = "Moderate depression"
        crossed = True
    elif raw <= 19:
        severity = "Moderately severe depression"
        crossed = True
    else:
        severity = "Severe depression"
        crossed = True
    item9 = int(answers[8]) if len(answers) >= 9 else 0
    return {
        "scale": "PHQ-9",
        "raw_score": raw,
        "severity": severity,
        "threshold_crossed": crossed,
        "crisis_flag": item9 >= 1,
        "clinical_note": "PHQ-9 cutoffs: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 mod-severe, 20-27 severe.",
    }


def _score_gad7(answers: list[int]) -> dict[str, Any]:
    raw = sum(int(a) for a in answers)
    if raw <= 4:
        severity = "Minimal anxiety"
        crossed = False
    elif raw <= 9:
        severity = "Mild anxiety"
        crossed = False
    elif raw <= 14:
        severity = "Moderate anxiety"
        crossed = True
    else:
        severity = "Severe anxiety"
        crossed = True
    return {
        "scale": "GAD-7",
        "raw_score": raw,
        "severity": severity,
        "threshold_crossed": crossed,
        "clinical_note": "GAD-7 cutoffs: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe.",
    }


def _recommend_level_of_care(scores: dict[str, dict[str, Any]]) -> str:
    phq9 = scores.get("PHQ-9") or {}
    gad7 = scores.get("GAD-7") or {}
    phq9_raw = int(phq9.get("raw_score") or 0)
    gad7_raw = int(gad7.get("raw_score") or 0)

    if phq9.get("crisis_flag") or phq9_raw >= 20:
        return "Inpatient evaluation"
    if phq9_raw >= 15 or gad7_raw >= 15:
        return "Intensive outpatient"
    if phq9_raw >= 10 or gad7_raw >= 10:
        return "Outpatient therapy"
    return "Self-help resources"


def _format_report(
    presenting_concern: str,
    scores: dict[str, dict[str, Any]],
    level_of_care: str,
    session_id: str,
) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "## Psychological Assessment Report",
        f"Date: {today}",
        f"Session ID: {session_id}",
        "",
        "### Presenting Concern",
        presenting_concern.strip() or "User initiated a structured mental health assessment.",
        "",
        "### Scale Results",
    ]
    for key in ("WHO-5", "PHQ-9", "GAD-7"):
        result = scores.get(key)
        if not result:
            continue
        lines.append(
            f"- {key}: raw {result['raw_score']} | {result['severity']}"
            f"{' (above clinical threshold)' if result.get('threshold_crossed') else ''}"
        )
    lines.extend(
        [
            "",
            "### Clinical Formulation",
            (
                "Self-reported responses across the administered validated screening instruments "
                "indicate a pattern consistent with the severity bands listed above. "
                "The findings should be interpreted within full clinical context including history, "
                "functional impact, and risk factors. No formal diagnosis is implied by this screening."
            ),
            "",
            "### Recommended Level of Care",
            f"- {level_of_care}",
            "",
            "### Disclaimer",
            DISCLAIMER,
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider lookup (NPI Registry + optional Google distance ranking)
# ---------------------------------------------------------------------------

def _extract_provider_rows(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    for item in raw_results:
        basic = item.get("basic") or {}
        addresses = item.get("addresses") or []
        taxonomies = item.get("taxonomies") or []
        sole_prop = item.get("is_sole_proprietor", basic.get("sole_proprietor"))
        if not addresses and (sole_prop is False or str(sole_prop).upper() == "NO"):
            continue

        location_address = None
        for address in addresses:
            if address.get("address_purpose") == "LOCATION":
                location_address = address
                break
        if location_address is None and addresses:
            location_address = addresses[0]
        if location_address is None:
            continue

        address_text = str(location_address.get("address_1") or "")
        if not address_text.strip():
            continue
        city = str(location_address.get("city") or "")
        state = str(location_address.get("state") or "")
        full_address = ", ".join(part for part in [address_text, city, state] if part)

        if basic.get("organization_name"):
            name = str(basic.get("organization_name"))
            credential = ""
        else:
            first = str(basic.get("first_name") or "")
            last = str(basic.get("last_name") or "")
            name = f"{first} {last}".strip()
            credential = str(basic.get("credential") or "")

        specialty = ""
        for taxonomy in taxonomies:
            if taxonomy.get("desc"):
                specialty = str(taxonomy.get("desc"))
                break

        providers.append(
            {
                "npi": str(item.get("number") or ""),
                "name": name,
                "credential": credential,
                "specialty": specialty,
                "address": full_address,
                "phone": str(location_address.get("telephone_number") or ""),
                "city": city,
                "state": state,
            }
        )
    return providers


def _search_npi_providers(
    city: str, state: str, taxonomy_terms: list[str]
) -> list[dict[str, Any]]:
    """Search NPI Registry, then enforce a strict state filter and prefer same-city.

    NPI Registry's ``city=`` parameter is fuzzy and can return providers from
    other cities/states. We re-filter on the practice-location address so we
    only show providers actually licensed in the user's requested state, then
    sort same-city matches first.
    """
    providers: list[dict[str, Any]] = []
    seen_npi: set[str] = set()
    requested_state = state.strip().upper()
    requested_city = city.strip().lower()

    for term in taxonomy_terms:
        try:
            response = requests.get(
                "https://npiregistry.cms.hhs.gov/api/",
                params={
                    "taxonomy_description": term,
                    "city": city,
                    "state": state,
                    "limit": 20,
                    "version": "2.1",
                },
                timeout=10,
            )
            payload = response.json() or {}
        except Exception:
            continue
        rows = _extract_provider_rows(payload.get("results") or [])
        for row in rows:
            npi = row.get("npi", "")
            if not npi or npi in seen_npi:
                continue
            row_state = str(row.get("state") or "").strip().upper()
            if requested_state and row_state and row_state != requested_state:
                continue  # strict state filter — never show out-of-state providers
            seen_npi.add(npi)
            providers.append(row)

    def _same_city_first(row: dict[str, Any]) -> int:
        return 0 if str(row.get("city") or "").strip().lower() == requested_city else 1

    providers.sort(key=_same_city_first)
    return providers


def _rank_by_google_distance(
    origin_text: str, providers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not providers:
        return []
    api_key = os.environ.get("GOOGLE_MAPS_KEY", "").strip()
    if not api_key:
        for provider in providers:
            provider["distance_km"] = None
        return providers
    try:
        destinations = "|".join(provider["address"] for provider in providers if provider.get("address"))
        response = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": origin_text,
                "destinations": destinations,
                "departure_time": "now",
                "key": api_key,
            },
            timeout=10,
        )
        payload = response.json() or {}
    except Exception:
        for provider in providers:
            provider["distance_km"] = None
        return providers

    rows = payload.get("rows") or []
    elements = rows[0].get("elements") if rows else []
    enriched: list[dict[str, Any]] = []
    for idx, provider in enumerate(providers):
        element = elements[idx] if idx < len(elements) else {}
        duration = ((element.get("duration_in_traffic") or {}).get("value"))
        distance_m = ((element.get("distance") or {}).get("value"))
        provider_copy = dict(provider)
        provider_copy["_duration_sort"] = (
            int(duration) if isinstance(duration, int) else 10**12
        )
        provider_copy["distance_km"] = (
            round((distance_m or 0) / 1000, 2) if isinstance(distance_m, int) else None
        )
        enriched.append(provider_copy)
    enriched.sort(key=lambda item: item["_duration_sort"])
    for provider in enriched:
        provider.pop("_duration_sort", None)
    return enriched


def _format_providers(providers: list[dict[str, Any]]) -> str:
    if not providers:
        return "I couldn't find any verified providers in that area through the NPI registry right now."
    lines = ["Here are verified mental health providers near you (from the US NPI Registry):", ""]
    for idx, provider in enumerate(providers, start=1):
        cred = f", {provider['credential']}" if provider.get("credential") else ""
        distance = (
            f" | ~{provider['distance_km']} km"
            if provider.get("distance_km") not in (None, -1, -1.0)
            else ""
        )
        specialty = f" | {provider['specialty']}" if provider.get("specialty") else ""
        phone = f" | Phone: {provider['phone']}" if provider.get("phone") else ""
        lines.append(
            f"{idx}. {provider['name']}{cred}{specialty}\n"
            f"   Address: {provider['address']}{distance}{phone}\n"
            f"   NPI: {provider['npi']}"
        )
    lines.append("")
    lines.append("Reply with the number (1, 2, or 3) of the provider you'd like to book with, "
                 "or type 'none' to skip.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Twilio SMS
# ---------------------------------------------------------------------------

def _send_twilio_sms(to_number: str, body: str) -> tuple[bool, str]:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.environ.get("TWILIO_FROM", "").strip()
    if not (sid and token and from_number):
        return False, "Twilio credentials not configured; SMS skipped."
    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            auth=HTTPBasicAuth(sid, token),
            data={"From": from_number, "To": to_number, "Body": body},
            timeout=10,
        )
        if 200 <= response.status_code < 300:
            return True, "SMS sent successfully."
        return False, f"Twilio responded with HTTP {response.status_code}: {response.text[:200]}"
    except Exception as exc:
        return False, f"Twilio request failed: {exc}"


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

_MENTION_PREFIX_RE = re.compile(r"^(?:@\S+\s*)+")


def _extract_text(msg: ChatMessage) -> str:
    """Pull all text from a ChatMessage and strip any leading @handle mentions.

    ASI:One forwards user messages to the agent prefixed with the agent handle,
    e.g. ``@orchestratoragent-2 San Francisco, CA``. We strip that so it does
    not leak into stored fields (presenting concern, city, etc.).
    """
    parts: list[str] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
    text = " ".join(parts).strip()
    text = _MENTION_PREFIX_RE.sub("", text).strip()
    return text


async def _send_reply(ctx: Context, sender: str, text: str, end_session: bool = True) -> None:
    content: list[Any] = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=content,
        ),
    )


def _load_flow(ctx: Context, session_id: str) -> dict[str, Any]:
    flow = ctx.storage.get(f"session:{session_id}:flow")
    return dict(flow) if flow else {"stage": "warm_open"}


def _save_flow(ctx: Context, session_id: str, flow: dict[str, Any]) -> None:
    ctx.storage.set(f"session:{session_id}:flow", flow)


def _set_state(ctx: Context, session_id: str, state: str) -> None:
    ctx.storage.set(f"session:{session_id}:state", state)


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------

def _handle_warm_open(flow: dict[str, Any], user_text: str) -> str:
    flow["presenting_concern"] = user_text
    flow["stage"] = "who5"
    flow["who5_index"] = 0
    flow["who5_answers"] = []
    flow["awaiting"] = True
    return (
        "Thank you for sharing that. I can help you complete a structured, validated "
        "mental health assessment. We'll start with the WHO-5 wellbeing scale, then move "
        "into the PHQ-9 depression screen.\n\n"
        f"{WHO5_QUESTIONS[0]}\n\n{WHO5_OPTIONS_HINT}"
    )


def _handle_who5(flow: dict[str, Any], user_text: str) -> str:
    answers = list(flow.get("who5_answers") or [])
    index = int(flow.get("who5_index") or 0)
    if flow.get("awaiting"):
        answers.append(_parse_who5_answer(user_text))
        index += 1
        flow["who5_answers"] = answers
        flow["who5_index"] = index

    if index < len(WHO5_QUESTIONS):
        flow["awaiting"] = True
        return f"{WHO5_QUESTIONS[index]}\n\n{WHO5_OPTIONS_HINT}"

    # WHO-5 complete -> always proceed to PHQ-9 (poor wellbeing always advances; otherwise still
    # confirms the depression screen, which is a safe default for an assessment tool).
    flow["stage"] = "phq9"
    flow["phq9_index"] = 0
    flow["phq9_answers"] = []
    flow["awaiting"] = True
    return (
        "Thank you. WHO-5 is complete. Now we'll move to the PHQ-9 depression screen.\n\n"
        f"{PHQ9_QUESTIONS[0]}\n\n{PHQ_GAD_OPTIONS_HINT}"
    )


def _handle_phq9(flow: dict[str, Any], user_text: str) -> tuple[str, bool]:
    """Returns (reply_text, crisis_triggered)."""
    answers = list(flow.get("phq9_answers") or [])
    index = int(flow.get("phq9_index") or 0)

    if flow.get("awaiting"):
        score = _parse_phq_gad_answer(user_text)
        answers.append(score)
        index += 1
        flow["phq9_answers"] = answers
        flow["phq9_index"] = index

        # Item 9 just answered -> item9 is at position 8 (0-indexed)
        if index == 9 and answers[8] >= 1:
            flow["stage"] = "crisis"
            flow["awaiting"] = False
            flow["crisis_trigger"] = "PHQ-9 item 9 indicated thoughts of self-harm."
            return CRISIS_RESPONSE_TEXT, True

    if index < len(PHQ9_QUESTIONS):
        flow["awaiting"] = True
        return f"{PHQ9_QUESTIONS[index]}\n\n{PHQ_GAD_OPTIONS_HINT}", False

    # PHQ-9 complete. Decide whether to continue to GAD-7.
    phq9_score = _score_phq9(answers)
    flow["scores"] = flow.get("scores") or {}
    flow["scores"]["PHQ-9"] = phq9_score

    if phq9_score["raw_score"] >= 10:
        flow["stage"] = "gad7"
        flow["gad7_index"] = 0
        flow["gad7_answers"] = []
        flow["awaiting"] = True
        return (
            f"Thank you. PHQ-9 is complete (score {phq9_score['raw_score']} - "
            f"{phq9_score['severity']}). Because depression symptoms are at or above the moderate "
            "threshold, I'll also administer the GAD-7 anxiety screen.\n\n"
            f"{GAD7_QUESTIONS[0]}\n\n{PHQ_GAD_OPTIONS_HINT}"
        ), False

    # PHQ-9 < 10 -> skip GAD-7, go straight to report
    flow["stage"] = "report_offer"
    flow["awaiting"] = False
    return (
        f"Thank you. PHQ-9 is complete (score {phq9_score['raw_score']} - "
        f"{phq9_score['severity']}). Would you like me to generate your assessment report now? (yes/no)"
    ), False


def _handle_gad7(flow: dict[str, Any], user_text: str) -> str:
    answers = list(flow.get("gad7_answers") or [])
    index = int(flow.get("gad7_index") or 0)
    if flow.get("awaiting"):
        answers.append(_parse_phq_gad_answer(user_text))
        index += 1
        flow["gad7_answers"] = answers
        flow["gad7_index"] = index

    if index < len(GAD7_QUESTIONS):
        flow["awaiting"] = True
        return f"{GAD7_QUESTIONS[index]}\n\n{PHQ_GAD_OPTIONS_HINT}"

    gad7_score = _score_gad7(answers)
    flow["scores"] = flow.get("scores") or {}
    flow["scores"]["GAD-7"] = gad7_score
    flow["stage"] = "report_offer"
    flow["awaiting"] = False
    return (
        f"Thank you. GAD-7 is complete (score {gad7_score['raw_score']} - "
        f"{gad7_score['severity']}). All scales are now done. "
        "Would you like me to generate your assessment report now? (yes/no)"
    )


def _build_report_and_offer_booking(flow: dict[str, Any], session_id: str) -> str:
    scores = dict(flow.get("scores") or {})

    # Backfill WHO-5 from raw answers if not already scored.
    who5_answers = flow.get("who5_answers") or []
    if "WHO-5" not in scores and len(who5_answers) == 5:
        scores["WHO-5"] = _score_who5(who5_answers)

    level_of_care = _recommend_level_of_care(scores)
    flow["level_of_care"] = level_of_care
    flow["scores"] = scores

    report_text = _format_report(
        presenting_concern=flow.get("presenting_concern") or "",
        scores=scores,
        level_of_care=level_of_care,
        session_id=session_id,
    )
    flow["stage"] = "booking_offer"
    flow["awaiting"] = False

    follow_up = (
        f"\n\nBased on these results, the recommended level of care is: **{level_of_care}**.\n"
        "Would you like me to help you find a licensed mental health professional near you? (yes/no)"
    )
    return report_text + follow_up


def _start_booking_or_close(flow: dict[str, Any], user_text: str) -> tuple[str, bool]:
    """Handle yes/no after report. Returns (reply, end_session)."""
    if _is_yes(user_text):
        flow["stage"] = "booking_location"
        flow["awaiting"] = True
        return (
            "Great. To search for verified providers, please share your **city and 2-letter state code** "
            "(US only, e.g. 'San Francisco, CA' or 'Boston, MA'). The lookup uses the public NPI Registry."
        ), False
    if _is_no(user_text):
        flow["stage"] = "session_close"
        flow["awaiting"] = False
        return (
            "Understood. Your assessment is saved for this session. "
            "Please consider sharing the report with a licensed mental health professional. "
            "Take care of yourself."
        ), True
    flow["awaiting"] = False
    return "Please reply 'yes' if you would like provider recommendations, or 'no' to close this session.", False


def _handle_booking_location(flow: dict[str, Any], user_text: str, session_id: str, ctx: Context) -> str:
    cleaned = _MENTION_PREFIX_RE.sub("", user_text).strip()
    parts = [piece.strip() for piece in cleaned.split(",") if piece.strip()]
    if len(parts) < 2:
        return (
            "Please share both city and 2-letter state code, separated by a comma. "
            "Example: 'San Francisco, CA'."
        )
    city = re.sub(r"^@\S+\s*", "", parts[0]).strip()
    state = re.sub(r"[^A-Za-z]", "", parts[1]).upper()[:2]
    if len(state) != 2 or not city:
        return (
            "Please send the location as 'City, ST' where ST is the 2-letter US state code "
            "(e.g. 'Boston, MA'). Sample valid inputs: 'San Francisco, CA', 'Austin, TX'."
        )

    flow["booking"] = {"city": city, "state": state}
    level = flow.get("level_of_care") or "Outpatient therapy"
    taxonomy_codes = TAXONOMY_MAP.get(level, TAXONOMY_MAP["Outpatient therapy"])

    if not taxonomy_codes:
        flow["stage"] = "session_close"
        flow["awaiting"] = False
        resources = "\n".join(f"- {res}" for res in CURATED_SELF_HELP_RESOURCES)
        return (
            f"Based on your scores ({level}), you may benefit from self-help resources first. "
            f"Here are vetted options:\n\n{resources}\n\nTake care."
        )

    providers = _search_npi_providers(city, state, taxonomy_codes)
    providers = _rank_by_google_distance(f"{city}, {state}", providers)[:3]
    ctx.storage.set(f"session:{session_id}:providers", providers)
    flow["booking"]["providers"] = providers

    if not providers:
        flow["stage"] = "session_close"
        flow["awaiting"] = False
        resources = "\n".join(f"- {res}" for res in CURATED_SELF_HELP_RESOURCES)
        return (
            f"I couldn't find verified NPI-registered providers in {city}, {state} for the "
            f"{level} level. Here are some general resources you can use right now:\n\n{resources}"
        )

    flow["stage"] = "booking_select"
    flow["awaiting"] = True
    return _format_providers(providers)


def _handle_booking_select(flow: dict[str, Any], user_text: str) -> str:
    providers = list((flow.get("booking") or {}).get("providers") or [])
    lowered = user_text.lower().strip()
    if "none" in lowered or _is_no(user_text):
        flow["stage"] = "session_close"
        flow["awaiting"] = False
        return "No problem. Your assessment results are saved. Take care."

    digit_match = re.search(r"[1-9]", user_text)
    if not digit_match:
        return "Please reply with the number (1, 2, or 3) of the provider you'd like to book with."
    selected_index = int(digit_match.group(0)) - 1
    if not (0 <= selected_index < len(providers)):
        return f"Please choose a number between 1 and {len(providers)}."

    flow["booking"]["selected_index"] = selected_index
    flow["booking"]["selected_provider"] = providers[selected_index]
    flow["stage"] = "booking_datetime"
    flow["awaiting"] = True
    selected = providers[selected_index]
    return (
        f"You selected: {selected['name']} {('(' + selected['credential'] + ')') if selected.get('credential') else ''}.\n"
        "What date and time would you prefer for the appointment? "
        "(e.g. 'Next Tuesday at 3pm' or '2026-05-05 14:00')"
    )


def _handle_booking_datetime(flow: dict[str, Any], user_text: str) -> str:
    flow["booking"]["preferred_datetime"] = user_text.strip()
    flow["stage"] = "booking_phone"
    flow["awaiting"] = True
    return (
        "Thanks. Please share your phone number in international format "
        "(e.g. +1 555 123 4567 or +91 98765 43210) so we can text you a confirmation."
    )


def _handle_booking_phone(flow: dict[str, Any], user_text: str, ctx: Context, session_id: str) -> str:
    phone_match = re.search(r"\+?\d[\d\s\-()]{6,}", user_text)
    if not phone_match:
        return "I couldn't read a phone number. Please send it in international format like +1 555 123 4567."
    phone = re.sub(r"[^\d+]", "", phone_match.group(0))
    if not phone.startswith("+"):
        phone = "+" + phone

    booking = flow.get("booking") or {}
    provider = booking.get("selected_provider") or {}
    preferred_dt = booking.get("preferred_datetime") or "TBD"

    body = (
        f"Mental Health Check-in: appointment requested with "
        f"{provider.get('name', 'provider')}"
        f"{(' (' + provider.get('credential', '') + ')') if provider.get('credential') else ''} for "
        f"{preferred_dt}. Address: {provider.get('address', 'see provider')}. "
        f"Phone: {provider.get('phone', 'N/A')}. Reply CANCEL to cancel."
    )
    sent, sms_status = _send_twilio_sms(phone, body)

    booking["status"] = "confirmed" if sent else "pending"
    booking["sms_status"] = sms_status
    booking["user_phone"] = phone
    flow["booking"] = booking
    ctx.storage.set(f"session:{session_id}:booking", booking)
    flow["stage"] = "session_close"
    flow["awaiting"] = False

    confirmation = (
        f"Booking summary:\n"
        f"- Provider: {provider.get('name', 'N/A')}\n"
        f"- Address: {provider.get('address', 'N/A')}\n"
        f"- When: {preferred_dt}\n"
        f"- Your phone: {phone}\n"
        f"- Status: {booking['status']}\n"
        f"- SMS: {sms_status}\n\n"
        "Please call the provider directly to lock the slot. Take care of yourself."
    )
    return confirmation


# ---------------------------------------------------------------------------
# Chat protocol handlers
# ---------------------------------------------------------------------------

@chat_proto.on_message(model=ChatMessage, allow_unverified=True)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """Hosted chat entrypoint compatible with ASI:One/Agentverse chat protocol."""
    # Required acknowledgement for AgentChatProtocol.
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    user_text = _extract_text(msg)
    if not user_text:
        return

    session_key = f"sender_session:{sender}"
    session_id = ctx.storage.get(session_key)
    if not session_id:
        session_id = str(uuid4())
        ctx.storage.set(session_key, session_id)

    flow = _load_flow(ctx, session_id)
    stage = str(flow.get("stage") or "warm_open")

    try:
        if stage == "warm_open":
            reply = _handle_warm_open(flow, user_text)
            _set_state(ctx, session_id, "WHO5_SCREEN")
        elif stage == "who5":
            reply = _handle_who5(flow, user_text)
            _set_state(ctx, session_id, "PHQ9" if flow.get("stage") == "phq9" else "WHO5_SCREEN")
        elif stage == "phq9":
            reply, crisis = _handle_phq9(flow, user_text)
            if crisis:
                _set_state(ctx, session_id, "CRISIS")
            else:
                _set_state(
                    ctx,
                    session_id,
                    "GAD7" if flow.get("stage") == "gad7" else (
                        "REPORT_OFFER" if flow.get("stage") == "report_offer" else "PHQ9"
                    ),
                )
        elif stage == "gad7":
            reply = _handle_gad7(flow, user_text)
            _set_state(ctx, session_id, "REPORT_OFFER" if flow.get("stage") == "report_offer" else "GAD7")
        elif stage == "report_offer":
            if _is_yes(user_text):
                reply = _build_report_and_offer_booking(flow, session_id)
                _set_state(ctx, session_id, "REPORT_READY")
            elif _is_no(user_text):
                flow["stage"] = "session_close"
                _set_state(ctx, session_id, "SESSION_CLOSE")
                reply = (
                    "No problem. Your responses are stored for this session. "
                    "When you're ready, ask me to 'generate my report' or come back for booking. "
                    "Take care."
                )
            else:
                reply = "Please reply 'yes' to generate your report, or 'no' to close the session."
        elif stage == "booking_offer":
            reply, end_now = _start_booking_or_close(flow, user_text)
            if end_now:
                _set_state(ctx, session_id, "SESSION_CLOSE")
            else:
                _set_state(
                    ctx,
                    session_id,
                    "BOOKING" if flow.get("stage") == "booking_location" else "REPORT_READY",
                )
        elif stage == "booking_location":
            reply = _handle_booking_location(flow, user_text, session_id, ctx)
            _set_state(ctx, session_id, "BOOKING")
        elif stage == "booking_select":
            reply = _handle_booking_select(flow, user_text)
            _set_state(ctx, session_id, "BOOKING")
        elif stage == "booking_datetime":
            reply = _handle_booking_datetime(flow, user_text)
            _set_state(ctx, session_id, "BOOKING")
        elif stage == "booking_phone":
            reply = _handle_booking_phone(flow, user_text, ctx, session_id)
            _set_state(ctx, session_id, "SESSION_CLOSE")
        elif stage == "crisis":
            # Stay in crisis mode; gently re-anchor the user without restarting flow.
            reply = (
                "I'm staying with you. Please use one of the helplines I shared, or contact local "
                "emergency services if you are in immediate danger.\n\n" + CRISIS_RESPONSE_TEXT
            )
            _set_state(ctx, session_id, "CRISIS")
        elif stage == "session_close":
            # Allow user to ask for the report again after closing.
            if "report" in user_text.lower():
                reply = _build_report_and_offer_booking(flow, session_id)
                _set_state(ctx, session_id, "REPORT_READY")
            elif _is_yes(user_text) or "book" in user_text.lower() or "find" in user_text.lower():
                flow["stage"] = "booking_location"
                flow["awaiting"] = True
                reply = (
                    "Sure. Please share your city and 2-letter state code, e.g. 'San Francisco, CA'."
                )
                _set_state(ctx, session_id, "BOOKING")
            else:
                reply = (
                    "Your session is closed. Type 'start' to begin a new assessment, "
                    "'generate my report' to view the report again, or 'find a provider' to look up nearby help."
                )
                if "start" in user_text.lower():
                    flow = {"stage": "warm_open"}
                    _set_state(ctx, session_id, "WARM_OPEN")
                    reply = (
                        "Starting a fresh assessment. Tell me what's on your mind, or what you'd like "
                        "to be screened for today."
                    )
        else:
            flow["stage"] = "warm_open"
            reply = (
                "Let's begin. Tell me what's on your mind, or what you'd like to be screened for today."
            )
    except Exception as exc:
        ctx.logger.error(f"Pipeline error for session {session_id}: {exc}")
        reply = (
            "Sorry, something went wrong while processing that. "
            "Please try rephrasing your last message."
        )

    _save_flow(ctx, session_id, flow)
    await _send_reply(ctx, sender, reply, end_session=True)


@chat_proto.on_message(model=ChatAcknowledgement, allow_unverified=True)
async def handle_chat_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
    """Accept acknowledgements to satisfy protocol interactions."""
    ctx.logger.info(f"Received chat acknowledgement from {sender}")


agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__":
    agent.run()
