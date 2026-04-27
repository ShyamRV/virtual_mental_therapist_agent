# Virtual Mental Therapist Agent

AI-powered single-agent mental health assistant for structured screening, report generation, crisis handling, and provider booking support.

## What It Does

- Runs WHO-5, PHQ-9, and GAD-7 assessments in chat
- Detects crisis risk and immediately returns emergency resources
- Generates a structured psychological assessment summary
- Finds licensed providers via the NPI Registry
- Supports appointment intent + Twilio SMS confirmation flow

## Main Runtime File

- `mental_health_agent/hosted/orchestrator_agent.py`

## Environment Variables

Set these in Agentverse Secrets for the orchestrator:

- `AGENT_SEED` (required)
- `GOOGLE_MAPS_KEY` (optional, provider distance ranking)
- `TWILIO_ACCOUNT_SID` (optional, SMS)
- `TWILIO_AUTH_TOKEN` (optional, SMS)
- `TWILIO_FROM` (optional, SMS sender)

## Deployment

1. Open Agentverse and create a blank agent.
2. Paste `mental_health_agent/hosted/orchestrator_agent.py` into `agent.py`.
3. Add required secrets.
4. Run the agent and enable chat protocol.

## Architecture

See `ARCHITECTURE.md` for the single-agent system design and flow diagram.
