# Single-Agent Architecture

This project runs as a single hosted orchestrator agent on Agentverse.

## System Diagram

```mermaid
flowchart TD
    U[ASI:One User Message] --> P[AgentChatProtocol Handler]
    P --> S[Session State Manager ctx.storage]
    S --> E[Flow Engine Deterministic FSM]

    E --> Q[Question Engine WHO-5 PHQ-9 GAD-7]
    E --> C[Crisis Engine]
    E --> R[Report Engine]
    E --> D[Provider Discovery Engine]
    E --> B[Booking Engine SMS]

    D --> NPI[NPI Registry API]
    D --> GDM[Google Distance Matrix API optional]
    B --> TW[Twilio REST API optional]

    E --> O[Chat Response Builder]
    O --> U
```

## Runtime State Machine

- `WARM_OPEN`
- `WHO5_SCREEN`
- `PHQ9`
- `GAD7` (conditional)
- `CRISIS` (interrupt state)
- `REPORT_READY`
- `BOOKING`
- `SESSION_CLOSE`

All state is persisted in `ctx.storage` using session-scoped keys.

## Key Design Decisions

- Single runtime agent (no inter-agent routing at chat time)
- Deterministic scoring and branching rules
- Crisis interruption has highest priority
- Graceful API degradation:
  - No Google Maps key: provider list still returns without distance ranking
  - No Twilio credentials: booking summary still completes with SMS skipped status

## External Integrations

- NPI Registry API: provider lookup
- Google Distance Matrix API (optional): distance/time ranking
- Twilio REST API (optional): SMS confirmation
