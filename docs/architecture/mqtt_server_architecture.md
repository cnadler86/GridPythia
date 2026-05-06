# GridPythia Server Architecture

## Current State (After Review)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           GridPythia Package                            │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
│   config/   │ prediction/ │optimization/│ simulation/ │   services/    │
│             │             │             │             │                │
│ AppConfig   │ Prediction  │ Linear-     │ Grid-       │ GridPythia-    │
│ Prediction- │ PredData    │ Optimizer   │ Simulation  │ Service        │
│ Config      │ Provider-   │ Linear-     │ Inverter-   │ Prediction-    │
│ Optim-      │ Registry    │ Solution    │ Base        │ Cache          │
│ Config      │             │             │ Battery     │                │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │     server/       │
                          │                   │
                          │ FastAPI + Routers │
                          │ (REST API + GUI)  │
                          └───────────────────┘
```

## Target Architecture for MQTT Integration

```text
                    ┌──────────────────────────────────────┐
                    │           External World             │
                    │                                      │
                    │  ┌──────────┐      ┌─────────────┐   │
                    │  │ Inverters│      │ Dashboard   │   │
                    │  │ (MQTT)   │      │ (WebSocket) │   │
                    │  └────┬─────┘      └──────┬──────┘   │
                    └───────┼──────────────────┼───────────┘
                            │                  │
                    ┌───────▼──────────────────▼───────────┐
                    │           Gateway Layer              │
                    │                                      │
                    │  ┌──────────┐      ┌─────────────┐   │
                    │  │  MQTT    │      │   FastAPI   │   │
                    │  │ Handler  │      │   Server    │   │
                    │  └────┬─────┘      └──────┬──────┘   │
                    └───────┼──────────────────┼───────────┘
                            │                  │
                    ┌───────▼──────────────────▼───────────┐
                    │         Message Bus (Internal)       │
                    │                                      │
                    │  ┌─────────────────────────────────┐ │
                    │  │     EventBus / Pub-Sub          │ │
                    │  │                                 │ │
                    │  │  Topics:                        │ │
                    │  │  - inverter/+/status            │ │
                    │  │  - prediction/updated           │ │
                    │  │  - optimization/result          │ │
                    │  │  - schedule/inverter/+          │ │
                    │  └─────────────────────────────────┘ │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │        Application Services          │
                    │                                      │
                    │  ┌─────────────────────────────────┐ │
                    │  │       GridPythiaService         │ │
                    │  │   (config, providers, optimizer) │ │
                    │  └─────────────────────────────────┘ │
                    │                                      │
                    │  ┌─────────────────────────────────┐ │
                    │  │       InverterCoordinator       │ │
                    │  │   (state mgmt, scheduling)      │ │
                    │  └─────────────────────────────────┘ │
                    │                                      │
                    │  ┌─────────────────────────────────┐ │
                    │  │       SchedulerService          │ │
                    │  │   (periodic optimization)       │ │
                    │  └─────────────────────────────────┘ │
                    └──────────────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │           Domain Layer               │
                    │                                      │
                    │  prediction/ │ optimization/ │ simulation/
                    │  (providers) │ (solver)      │ (devices)
                    └──────────────────────────────────────┘
```

## Key Components

### 1. EventBus (New)

Central pub-sub for internal communication. Decouples MQTT, HTTP, and core services.

```python
# GridPythia/events/bus.py
class EventBus:
    async def publish(self, topic: str, payload: Any) -> None: ...
    async def subscribe(self, pattern: str, handler: Callable) -> None: ...
```

### 2. InverterCoordinator (New)

Manages real-time inverter state and coordinates mode changes.

```python
# GridPythia/coordination/inverter_coordinator.py
class InverterCoordinator:
    """Tracks inverter states and dispatches schedules."""
    
    def update_inverter_status(self, device_id: str, soc: float, mode: InverterMode) -> None
    def apply_schedule(self, plan: InverterPlan) -> None
    def get_current_state(self, device_id: str) -> InverterState
```

### 3. SchedulerService (New)

Triggers periodic prediction fetches and optimizations.

```python
# GridPythia/services/scheduler.py
class SchedulerService:
    """Periodic optimization runner."""
    
    async def start(self) -> None  # Start background scheduler
    async def stop(self) -> None   # Graceful shutdown
    async def trigger_optimization(self) -> LinearSolution
```

### 4. MQTT Handler (New)

Bridges external MQTT to internal EventBus.

```python
# GridPythia/gateway/mqtt.py
class MQTTGateway:
    """MQTT client that maps external topics to EventBus."""
    
    async def connect(self, broker: str) -> None
    async def on_message(self, topic: str, payload: bytes) -> None
```

## MQTT Topic Structure

```text
gridpythia/
├── inverter/
│   ├── {device_id}/
│   │   ├── status          # Inverter publishes: {"soc": 50.2, "mode": 2}
│   │   └── command         # Server publishes: {"mode": 3, "power_w": 1000}
├── prediction/
│   ├── status              # {"last_fetch": "...", "next_fetch": "..."}
│   └── data                # Full PredictionData as JSON (for dashboard)
├── optimization/
│   ├── status              # {"last_run": "...", "objective": "cost"}
│   └── result              # LinearSolution summary
└── config/
    └── reload              # Trigger config reload
```

## Data Flow: Optimization Cycle

```text
1. SchedulerService triggers every N minutes
       │
       ▼
2. Collect current inverter states from InverterCoordinator
       │
       ▼
3. GridPythiaService.fetch_predictions()
       │
       ▼
4. GridPythiaService.optimize(soc=current_socs, states=current_inverter_states)
       │
       ▼
5. InverterCoordinator.apply_schedule(solution)
       │
       ▼
6. EventBus.publish("optimization/result", summary, inverter_cmds)
       │
       ▼
7. MQTTGateway publishes commands to inverters
```

## Implementation Order

### Phase 1: Event Infrastructure

1. Create `GridPythia/events/` package with EventBus
2. Add event types for inverter status, predictions, optimization
3. Unit tests for EventBus

### Phase 2: Coordination Layer

1. Create InverterCoordinator with in-memory state
2. Wire to GridPythiaService
3. Add InverterState dataclass

### Phase 3: MQTT Gateway

1. Add asyncio-mqtt dependency
2. Implement MQTTGateway with reconnection logic
3. Bridge MQTT topics to EventBus

### Phase 4: Scheduler

1. Implement SchedulerService with APScheduler or asyncio tasks
2. Connect to GridPythiaService and InverterCoordinator
3. Add manual trigger endpoint

### Phase 5: Dashboard Integration

1. Add WebSocket endpoint for real-time updates
2. Subscribe to EventBus and push to WebSocket clients
3. Update frontend to use WebSocket

## Configuration Extension

```yaml
# config.yaml additions for MQTT
server:
  mqtt:
    enabled: true
    broker: "mqtt://localhost:1883"
    client_id: "gridpythia"
    username: ""
    password: ""
    topics:
      inverter_status: "gridpythia/inverter/+/status"
      inverter_command: "gridpythia/inverter/{device_id}/command"
  
  scheduler:
    optimization_interval_minutes: 15
  
  dashboard:
    websocket_enabled: true
```

## Dependencies to Add

```toml
# pyproject.toml
[project.dependencies]
asyncio-mqtt = ">=0.16"      # MQTT client
schedule = ">=1.2"           # Background scheduling
websockets = ">=12.0"        # WebSocket for dashboard
```
