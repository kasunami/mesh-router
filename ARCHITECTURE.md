# MeshRouter architecture

MeshRouter is an OpenAI-compatible routing and control-plane service for a
heterogeneous set of model-serving lanes. It is one component of a larger mesh,
not a general-purpose job orchestrator or a model server.

## Request flow

1. A client or MeshComputer submits an OpenAI-style request, optionally with
   worker, lane-type, or exact-lane routing hints.
2. MeshRouter validates the request and resolves an eligible candidate from
   configured inventory, model compatibility, health observations, and current
   MeshWorker state.
3. For MeshWorker-managed lanes, MeshRouter can request the desired model over
   the Kafka control plane before using the gRPC data plane.
4. MeshRouter forwards the request to the selected HTTP or gRPC backend.
5. The response is returned with request and resolved-route metadata. Best-
   effort performance observations and request lifecycle state are recorded
   outside the model output.

## Responsibility boundaries

- **MeshComputer** owns higher-level job orchestration, dependencies, and job
  state.
- **MeshRouter** owns route resolution, lane eligibility, proxying, request
  lifecycle records, and bounded recovery around a routed request.
- **MeshWorker** owns host-level service and model lifecycle and reports its
  actual state.
- **Model-serving lanes** perform inference; they do not choose their own route.
- **MeshBrain** stores shared operational context and agent handoffs. It is not
  part of the model-output decision loop.

## State ownership

PostgreSQL stores configured hosts, lanes, models, compatibility policy,
request/lease records, MeshWorker state, and performance observations. A
separate PostgreSQL database may be configured for MeshWorker state.

Redis is an optional low-latency cache for short-lived runtime state. Cached
entries expire and do not replace PostgreSQL as the durable audit and fallback
store. Kafka carries MeshWorker commands, responses, state, heartbeats,
cancellations, and dead-letter events. Downstream inference services remain the
authority for the actual inference result.

## Trust boundaries

Client input, routing hints, worker-reported state, and downstream model output
cross separate trust boundaries. MeshRouter validates structured request and
state data before using it for placement. Credentials and internal tokens are
provided at runtime through environment variables or the deployment secret
manager; they are not stored in inventory metadata or source control.

The service does not by itself make an untrusted network safe. Deployments must
apply appropriate network policy, TLS termination, and authentication around
client and operator-facing endpoints.

## Fail-closed behavior

An exact lane pin is a requirement, not a preference: the request fails when
that lane is missing, unhealthy, incompatible, or unable to serve the model.
Candidates that fail readiness, backend-compatibility, suspension, or model-
viability checks are excluded. Placeholder production secrets are rejected
unless development mode is explicitly enabled.

Less-specific hints may permit another eligible candidate. That fallback is a
routing policy decision and is kept separate from model-generated content.

## Recovery paths

MeshRouter uses bounded timeouts, readiness checks, model-load commands, and
explicit error responses. Failed or canceled requests are not treated as valid
performance evidence. MeshWorker heartbeats and state overlays allow a stale or
unavailable lane to leave the candidate set without asking a model to diagnose
or repair itself.

Live certification is operator-initiated. The documented default is a dry run,
so evaluation does not contact private workers or mutate a deployment.
