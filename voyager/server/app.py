"""Read-only FastAPI application for Voyager saved replays."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from voyager.replay.catalog import ReplayCatalog, catalog_entry_payload
from voyager.replay.loader import (
    CorruptReplayError,
    IncompleteReplayError,
    ReplayError,
    ReplayLoader,
    UnsupportedReplayError,
)
from voyager.replay.schema import REPLAY_SCHEMA_VERSION
from voyager.replay.serialization import canonical_json_bytes, sha256_value

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def create_app(
    *,
    replay_roots: list[tuple[str, str | Path]] | None = None,
    frontend_dir: str | Path | None = None,
    cache_size: int | None = None,
) -> FastAPI:
    roots: list[tuple[str, str | Path]] = replay_roots or list(_configured_roots())
    catalog = ReplayCatalog(
        roots,
        cache_size=cache_size or int(os.environ.get("VOYAGER_REPLAY_CACHE_SIZE", "8")),
    )
    frontend = (
        Path(frontend_dir).resolve()
        if frontend_dir is not None
        else Path(
            os.environ.get("VOYAGER_FRONTEND_DIR", str(REPOSITORY_ROOT / "web/dist"))
        ).resolve()
    )
    app = FastAPI(title="Voyager Replay Platform", version="1.0.0")
    app.state.catalog = catalog

    @app.exception_handler(ReplayError)
    async def replay_error_handler(_request: Request, exc: ReplayError) -> JSONResponse:
        status = 422
        code = "replay_error"
        if isinstance(exc, CorruptReplayError):
            code = "corrupt_replay"
        elif isinstance(exc, IncompleteReplayError):
            code = "incomplete_replay"
        elif isinstance(exc, UnsupportedReplayError):
            code = "unsupported_replay"
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": str(exc)}})

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "replays": len(catalog.scan())}

    @app.get("/api/v1/meta")
    def meta() -> dict[str, object]:
        return {
            "api_version": "1.0.0",
            "replay_version": REPLAY_SCHEMA_VERSION,
            "capabilities": [
                "catalog",
                "random_access",
                "events",
                "metrics",
                "comparison",
                "legacy_stage6a",
            ],
        }

    @app.get("/api/v1/replays")
    def replays(
        policy: str | None = None,
        kind: str | None = None,
        scenario: str | None = None,
        seed: int | None = None,
        inference_mode: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        min_survivors: int | None = Query(default=None, ge=0),
        max_survivors: int | None = Query(default=None, ge=0),
        achievement: str | None = None,
        cursor: str | None = None,
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        entries = catalog.query(
            policy=policy,
            kind=kind,
            scenario=scenario,
            seed=seed,
            inference_mode=inference_mode,
            status=status,
            tag=tag,
            min_survivors=min_survivors,
            max_survivors=max_survivors,
            achievement=achievement,
        )
        start = _decode_cursor(cursor) if cursor else 0
        page = entries[start : start + limit]
        next_cursor = _encode_cursor(start + limit) if start + limit < len(entries) else None
        return {
            "items": [catalog_entry_payload(entry) for entry in page],
            "next_cursor": next_cursor,
            "total": len(entries),
        }

    @app.get("/api/v1/replays/{replay_id}")
    def replay(replay_id: str, request: Request) -> Response:
        entry = _entry_or_404(catalog, replay_id)
        payload = {
            **entry.manifest.model_dump(mode="json"),
            "catalog_source": entry.source,
            "available_series": [
                str(registry_entry["id"])
                for registry_entry in entry.manifest.registries.get("metrics", [])
                if "id" in registry_entry
            ],
        }
        return _etag_json(request, payload, entry.manifest.manifest_sha256)

    @app.get("/api/v1/replays/{replay_id}/initial")
    def initial(replay_id: str, request: Request) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        return _etag_json(request, loader.initial())

    @app.get("/api/v1/replays/{replay_id}/state/{tick}")
    def state(replay_id: str, tick: int, request: Request) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        try:
            payload = loader.state_at(tick)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail=str(exc)) from exc
        return _etag_json(request, payload)

    @app.get("/api/v1/replays/{replay_id}/timeline")
    def timeline(
        replay_id: str,
        request: Request,
        start: int = Query(ge=0),
        end: int = Query(ge=0),
    ) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        try:
            records = loader.timeline(start, end)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail=str(exc)) from exc
        return _stream_json(request, {"start": start, "end": end, "records": records})

    @app.get("/api/v1/replays/{replay_id}/events")
    def events(
        replay_id: str,
        request: Request,
        start: int = Query(default=0, ge=0),
        end: int | None = Query(default=None, ge=0),
        type: str | None = None,
        agent: str | None = None,
    ) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        try:
            payload = loader.events(start, end, event_type=type, agent=agent)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail=str(exc)) from exc
        return _stream_json(request, {"events": payload})

    @app.get("/api/v1/replays/{replay_id}/metrics")
    def metrics(
        replay_id: str,
        request: Request,
        series: str | None = None,
        agent: str | None = None,
        role: str | None = None,
    ) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        payload = loader.metrics()
        if series is not None:
            global_series = payload.get("global", {})
            if series not in global_series:
                raise HTTPException(status_code=404, detail=f"Unknown metric series {series!r}.")
            payload = {"global": {series: global_series[series]}}
        if agent is not None:
            payload = {
                **payload,
                "reward_components_by_agent": {
                    agent: payload.get("reward_components_by_agent", {}).get(agent, {})
                },
            }
        if role is not None:
            payload = {
                **payload,
                "reward_components_by_role": {
                    role: payload.get("reward_components_by_role", {}).get(role, {})
                },
            }
        return _etag_json(request, payload)

    @app.get("/api/v1/replays/{replay_id}/camera")
    def camera(replay_id: str, request: Request) -> Response:
        loader = _loader_or_404(catalog, replay_id)
        return _etag_json(request, loader.camera())

    @app.get("/api/v1/compare")
    def compare(left: str, right: str, request: Request) -> Response:
        left_loader = _loader_or_404(catalog, left)
        right_loader = _loader_or_404(catalog, right)
        incompatibilities = _comparison_incompatibilities(left_loader, right_loader)
        if incompatibilities:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "incompatible_replays",
                        "message": "The selected runs cannot be synchronized.",
                        "reasons": incompatibilities,
                    }
                },
            )
        left_summary = left_loader.manifest.terminal_summary
        right_summary = right_loader.manifest.terminal_summary
        metric_names = ("survivors", "deaths", "dense_return", "shelter_progress")
        payload = {
            "left": catalog_entry_payload(catalog.get(left)),
            "right": catalog_entry_payload(catalog.get(right)),
            "terminal_deltas": {
                name: float(right_summary[name]) - float(left_summary[name])
                for name in metric_names
            },
            "series": {
                name: _series_delta(
                    left_loader.metrics().get("global", {}).get(name, []),
                    right_loader.metrics().get("global", {}).get(name, []),
                )
                for name in ("survivors", "camp_food", "shelter_progress")
            },
        }
        return _etag_json(request, payload)

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str) -> Response:
        if spa_path.startswith(("api/", "healthz")):
            raise HTTPException(status_code=404, detail="Not found.")
        requested = (frontend / spa_path).resolve()
        if frontend in requested.parents and requested.is_file():
            return FileResponse(requested)
        index = frontend / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "frontend_not_built",
                    "message": "Build web/ before starting the production server.",
                }
            },
        )

    return app


def _entry_or_404(catalog: ReplayCatalog, replay_id: str) -> Any:
    try:
        return catalog.get(replay_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Unknown replay {replay_id!r}.") from exc


def _loader_or_404(catalog: ReplayCatalog, replay_id: str) -> ReplayLoader:
    entry = _entry_or_404(catalog, replay_id)
    return ReplayLoader(entry.directory, cache_size=catalog.cache_size)


def _etag_json(request: Request, payload: object, etag: str | None = None) -> Response:
    resolved = etag or sha256_value(payload)
    quoted = f'"{resolved}"'
    if request.headers.get("if-none-match") == quoted:
        return Response(status_code=304, headers={"ETag": quoted})
    return Response(
        content=canonical_json_bytes(payload),
        media_type="application/json",
        headers={"ETag": quoted, "Cache-Control": "public, max-age=31536000, immutable"},
    )


def _stream_json(request: Request, payload: object) -> Response:
    data = canonical_json_bytes(payload)
    etag = sha256_value(payload)
    quoted = f'"{etag}"'
    if request.headers.get("if-none-match") == quoted:
        return Response(status_code=304, headers={"ETag": quoted})

    def chunks() -> Any:
        for start in range(0, len(data), 64 * 1024):
            yield data[start : start + 64 * 1024]

    return StreamingResponse(
        chunks(),
        media_type="application/json",
        headers={"ETag": quoted, "Cache-Control": "public, max-age=31536000, immutable"},
    )


def _comparison_incompatibilities(
    left: ReplayLoader, right: ReplayLoader
) -> list[str]:
    reasons: list[str] = []
    pairs = {
        "evaluation seed": (
            left.manifest.source.evaluation_seed,
            right.manifest.source.evaluation_seed,
        ),
        "scenario": (left.manifest.versions.scenario, right.manifest.versions.scenario),
        "environment configuration": (
            left.manifest.environment_config,
            right.manifest.environment_config,
        ),
        "duration": (left.manifest.world_steps, right.manifest.world_steps),
    }
    for label, (left_value, right_value) in pairs.items():
        if left_value != right_value:
            reasons.append(f"{label} differs")
    left_initial, right_initial = left.initial(), right.initial()
    if left_initial.get("terrain") != right_initial.get("terrain"):
        reasons.append("map terrain differs")
    return reasons


def _series_delta(left: list[float], right: list[float]) -> list[float]:
    length = min(len(left), len(right))
    return [float(right[index]) - float(left[index]) for index in range(length)]


def _encode_cursor(value: int) -> str:
    return base64.urlsafe_b64encode(str(value).encode()).decode().rstrip("=")


def _decode_cursor(value: str) -> int:
    try:
        padded = value + "=" * (-len(value) % 4)
        return max(0, int(base64.urlsafe_b64decode(padded).decode()))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid cursor.") from exc


def _configured_roots() -> list[tuple[str, Path]]:
    configured = os.environ.get("VOYAGER_REPLAY_ROOTS")
    if configured:
        return [
            (f"configured-{index}", Path(value))
            for index, value in enumerate(configured.split(os.pathsep))
            if value
        ]
    return [
        ("curated", REPOSITORY_ROOT / "benchmarks/replays/stage6_curated_v1"),
        ("local", REPOSITORY_ROOT / "runs/replays"),
    ]
