"""Chainlit UI: `chainlit run app.py` (install the `ui` extra first)."""

from __future__ import annotations

import asyncio

import chainlit as cl

from src import log
from src.pipeline import Pipeline, build_pipeline

log.setup()
_pipeline: Pipeline | None = None


def pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@cl.on_chat_start
async def start() -> None:
    try:
        await asyncio.to_thread(pipeline)
        status = "ready"
    except Exception as e:  # show configuration problems in the UI instead of a blank page
        status = f"not ready: {e}"
    await cl.Message(content=f"PharmaRAG-Diagnostic is {status}. Ask about the FDA labels for semaglutide, "
                             "metformin, atorvastatin, apixaban or sertraline.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    out = await asyncio.to_thread(pipeline().run, message.content)
    hits = out.get("hits", [])
    elements = [cl.Text(name=f"[{i}] {h.payload['compound']} {h.payload['section']} "
                             f"({h.payload['effective_date']})", content=h.text, display="side")
                for i, h in enumerate(hits, 1)]
    await cl.Message(content=f"{out['answer']}\n\n*outcome: {out['outcome']}*", elements=elements).send()
