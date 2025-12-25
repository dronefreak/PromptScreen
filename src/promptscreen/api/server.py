from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..defence.abstract_defence import AbstractDefence


def create_app(guards: dict[str, AbstractDefence]) -> FastAPI:
    class EvaluationRequest(BaseModel):
        prompt: str
        defences: list[str]
        mode: str

    class DefenceResult(BaseModel):
        is_safe: bool
        details: str

    app = FastAPI(
        title="LLM Defence Suite API",
        description="Evaluates prompts against security defences.",
    )

    @app.post("/evaluate", response_model=dict[str, DefenceResult])
    async def evaluate_prompt(request: EvaluationRequest):
        mode = request.mode.lower()

        if mode == "separate":
            results = {}
            for defence_name in request.defences:
                guard = guards.get(defence_name)

                if guard:
                    analysis = guard.analyse(request.prompt)
                    results[defence_name] = DefenceResult(
                        is_safe=analysis.get_verdict(), details=analysis.get_type()
                    )
                else:
                    results[defence_name] = DefenceResult(
                        is_safe=False,
                        details=f"Error: Defence '{defence_name}' not available.",
                    )
            return results

        elif mode == "chain":
            for defence_name in request.defences:
                guard = guards.get(defence_name)

                if not guard:
                    return {
                        defence_name: DefenceResult(
                            is_safe=False,
                            details=f"Error: Defence '{defence_name}' not available.",
                        )
                    }

                analysis = guard.analyse(request.prompt)
                is_safe = analysis.get_verdict()
                details = analysis.get_type()

                if not is_safe:
                    return {
                        defence_name: DefenceResult(is_safe=is_safe, details=details)
                    }

            return {
                "ChainResult": DefenceResult(
                    is_safe=True,
                    details="All defences passed in chain evaluation.",
                )
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{request.mode}'. Supported modes: 'separate', 'chain'.",
            )

    @app.get("/defences", response_model=list[str])
    async def get_available_defences():
        return list(guards.keys())

    return app
