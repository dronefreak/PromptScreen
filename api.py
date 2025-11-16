from fastapi import FastAPI
from pydantic import BaseModel
from defence.abstract_defence import AbstractDefence


def create_app(guards: dict[str, "AbstractDefence"]) -> FastAPI:

    class EvaluationRequest(BaseModel):
        prompt: str
        defences: list[str]

    class DefenceResult(BaseModel):
        is_safe: bool
        details: str

    app = FastAPI(
        title="LLM Defence Suite API",
        description="Evaluates prompts against security defences.",
    )

    @app.post("/evaluate", response_model=dict[str, DefenceResult])
    async def evaluate_prompt(request: EvaluationRequest):
        results = {}
        for defence_name in request.defences:
            guard = guards.get(defence_name.capitalize())

            if guard:
                analysis = guard.analyse(request.prompt)
                results[defence_name.capitalize()] = DefenceResult(
                    is_safe=analysis.get_verdict(), details=analysis.get_type()
                )
            else:
                results[defence_name.capitalize()] = DefenceResult(
                    is_safe=False,
                    details=f"Error: Defence '{defence_name}' not available.",
                )
        return results

    @app.get("/defences", response_model=list[str])
    async def get_available_defences():
        return list(guards.keys())

    return app
