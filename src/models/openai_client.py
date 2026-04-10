from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ValidationError

from src.reporting import preview


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        return stripped[first_object : last_object + 1]
    first_array = stripped.find("[")
    last_array = stripped.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        return stripped[first_array : last_array + 1]
    return stripped


def _validation_error_input_text(exc: ValidationError) -> str:
    try:
        for error in exc.errors():
            raw_input = error.get("input")
            if isinstance(raw_input, str) and raw_input.strip():
                return raw_input
    except Exception:
        return ""
    return ""


class OpenAITextClient:
    def __init__(self, api_key: str | None = None, debug: bool = False) -> None:
        token = api_key or os.getenv("OPENAI_API_KEY")
        if not token:
            raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = OpenAI(api_key=token)
        self._async_client = AsyncOpenAI(api_key=token)
        self._debug = debug

    def _log(self, message: str) -> None:
        if self._debug:
            print(message, flush=True)

    @staticmethod
    def _permission_error_detail(exc: PermissionDeniedError) -> str:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message).strip()
        return str(exc).strip()

    def _raise_permission_denied(
        self,
        *,
        label: str,
        model: str,
        exc: PermissionDeniedError,
    ) -> None:
        detail = self._permission_error_detail(exc)
        hints = [
            f"OpenAI returned HTTP 403 for {label} using model '{model}'.",
            "This usually means the API key's project does not have access to that model or capability.",
            "Check OPENAI_API_KEY and confirm the configured model names are available to that project.",
        ]
        if model.lower().startswith("gpt-5"):
            hints.append(
                "The current .env uses a GPT-5 family judge by default; if your project lacks GPT-5 access, set JUDGE_MODEL to a model you do have, such as gpt-4.1-mini-2025-04-14."
            )
        hints.append("Run the pipeline with --debug --concurrency 1 if you need to confirm which call fails first.")
        if detail:
            hints.append(f"OpenAI message: {detail}")
        raise PermissionError(" ".join(hints)) from exc

    @staticmethod
    def _usage_preview(response: Any) -> str:
        usage = getattr(response, "usage", None)
        if usage is None:
            return ""
        try:
            data = usage.to_dict()
        except Exception:
            data = getattr(usage, "__dict__", None) or str(usage)
        return preview(str(data), 220)

    @staticmethod
    def _output_types(response: Any) -> list[str]:
        output = getattr(response, "output", None) or []
        return [getattr(item, "type", "<unknown>") for item in output]

    @staticmethod
    def _supports_reasoning(model: str) -> bool:
        lowered = model.lower()
        return lowered.startswith("gpt-5") or lowered.startswith(("o1", "o3", "o4"))

    @classmethod
    def _supports_temperature(cls, model: str) -> bool:
        return not cls._supports_reasoning(model)

    def _build_reasoning_options(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        reasoning_summary: str | None,
    ) -> dict[str, str] | None:
        effort = (reasoning_effort or "").strip().lower()
        summary = (reasoning_summary or "").strip().lower()
        if not effort or not self._supports_reasoning(model):
            return None

        payload: dict[str, str] = {"effort": effort}
        if summary:
            payload["summary"] = summary
        return payload

    def _build_request_kwargs(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: str | None,
        reasoning_summary: str | None,
    ) -> tuple[dict[str, Any], bool, dict[str, str] | None]:
        reasoning = self._build_reasoning_options(
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )
        supports_temperature = self._supports_temperature(model)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_output_tokens": max_output_tokens,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }
        if supports_temperature:
            request_kwargs["temperature"] = temperature
        if reasoning is not None:
            request_kwargs["reasoning"] = reasoning
        return request_kwargs, supports_temperature, reasoning

    def _log_request_start(
        self,
        *,
        label: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        user_prompt: str,
        supports_temperature: bool,
        reasoning: dict[str, str] | None,
        mode: str,
    ) -> None:
        reasoning_text = f" reasoning={reasoning}" if reasoning is not None else " reasoning=off"
        temperature_text = f"temp={temperature}" if supports_temperature else "temp=unsupported"
        self._log(
            f"    API start : {label}\n"
            f"      model   : {model}\n"
            f"      mode    : {mode}\n"
            f"      options : {temperature_text} max_tokens={max_output_tokens}{reasoning_text}\n"
            f"      prompt  : {preview(user_prompt, 180)}"
        )

    def _raise_empty_structured_response(
        self,
        *,
        label: str,
        response: Any,
    ) -> None:
        refusal = getattr(response, "refusal", None)
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        output_types = self._output_types(response)
        preview_text = preview(getattr(response, "output_text", ""), 260) or "<empty response>"
        refusal_text = preview(str(refusal), 260) if refusal else ""
        incomplete_text = preview(str(incomplete_details), 260) if incomplete_details else ""
        detail = refusal_text or preview_text or incomplete_text
        if (
            getattr(incomplete_details, "reason", None) == "max_output_tokens"
            or "max_output_tokens" in incomplete_text
        ):
            raise ValueError(
                f"Structured parse ran out of tokens for {label}. "
                f"Reasoning models count hidden reasoning tokens against max_output_tokens. "
                f"Increase JUDGE_MAX_OUTPUT_TOKENS or lower JUDGE_REASONING_EFFORT. "
                f"status={status} output_types={output_types} usage={self._usage_preview(response)}"
            )
        raise ValueError(
            f"Structured parse returned no parsed object for {label}. "
            f"status={status} output_types={output_types} "
            f"response preview: {detail or '<empty response>'} "
            f"usage={self._usage_preview(response)}"
        )

    @staticmethod
    def _recovery_max_output_tokens(max_output_tokens: int) -> int:
        bumped = max(max_output_tokens + 200, int(max_output_tokens * 1.5))
        return min(bumped, 1200)

    @staticmethod
    def _validate_json_text(text: str, response_model: type[BaseModel]) -> dict[str, Any]:
        json_blob = _extract_json_blob(text)
        payload = json.loads(json_blob)
        return response_model.model_validate(payload).model_dump()

    def complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        label: str = "request",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_format: dict[str, Any] | None = None,
    ) -> str:
        request_kwargs, supports_temperature, reasoning = self._build_request_kwargs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )
        self._log_request_start(
            label=label,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            user_prompt=user_prompt,
            supports_temperature=supports_temperature,
            reasoning=reasoning,
            mode="text",
        )
        start = time.perf_counter()
        if text_format is not None:
            request_kwargs["text"] = {"format": text_format}

        try:
            response = self._client.responses.create(**request_kwargs)
        except PermissionDeniedError as exc:
            self._raise_permission_denied(label=label, model=model, exc=exc)
        elapsed = time.perf_counter() - start
        output_text = response.output_text.strip()
        self._log(
            f"    API done  : {label}\n"
            f"      elapsed : {elapsed:.2f}s\n"
            f"      output  : {preview(output_text, 180)}"
        )
        return output_text

    async def acomplete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        label: str = "request",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_format: dict[str, Any] | None = None,
    ) -> str:
        request_kwargs, supports_temperature, reasoning = self._build_request_kwargs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )
        self._log_request_start(
            label=label,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            user_prompt=user_prompt,
            supports_temperature=supports_temperature,
            reasoning=reasoning,
            mode="text_async",
        )
        start = time.perf_counter()
        if text_format is not None:
            request_kwargs["text"] = {"format": text_format}

        try:
            response = await self._async_client.responses.create(**request_kwargs)
        except PermissionDeniedError as exc:
            self._raise_permission_denied(label=label, model=model, exc=exc)
        elapsed = time.perf_counter() - start
        output_text = response.output_text.strip()
        self._log(
            f"    API done  : {label}\n"
            f"      elapsed : {elapsed:.2f}s\n"
            f"      output  : {preview(output_text, 180)}"
        )
        return output_text

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        label: str = "request_json",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        request_kwargs, supports_temperature, reasoning = self._build_request_kwargs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

        if response_model is not None:
            self._log_request_start(
                label=label,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                user_prompt=user_prompt,
                supports_temperature=supports_temperature,
                reasoning=reasoning,
                mode="structured_parse",
            )
            start = time.perf_counter()
            try:
                response = self._client.responses.parse(**request_kwargs, text_format=response_model)
            except PermissionDeniedError as exc:
                self._raise_permission_denied(label=label, model=model, exc=exc)
            except ValidationError as exc:
                elapsed = time.perf_counter() - start
                raw_text = _validation_error_input_text(exc)
                self._log(
                    f"    API fail  : {label}\n"
                    f"      elapsed : {elapsed:.2f}s\n"
                    f"      mode    : structured_parse\n"
                    f"      error   : {preview(str(exc), 220)}"
                )
                if raw_text:
                    try:
                        return self._validate_json_text(raw_text, response_model)
                    except (json.JSONDecodeError, ValidationError):
                        pass

                retry_tokens = self._recovery_max_output_tokens(max_output_tokens)
                self._log(
                    f"      retry   : json_object fallback with max_tokens={retry_tokens}"
                )
                text = self.complete_text(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=retry_tokens,
                    label=f"{label}:json_retry",
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    text_format={"type": "json_object"},
                )
                try:
                    return self._validate_json_text(text, response_model)
                except (json.JSONDecodeError, ValidationError) as retry_exc:
                    preview_text = preview(text, 260) or "<empty response>"
                    raise ValueError(
                        f"Structured parse failed for {label}, and json fallback also failed. "
                        f"Try increasing JUDGE_MAX_OUTPUT_TOKENS or lowering JUDGE_REASONING_EFFORT. "
                        f"Fallback output preview: {preview_text}"
                    ) from retry_exc
            elapsed = time.perf_counter() - start
            parsed = getattr(response, "output_parsed", None)
            refusal = getattr(response, "refusal", None)
            self._log(
                f"    API done  : {label}\n"
                f"      elapsed : {elapsed:.2f}s\n"
                f"      status  : {getattr(response, 'status', None)}\n"
                f"      output  : {self._output_types(response)}\n"
                f"      usage   : {self._usage_preview(response)}\n"
                f"      parsed  : {preview(str(parsed) if parsed is not None else refusal or '', 180)}"
            )
            if parsed is not None:
                return parsed.model_dump()
            self._raise_empty_structured_response(label=label, response=response)

        text = self.complete_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            label=label,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_format={"type": "json_object"},
        )
        json_blob = _extract_json_blob(text)
        try:
            return json.loads(json_blob)
        except json.JSONDecodeError as exc:
            preview_text = preview(text, 260) or "<empty response>"
            raise ValueError(
                f"Model returned invalid JSON for {label}. Raw output preview: {preview_text}"
            ) from exc

    async def acomplete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        label: str = "request_json",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        request_kwargs, supports_temperature, reasoning = self._build_request_kwargs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

        if response_model is not None:
            self._log_request_start(
                label=label,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                user_prompt=user_prompt,
                supports_temperature=supports_temperature,
                reasoning=reasoning,
                mode="structured_parse_async",
            )
            start = time.perf_counter()
            try:
                response = await self._async_client.responses.parse(
                    **request_kwargs,
                    text_format=response_model,
                )
            except PermissionDeniedError as exc:
                self._raise_permission_denied(label=label, model=model, exc=exc)
            except ValidationError as exc:
                elapsed = time.perf_counter() - start
                raw_text = _validation_error_input_text(exc)
                self._log(
                    f"    API fail  : {label}\n"
                    f"      elapsed : {elapsed:.2f}s\n"
                    f"      mode    : structured_parse_async\n"
                    f"      error   : {preview(str(exc), 220)}"
                )
                if raw_text:
                    try:
                        return self._validate_json_text(raw_text, response_model)
                    except (json.JSONDecodeError, ValidationError):
                        pass

                retry_tokens = self._recovery_max_output_tokens(max_output_tokens)
                self._log(
                    f"      retry   : json_object fallback with max_tokens={retry_tokens}"
                )
                text = await self.acomplete_text(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=retry_tokens,
                    label=f"{label}:json_retry",
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    text_format={"type": "json_object"},
                )
                try:
                    return self._validate_json_text(text, response_model)
                except (json.JSONDecodeError, ValidationError) as retry_exc:
                    preview_text = preview(text, 260) or "<empty response>"
                    raise ValueError(
                        f"Structured parse failed for {label}, and json fallback also failed. "
                        f"Try increasing JUDGE_MAX_OUTPUT_TOKENS or lowering JUDGE_REASONING_EFFORT. "
                        f"Fallback output preview: {preview_text}"
                    ) from retry_exc
            elapsed = time.perf_counter() - start
            parsed = getattr(response, "output_parsed", None)
            refusal = getattr(response, "refusal", None)
            self._log(
                f"    API done  : {label}\n"
                f"      elapsed : {elapsed:.2f}s\n"
                f"      status  : {getattr(response, 'status', None)}\n"
                f"      output  : {self._output_types(response)}\n"
                f"      usage   : {self._usage_preview(response)}\n"
                f"      parsed  : {preview(str(parsed) if parsed is not None else refusal or '', 180)}"
            )
            if parsed is not None:
                return parsed.model_dump()
            self._raise_empty_structured_response(label=label, response=response)

        text = await self.acomplete_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            label=label,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_format={"type": "json_object"},
        )
        json_blob = _extract_json_blob(text)
        try:
            return json.loads(json_blob)
        except json.JSONDecodeError as exc:
            preview_text = preview(text, 260) or "<empty response>"
            raise ValueError(
                f"Model returned invalid JSON for {label}. Raw output preview: {preview_text}"
            ) from exc
