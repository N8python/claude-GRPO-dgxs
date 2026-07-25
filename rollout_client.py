"""Client for Bob's rollout server. Used by the GRPO trainer and the evaluator."""

import time

import requests
from safetensors.torch import save as save_bytes


class RolloutClient:
    def __init__(self, url: str, timeout: float = 1800.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # -- introspection ----------------------------------------------------
    def health(self):
        r = self._session.get(f"{self.url}/health", timeout=30)
        r.raise_for_status()
        return r.json()

    def info(self):
        r = self._session.get(f"{self.url}/info", timeout=30)
        r.raise_for_status()
        return r.json()

    def wait_ready(self, timeout: float = 900.0, poll: float = 5.0):
        """Block until Bob answers. vLLM takes a while to warm up."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                return self.health()
            except Exception as e:  # connection refused while vLLM loads
                last = e
                time.sleep(poll)
        raise TimeoutError(f"{self.url} not ready after {timeout}s (last error: {last})")

    # -- rollouts ---------------------------------------------------------
    def generate(
        self,
        prompt_token_ids,
        n=1,
        temperature=0.7,
        top_p=0.0,
        top_k=0,
        max_tokens=768,
        seed=None,
        logprobs=True,
    ):
        """Returns the raw server payload; results are group-major (p*n + k)."""
        body = {
            "prompt_token_ids": [list(map(int, ids)) for ids in prompt_token_ids],
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "seed": seed,
            "logprobs": logprobs,
        }
        r = self._session.post(f"{self.url}/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # -- weight sync ------------------------------------------------------
    def update_weights(self, lora_state: dict, scale: float, step: int = -1):
        """POST LoRA A/B as safetensors. ~22 MB at rank 8 for a 1B model."""
        payload = save_bytes({k: v.contiguous() for k, v in lora_state.items()})
        r = self._session.post(
            f"{self.url}/update_weights",
            params={"scale": scale, "step": step},
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def reset_weights(self):
        r = self._session.post(f"{self.url}/reset_weights", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
