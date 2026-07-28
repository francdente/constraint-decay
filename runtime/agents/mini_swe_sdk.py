import json
import os
from pathlib import Path
import yaml

import litellm
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


TRAJECTORIES_PATH = Path("/trajectories") / "mini"


class _CustomAgent(DefaultAgent):
    def add_message(self, role: str, content: str, **kwargs):
        print(f"=== [{role}] cost={self.model.cost:.4f}$ ===")
        print(content)
        print("===")
        super().add_message(role, content, **kwargs)
        self._save_trajectory()

    def _save_trajectory(self):
        TRAJECTORIES_PATH.mkdir(exist_ok=True)
        messages = [
            {"role": m.get("role", "?"), "content": m.get("content", "")}
            for m in self.messages
        ]
        result = {"status": "in_progress", "message": "", "trajectories": messages}
        (TRAJECTORIES_PATH / "trajectory.json").write_text(json.dumps(result, indent=4))


def _get_env_or_raise(key: str) -> str:
    var = os.getenv(key)
    if var is None:
        raise Exception(f"{key} is required!")
    return var


def main() -> None:
    model_name = _get_env_or_raise("LLM_MODEL")
    prompt = _get_env_or_raise("PROMPT")
    litellm.api_key = _get_env_or_raise("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL")
    if api_base is not None:
        litellm.api_base = api_base

    config_path = os.getenv("AGENT_CONFIG", "/agents/minisweagent.yaml")
    config_text = Path(config_path).read_text()
    full_config = yaml.safe_load(config_text)
    agent_config = full_config["agent"]
    model_config = full_config.get("model", {})

    system_guidance = os.getenv("SYSTEM_GUIDANCE")
    if system_guidance:
        agent_config["system_template"] += f"\n\n{system_guidance}"

    model = LitellmModel(model_name=model_name, **model_config)
    environment = LocalEnvironment()
    agent = _CustomAgent(model, environment, **agent_config)

    status, message = agent.run(prompt)
    print(f"Status: {status}\n{message}")

    # Final save with real status (overwrites the in_progress version)
    messages = [
        {"role": m.get("role", "?"), "content": m.get("content", "")}
        for m in agent.messages
    ]
    result = {
        "status": status,
        "message": message,
        "cost": model.cost,
        "n_calls": model.n_calls,
        "trajectories": messages,
    }
    TRAJECTORIES_PATH.mkdir(exist_ok=True)
    (TRAJECTORIES_PATH / "trajectory.json").write_text(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
