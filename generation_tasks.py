import os

from dotenv import load_dotenv

from constraint_decay.prompts.architectures import CLEAN_ARCHITECTURE_TEMPLATE
from constraint_decay.prompts.database import (
    DATABASE_POSTGRES_TEMPLATE,
    DATABASE_SQLITE_TEMPLATE,
)
from constraint_decay.prompts.evaluation import (
    EVALUATION_PIPELINE_NODE_TEMPLATE,
    EVALUATION_PIPELINE_PYTHON_TEMPLATE,
)
from constraint_decay.prompts.instruction import INSTRUCTION_TEMPLATE
from constraint_decay.prompts.mandatory_files import (
    MANDATORY_FILES_NODE_TEMPLATE,
    MANDATORY_FILES_PYTHON_TEMPLATE,
)
from constraint_decay.prompts.requirements import (
    ARCHITECTURE_REQUIREMENT,
    DATABASE_REQUIREMENT,
    REQUIREMENTS_TEMPLATE,
    SEQUELIZE_REQUIREMENT,
    SQLALCHEMY_REQUIREMENT,
)
from constraint_decay.prompts.server_configs import SERVER_CONFIG_TEMPLATE
from generate_task import main as generate

RUNTIMES = ["uv", "node"]

RUNTIME_DESCRIPTIONS = {"uv": "Python 3.12", "node": "Node.js"}

FRAMEWORKS = {
    "uv": [
        "aiohttp",
        "django",
        "fastapi",
        "flask",
    ],
    "node": [
        "express",
        "fastify",
        "hono",
        "koa",
    ],
}

ORMS = {"uv": "sqlalchemy", "node": "sequelize"}
ORM_REQUIREMENTS = {"uv": SQLALCHEMY_REQUIREMENT, "node": SEQUELIZE_REQUIREMENT}

PORTS = {"uv": "8080", "node": "3000"}

MANDATORY_FILES = {
    "uv": MANDATORY_FILES_PYTHON_TEMPLATE,
    "node": MANDATORY_FILES_NODE_TEMPLATE,
}

EVALUATION_PIPELINES = {
    "uv": EVALUATION_PIPELINE_PYTHON_TEMPLATE,
    "node": EVALUATION_PIPELINE_NODE_TEMPLATE,
}

SPECS = ["openapi"]

SPEC_FILES = {"openapi": "openapi.yml"}

POSTMAN_COLLECTIONS = {
    "openapi": "Conduit.postman_collection.json",
}

TASK_PROMPT_TEMPLATE = """{{}}

{instruction}

{requirements}
{architecture}
{database}

{mandatory_files}

{server_configuration}

{evaluation_pipeline}
"""


def _generate_variant(
    runtime: str,
    framework: str,
    spec: str,
    constraints: list[str],
    additional_requirements: list[str] = [],
    architecture: str = "",
    database: str = "",
) -> None:
    tasks_absolute_path = os.environ["TASKS_ABSOLUTE_PATH"]
    spec_absolute_path = os.path.join(os.getcwd(), "runtime", SPEC_FILES[spec])

    formatted_constraints = "-".join(constraints)
    task_name = f"{framework}-{spec}-{formatted_constraints}"
    task_description = f"{runtime} generation for {framework} with {spec}. Constraints: {formatted_constraints}."
    instruction = INSTRUCTION_TEMPLATE.format(runtime=RUNTIME_DESCRIPTIONS[runtime])
    requirements = REQUIREMENTS_TEMPLATE.format(
        framework=framework,
        additional_requirements="\n".join(additional_requirements),
    )
    mandatory_files = MANDATORY_FILES[runtime]
    server_configuration = SERVER_CONFIG_TEMPLATE.format(port=PORTS[runtime])
    evaluation_pipeline = EVALUATION_PIPELINES[runtime].format(
        framework=framework,
        port=PORTS[runtime],
    )
    prompt = TASK_PROMPT_TEMPLATE.format(
        instruction=instruction,
        requirements=requirements,
        architecture="\n" + architecture if len(architecture) > 0 else architecture,
        database="\n" + database if len(database) > 0 else database,
        mandatory_files=mandatory_files,
        server_configuration=server_configuration,
        evaluation_pipeline=evaluation_pipeline,
    )

    generate(
        runtime,
        task_name,
        task_description,
        None,
        tasks_absolute_path,
        spec_absolute_path,
        prompt,
        postman_collection=POSTMAN_COLLECTIONS[spec],
    )


def _generate_variants(runtime: str, framework: str, spec: str) -> None:
    # 0. Unconstrained
    _generate_variant(runtime, framework, spec, ["unconstrained"])

    # 1. Minimal Constraints
    _generate_variant(
        runtime,
        framework,
        spec,
        ["clean_architecture"],
        additional_requirements=[ARCHITECTURE_REQUIREMENT],
        architecture=CLEAN_ARCHITECTURE_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["sqlite"],
        additional_requirements=[DATABASE_REQUIREMENT],
        database=DATABASE_SQLITE_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["postgres"],
        additional_requirements=[DATABASE_REQUIREMENT],
        database=DATABASE_POSTGRES_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["clean_architecture", "sqlite"],
        additional_requirements=[ARCHITECTURE_REQUIREMENT, DATABASE_REQUIREMENT],
        architecture=CLEAN_ARCHITECTURE_TEMPLATE,
        database=DATABASE_SQLITE_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["clean_architecture", "postgres"],
        additional_requirements=[ARCHITECTURE_REQUIREMENT, DATABASE_REQUIREMENT],
        architecture=CLEAN_ARCHITECTURE_TEMPLATE,
        database=DATABASE_POSTGRES_TEMPLATE,
    )

    # 2. Dependencies Constraints
    _generate_variant(
        runtime,
        framework,
        spec,
        ["sqlite", ORMS[runtime]],
        additional_requirements=[DATABASE_REQUIREMENT, ORM_REQUIREMENTS[runtime]],
        database=DATABASE_SQLITE_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["postgres", ORMS[runtime]],
        additional_requirements=[DATABASE_REQUIREMENT, ORM_REQUIREMENTS[runtime]],
        database=DATABASE_POSTGRES_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["clean_architecture", "sqlite", ORMS[runtime]],
        additional_requirements=[
            ARCHITECTURE_REQUIREMENT,
            DATABASE_REQUIREMENT,
            ORM_REQUIREMENTS[runtime],
        ],
        architecture=CLEAN_ARCHITECTURE_TEMPLATE,
        database=DATABASE_SQLITE_TEMPLATE,
    )
    _generate_variant(
        runtime,
        framework,
        spec,
        ["clean_architecture", "postgres", ORMS[runtime]],
        additional_requirements=[
            ARCHITECTURE_REQUIREMENT,
            DATABASE_REQUIREMENT,
            ORM_REQUIREMENTS[runtime],
        ],
        architecture=CLEAN_ARCHITECTURE_TEMPLATE,
        database=DATABASE_POSTGRES_TEMPLATE,
    )


def main() -> None:
    for runtime in RUNTIMES:
        for framework in FRAMEWORKS[runtime]:
            for spec in SPECS:
                _generate_variants(runtime, framework, spec)


if __name__ == "__main__":
    load_dotenv()

    main()
