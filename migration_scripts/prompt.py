from literalai import LiteralClient

PROJECT_A_URL = "https://cloud.getliteral.ai"
PROJECT_B_URL = "https://cloud.getliteral.ai"

PROJECT_A_API_KEY = "API_KEY_A"
PROJECT_B_API_KEY = "API_KEY_B"

project_a_client = LiteralClient(url=PROJECT_A_URL, api_key=PROJECT_A_API_KEY)
project_b_client = LiteralClient(url=PROJECT_B_URL, api_key=PROJECT_B_API_KEY)


def migration_prompt(name: str, version: int):
    _from = project_a_client.api.get_prompt(name=name, version=version)

    project_b_client.api.get_or_create_prompt(
        name=name,
        template_messages=_from.template_messages,
        settings=_from.settings,
        tools=_from.tools,
    )


PROMPT_NAME = "PROMPT_NAME"
VERSION_LIMIT = 1

for i in range(0, VERSION_LIMIT):
    try:
        print(f"Migrating {PROMPT_NAME} v{i}...")
        migration_prompt(name=PROMPT_NAME, version=i)
        print(f"{PROMPT_NAME} v{i} migrated!")
    except Exception as e:
        print(f"Failed to migrate {PROMPT_NAME} v{i}: {str(e)}")
