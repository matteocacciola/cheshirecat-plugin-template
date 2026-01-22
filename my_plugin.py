from cat import tool, hook, plugin, StrayCat, CatMessage, AgenticWorkflowOutput, AgenticWorkflowTask, run_sync_or_async
from pydantic import BaseModel


class PluginSettings(BaseModel):
    favourite_language: str = "chinese"

@plugin
def settings_model():
    return PluginSettings


@tool
def get_alphabet(language: str, include_numbers: bool, cat: StrayCat):
    """Get the alphabet for a specific language, optionally including numbers."""
    prompt = f"Provide the alphabet for the language {language}, in a compact format."
    if include_numbers:
        prompt += " Include numbers as well."

    agent_input = AgenticWorkflowTask(user_prompt=prompt)
    
    alphabet = run_sync_or_async(cat.agentic_workflow.run, task=agent_input, llm=cat.large_language_model)
    return alphabet


@hook
def before_cat_sends_message(message: CatMessage, agent_output: AgenticWorkflowOutput, cat: StrayCat) -> CatMessage:
    # load plugin settings
    settings = cat.mad_hatter.get_plugin().load_settings()
    favourite_language = settings.get("favourite_language")

    # translate last message to favourite language
    last_message = message.text

    agent_input = AgenticWorkflowTask(user_prompt=f"Translate to {favourite_language}: {last_message}",)
    translation = run_sync_or_async(cat.agentic_workflow.run, task=agent_input, llm=cat.large_language_model)

    # append translation to chat response
    return CatMessage(text=translation.output)
