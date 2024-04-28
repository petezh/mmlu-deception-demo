from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from keys import OPENAI_API_KEY, ANTHROPIC_API_KEY

TEMPERATURE = 0.0

LLMs = {
    "gpt-3.5":ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-1106", temperature=TEMPERATURE, max_tokens=250),
    "gpt-4":ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-1106-preview", temperature=TEMPERATURE, max_tokens=250),
    "claude-3":ChatAnthropic(api_key=ANTHROPIC_API_KEY, model="claude-3-opus-20240229", temperature=TEMPERATURE, max_tokens=250),
}

DEFAULT_MODEL = "gpt-4"

def call(
        prompt: str,
        role: str = "You are a student answering a question.",
        model_name: str=DEFAULT_MODEL,
        streaming: bool=False,
        ):
    """
    Calls an LLM API with the prompt.
    """
    assert model_name in LLMs, f"Model {model_name} not found. Available models: {list(LLMs.keys())}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", role),
        ("user", "{input}")
    ])
    llm = LLMs[model_name]
    chain = prompt_template | llm 

    if streaming:
        stream = chain.stream({"input": prompt})
        get_content = lambda x: x.content
        return map(get_content, stream)

    else:
        response = chain.invoke({"input": prompt})
        return response.content
