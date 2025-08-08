import json
import re
from . import model_logic

def _extract_xml_tag(tag: str, text: str) -> str | None:
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def get_response_stream(
    user_input: str,
    explorer,
    provider: str,
    model: str,
    api_key: str,
    temp: float,
    tokens: int
):
    yield "🤔 Analyzing request and formulating a search plan..."

    search_generation_prompt = f"""
    You are a function-calling AI model that generates search parameters for a filesystem.
    Your task is to convert a user's request into a structured XML tool call.

    <instructions>
    1.  Read the user's question inside the <question> tag.
    2.  Based on the user's intent, construct an XML block: `<tool_call>...</tool_call>`.
    3.  Inside the tool call, provide a `<semantic_query>` and a `<filters>` block.
    4.  The `<filters>` block must contain a valid JSON object for metadata filtering, or leave it empty for no filter.
    </instructions>

    <tool_spec>
    <semantic_query> (string, required): A query that captures the conceptual meaning of the user's request.
    <filters> (json_object, optional): A dictionary for filtering. Available fields are:
        - "relative_path": (string) Use {{"$contains": "substring"}} for partial path matching. This is a special instruction handled by the code.
        - "is_dir": (boolean) Use `true` or `false`.
        - "size_bytes": (integer) Use {{"$gt": value}} (greater) or {{"$lt": value}} (less). These are the only valid numerical operators.
    </filters>
    </tool_spec>

    <example>
    <question>find me any python scripts about user login that are larger than 10kb</question>
    <tool_call>
      <semantic_query>python script for user authentication and login</semantic_query>
      <filters>
        {{
          "$and": [
            {{ "relative_path": {{"$contains": ".py"}} }},
            {{ "size_bytes": {{"$gt": 10240}} }}
          ]
        }}
      </filters>
    </tool_call>
    </example>

    <example>
    <question>show me the main readme file</question>
    <tool_call>
        <semantic_query>project overview summary introduction README</semantic_query>
        <filters>
            {{ "relative_path": {{"$contains": "README"}} }}
        </filters>
    </tool_call>
    </example>

    <question>{user_input}</question>
    """

    llm_response_str = ""
    try:
        for chunk in model_logic.call_model_stream(
            provider, model, [{"role": "user", "content": search_generation_prompt}], api_key, 0.0, 500
        ):
            if chunk.startswith("Error:"):
                yield f"Error during query generation: {chunk}"
                return
            llm_response_str += chunk
    except Exception as e:
        yield f"An exception occurred during query generation: {str(e)}"
        return
        
    search_query = _extract_xml_tag("semantic_query", llm_response_str)
    filters_str = _extract_xml_tag("filters", llm_response_str)
    metadata_filters = {}

    if not search_query:
        search_query = user_input
        yield f"⚠️ LLM failed to generate a valid semantic query. Using raw input.\n"
    
    if filters_str:
        try:
            metadata_filters = json.loads(filters_str)
        except json.JSONDecodeError:
            yield f"⚠️ LLM generated invalid JSON for filters. Ignoring filters.\n"
            metadata_filters = {}

    log_message = (
        f"### Step 1: Generated Search Plan\n"
        f"**Semantic Query:** `{search_query}`\n"
        f"**Metadata Filters:**\n"
        f"```json\n{json.dumps(metadata_filters, indent=2)}\n```"
    )
    yield log_message + "\n\n### Step 2: Executing Search..."
    
    search_results = explorer.search(search_query, metadata_filters=metadata_filters)

    if not search_results:
        yield log_message + "\n\n### Step 3: Synthesizing Response\n\nI couldn't find any files matching that specific search. The index might be empty, or you could try rephrasing your request."
        return

    context_str = "SEARCH RESULTS:\n"
    for r in search_results[:10]:
        context_str += f"- Full Path: {r['full_path']}\n  Similarity: {r['similarity']:.2f}\n"

    response_synthesis_prompt = f"""
    You are a helpful AI file system assistant.
    You have performed a search and received the following results.

    <search_results>
    {context_str}
    </search_results>

    <instructions>
    1.  Provide a comprehensive and helpful answer to the user's original question.
    2.  First, write a conversational summary of your findings.
    3.  Then, in a separate section, list the full paths of the most relevant files inside a Markdown code block titled "Relevant Files".
    4.  Do not invent file contents or information not present in the search results.
    </instructions>

    User's original question: "{user_input}"
    """

    full_response = ""
    messages_for_response = [{"role": "system", "content": response_synthesis_prompt}]
    try:
        for chunk in model_logic.call_model_stream(
            provider, model, messages_for_response, api_key, temp, tokens
        ):
            if chunk.startswith("Error:"):
                yield chunk
                return
            full_response += chunk
            yield log_message + "\n\n### Step 3: Synthesizing Response\n\n" + full_response
    except Exception as e:
        yield f"An exception occurred while generating the final response: {str(e)}"
