import gradio as gr
import pandas as pd
from pathlib import Path
import os
from core import SemanticExplorer, utils, chat_agent, model_logic, repo_loader

explorer = SemanticExplorer()

with gr.Blocks(theme=gr.themes.Soft(), title="SmartFile Explorer") as demo:
    gr.Markdown("# 🚀 SmartFile Explorer")
    gr.Markdown("An AI-powered assistant for filesystem exploration and semantic search.")
    initial_path = Path.cwd().anchor
    current_dir_state = gr.State(value=initial_path)
    with gr.Sidebar(open=False):
        gr.Markdown("### 🤖 Model Settings")
        provider_dropdown = gr.Dropdown(label="Select Provider", choices=model_logic.get_available_providers(), value=model_logic.get_available_providers()[0] if model_logic.get_available_providers() else None, interactive=True)
        model_dropdown = gr.Dropdown(label="Select Model", choices=model_logic.get_model_display_names_for_provider(provider_dropdown.value or ""), value=model_logic.get_default_model_display_name_for_provider(provider_dropdown.value or ""), interactive=True)
        api_key_textbox = gr.Textbox(label="API Key (Optional Override)", placeholder="Overrides key from .env file", type="password", interactive=True)
        temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.3, step=0.1, label="Temperature", interactive=True)
        max_tokens_slider = gr.Slider(minimum=256, maximum=8192, value=2048, step=256, label="Max New Tokens", interactive=True)
    with gr.Tabs():
        with gr.TabItem("💬 Chatbot Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SmartFile Assistant", bubble_full_width=False, height=600)
                    with gr.Row():
                        chat_input_textbox = gr.Textbox(placeholder="Ask about files, their content, or relationships...", scale=4, autofocus=True, container=False)
                        clear_chat_button = gr.Button("🗑️ Clear", variant="secondary")
            
        with gr.TabItem("📂 File Explorer"):
            with gr.Row():
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("### Navigation")
                    up_button = gr.Button("⬆️ Go Up")
                    home_button = gr.Button("🏠 Go to App Home")
                    root_button = gr.Button("̸ Go to Root")
                    gr.Markdown("### Go to Path:")
                    path_input = gr.Textbox(label="Enter path and press Enter", value=initial_path, interactive=True)
                with gr.Column(scale=3):
                    current_path_display = gr.Label(label="Current Directory")
                    file_list_df = gr.DataFrame(headers=["Name", "Type", "Size (bytes)", "Modified", "Permissions"], interactive=True, row_count=(15, "dynamic"))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### File Content Viewer")
                    selected_file_path = gr.Textbox(label="Selected File Path", interactive=False)
                    file_content_display = gr.Code(label="Content", language=None, lines=20, interactive=False)

        with gr.TabItem("🔎 Manual Search & Indexing"):
            gr.Markdown("### Persistent Vector Search")
            gr.Markdown("Click on a file in the search results to view its content below.")
            search_results_state = gr.State([])
            with gr.Row():
                index_path_input = gr.Textbox(label="Directory Path or Repository URL to Index", value=os.getcwd(), interactive=True)
                with gr.Column():
                    build_index_button = gr.Button("Build / Update Index", variant="primary", visible=True)
                    stop_button = gr.Button("Stop Building", variant="stop", visible=False)
            with gr.Row():
                clear_index_button = gr.Button("Clear Entire Index", variant="stop")
            index_status_label = gr.Label(label="Index Status")
            with gr.Row():
                search_query_input = gr.Textbox(label="Search Query", placeholder="e.g., 'a function to login a user' or 'test cases for the API'", interactive=True, scale=4)
            gr.Markdown("### Search Results")
            search_results_df = gr.DataFrame(headers=["Similarity", "Path", "Type", "Size (Bytes)", "Modified"], interactive=True)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### File Content Viewer (from Search)")
                    search_selected_file_path = gr.Textbox(label="Selected File Path", interactive=False)
                    search_content_display = gr.Code(label="Content", language=None, lines=20, interactive=False)

    def update_model_dropdown(provider):
        models = model_logic.get_model_display_names_for_provider(provider)
        default_model = model_logic.get_default_model_display_name_for_provider(provider)
        return gr.Dropdown(choices=models, value=default_model)

    provider_dropdown.change(fn=update_model_dropdown, inputs=[provider_dropdown], outputs=[model_dropdown])

    def chat_response_wrapper(chat_history, user_input, provider, model, api_key, temp, tokens):
        if not user_input.strip():
            yield chat_history
            return
        
        chat_history.append([user_input, None])

        response_generator = chat_agent.get_response_stream(
            user_input, explorer, provider, model, api_key, temp, tokens
        )
        
        for response_chunk in response_generator:
            chat_history[-1][1] = response_chunk
            yield chat_history

    def clear_chat():
        return [], ""

    chat_submit_args = {
        "fn": chat_response_wrapper,
        "inputs": [chatbot, chat_input_textbox, provider_dropdown, model_dropdown, api_key_textbox, temperature_slider, max_tokens_slider],
        "outputs": [chatbot],
    }
    chat_input_textbox.submit(**chat_submit_args)
    clear_chat_button.click(fn=clear_chat, outputs=[chatbot, chat_input_textbox])

    def update_file_list(path_str):
        df, label = utils.get_directory_contents(path_str)
        return df, label, path_str
    demo.load(update_file_list, [current_dir_state], [file_list_df, current_path_display, current_dir_state])
    path_input.submit(update_file_list, [path_input], [file_list_df, current_path_display, current_dir_state])
    def handle_row_select(evt: gr.SelectData, df: pd.DataFrame, current_dir: str):
        if evt.index is None: return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        item_path = os.path.join(current_dir, df.iloc[evt.index[0]]["Name"])
        if df.iloc[evt.index[0]]["Type"] == "📁 Folder":
            new_df, new_label = utils.get_directory_contents(item_path)
            return new_df, new_label, "# This is a directory.", "", item_path
        else:
            content, filepath = utils.get_file_content(item_path)
            return gr.update(), gr.update(), content, filepath, gr.update()
    file_list_df.select(handle_row_select, [file_list_df, current_dir_state], [file_list_df, current_path_display, file_content_display, selected_file_path, current_dir_state])
    up_button.click(lambda d: update_file_list(str(Path(d).parent)), [current_dir_state], [file_list_df, current_path_display, current_dir_state])
    home_button.click(lambda: update_file_list(os.getcwd()), [], [file_list_df, current_path_display, current_dir_state])
    root_button.click(lambda: update_file_list(Path.cwd().anchor), [], [file_list_df, current_path_display, current_dir_state])

    def do_build_index(path_or_url, progress=gr.Progress(track_tqdm=True)):
        for status in explorer.index_directory(path_or_url, progress_callback=progress):
            yield status
    def do_search(query):
        raw_results = explorer.search(query)
        df_data = [{
            "Similarity": f"{r['similarity']:.3f}", "Path": r['path'], "Type": r['type'],
            "Size (Bytes)": r['size'] if r['size'] is not None else "",
            "Modified": r['modified'].strftime('%Y-%m-%d %H:%M')
        } for r in raw_results]
        return pd.DataFrame(df_data), raw_results
    def do_clear_index():
        count = explorer.clear_index()
        return f"Successfully cleared {count} items from the index."
    def handle_search_row_select(evt: gr.SelectData, raw_results: list):
        if evt.index is None or not raw_results:
            return "", "# Select a file from the search results to view its content."
        selected_item = raw_results[evt.index[0]]
        full_path = selected_item.get("full_path")
        if not full_path:
            return "Error: Full path not found in search results.", "# Could not load content."
        if selected_item['type'] == "📁 Folder":
            return full_path, "# This is a directory. You can navigate to it in the File Explorer tab."
        
        if full_path.startswith("http"):
            content, err = repo_loader.get_remote_file_content(full_path)
            if err:
                return full_path, f"# Error loading remote file:\n{err}"
            return full_path, content
        else:
            content, _ = utils.get_file_content(full_path)
            return full_path, content
    
    demo.load(explorer.get_status, outputs=[index_status_label])
    def start_indexing(): return gr.update(visible=False), gr.update(visible=True)
    def stop_indexing(): explorer.cancel_indexing(); return gr.update(visible=True), gr.update(visible=False)
    def finish_indexing(): return gr.update(visible=True), gr.update(visible=False)
    click_event = build_index_button.click(start_indexing, outputs=[build_index_button, stop_button]) \
        .then(do_build_index, inputs=[index_path_input], outputs=[index_status_label]) \
        .then(finish_indexing, outputs=[build_index_button, stop_button])
    stop_button.click(stop_indexing, outputs=[build_index_button, stop_button], cancels=[click_event])
    clear_index_button.click(do_clear_index, outputs=[index_status_label])
    search_query_input.submit(fn=do_search, inputs=[search_query_input], outputs=[search_results_df, search_results_state])
    search_results_df.select(fn=handle_search_row_select, inputs=[search_results_state], outputs=[search_selected_file_path, search_content_display])

if __name__ == "__main__":
    demo.launch(debug=False)
