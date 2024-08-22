import gradio as gr
import dotenv
import os
import re
import uuid
import pandas as pd
from openai import OpenAI
from batch_apis import to_batch_jsonl, start_batch_job, get_batch_results, format_batch_results

dotenv.load_dotenv()
os.chdir(os.path.dirname(__file__))

DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUTPUT_COLUMN = "gpt_response"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9

URL = "/v1/chat/completions"

if not os.path.exists("data"):
    os.mkdir("data")
    os.mkdir("data/metas")
    os.mkdir("data/output")

file_path = {
    "jsonl": "data/metas/",
    "output": "data/output/"
}

def save_api_key(api_key):
    dotenv.set_key(".env", "OPENAI_API_KEY", api_key)

def get_params_from_prompt(sys_prompt, user_prompt):
    prompt = sys_prompt + user_prompt
    keys = re.findall(r'\{(\w+)\}', prompt)
    if len(keys) == 0:
        return "提示词中没有参数"
    return ", ".join(keys)

def get_params_from_prompt_list(prompt):
    keys = re.findall(r'\{(\w+)\}', prompt)
    return keys

def check_file(file):
    if file is None:
        gr.Warning("请上传文件!", duration=5)
        return [None, None]
    if not (file.name.endswith(".csv") or file.name.endswith(".xlsx")):
        gr.Warning("文件格式错误，请上传csv或xlsx文件!", duration=5)
        return [None, None]
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    return [file, gr.Dataframe(df, label="原始数据表", interactive=False, visible=True)]

def request_batch(api_key, model_name, temperature, top_p, frequency_penalty, presence_penalty, system_prompt, user_prompt, df, output_column):
    # return "temp.xlsx"
    if not model_name:
        model_name = DEFAULT_MODEL
    if not output_column:
        output_column = DEFAULT_OUTPUT_COLUMN

    if api_key is None:
        gr.Warning("请填写正确的OpenAI API Key!", duration=5)
        return
    if df is None:
        gr.Warning("请上传文件!", duration=5)
        return
    if system_prompt is None:
        gr.Warning("请填写系统提示词(System Prompt)!", duration=5)
        return
    if user_prompt is None:
        gr.Warning("请填写用户提示词(User Prompt)!", duration=5)
        return

    if output_column in df.columns:
        gr.Warning(f"输出参数名“{output_column}”已经在数据表中请检查!", duration=5)
        return

    prompt = system_prompt + user_prompt
    keys = get_params_from_prompt_list(prompt)
    params_exist = True
    for param in keys:
        if param not in df.columns:
            gr.Warning(f"参数“{param}”不在数据表中请检查!", duration=5)
            params_exist = False
    if not params_exist:
        return
    
    try:
        client = OpenAI(api_key=api_key)
        models = [m.id for m in client.models.list()]
        if model_name not in models:
            gr.Warning(f"模型 {model_name} 不存在!", duration=5)
            return
    except:
        gr.Error("OpenAI API Key 错误或网络环境错误！", duration=5)
        return
    
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }

    file_name = f"{uuid.uuid4()}"
    jsonl_request_file = file_path['jsonl'] + file_name + '_req.jsonl'
    if not to_batch_jsonl(df, jsonl_request_file, system_prompt, user_prompt, model_name, URL, options, json_mode=False):
        return gr.Error(f"写入Jsonl文件失败")
    batch_id = start_batch_job(jsonl_request_file)
    if not batch_id:
        return
    jsonl_response_file = file_path['jsonl'] + file_name + '_res.jsonl'
    if not get_batch_results(batch_id, jsonl_response_file):
        return
    excel_file_out = file_path['output'] + file_name + '.xlsx'
    if format_batch_results(df, jsonl_response_file, excel_file_out, output_column):
        return excel_file_out
    return
    



with gr.Blocks() as demo:
    gr.Markdown("## OpenAI 批处理 UI")
    with gr.Row():
        api_key = gr.Textbox(label="OpenAI API Key", value=os.getenv("OPENAI_API_KEY"), placeholder="输入你的OpenAI API Key")
        model_name = gr.Textbox(label="Model Name", placeholder=DEFAULT_MODEL)
        api_key.change(save_api_key, inputs=api_key)
    
    with gr.Row():
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=DEFAULT_TEMPERATURE, interactive=True)
        top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.1, value=DEFAULT_TOP_P, interactive=True)
        frequency_penalty = gr.Slider(label="Frequency Penalty", minimum=0.0, maximum=1.0, step=0.1, value=0.0, interactive=True)
        presence_penalty = gr.Slider(label="Presence Penalty", minimum=0.0, maximum=1.0, step=0.1, value=0.0, interactive=True)

    with gr.Row():
        with gr.Column():
            system_prompt = gr.Textbox(label="System Prompt", placeholder="输入系统提示词（用{ }表示参数）")
            user_prompt = gr.Textbox(label="User Prompt", placeholder="输入用户提示词（用{ }表示参数）", lines=5)
        with gr.Column():
            params = gr.Textbox(label="输入参数", placeholder="在左侧输入提示词自动识别参数名", interactive=False, lines=5)
            output_column = gr.Textbox(label="输出参数名", placeholder="gpt_response")
        user_prompt.change(fn=get_params_from_prompt, inputs=[system_prompt, user_prompt], outputs=params)
        system_prompt.change(fn=get_params_from_prompt, inputs=[system_prompt, user_prompt], outputs=params)

    file_in = gr.File(label="上传原始数据表")
    df = gr.Dataframe(label="原始数据表", interactive=False, visible=False)
    file_in.upload(check_file, inputs=file_in, outputs=[file_in, df])
    file_in.clear(lambda x: None, inputs=file_in, outputs=df)

    request_btn = gr.Button("请求批处理🚀")
    file_out = gr.File(label="下载处理后数据表")
    request_btn.click(request_batch, inputs=[api_key, model_name, temperature, top_p, frequency_penalty, presence_penalty, system_prompt, user_prompt, df,output_column], outputs=file_out)

    # with gr.Row() as metas:
    #     batch_id = gr.Textbox(label="Batch ID", placeholder="请求批处理任务后显示", interactive=False)
    #     batch_id = gr.Textbox(label="Batch ID", placeholder="请求批处理任务后显示", interactive=False)

    

if __name__ == "__main__":
    dotenv.load_dotenv()
    client = OpenAI()
    demo.launch()