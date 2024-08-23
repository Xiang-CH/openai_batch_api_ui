import json
import gradio as gr
import dotenv
import os
import re
import uuid
import pandas as pd
from openai import OpenAI
from batch_apis import to_batch_jsonl, start_batch_job, get_batch_results, format_batch_results

dotenv.load_dotenv()

DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUTPUT_COLUMN = "gpt_response"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9

URL = "/v1/chat/completions"

os.chdir(os.path.dirname(__file__))
if not os.path.exists("data"):
    os.mkdir("data")

file_path = {
    "templates": "data/templates/",
    "jsonl": "data/metas/",
    "output": "data/output/"
}

for path in file_path.values():
    if not os.path.exists(path):
        os.mkdir(path)

def save_api_key(api_key):
    dotenv.set_key(".env", "OPENAI_API_KEY", api_key)

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

def validate_data(user_prompt, response_format, json_schema):
    if user_prompt is None or user_prompt.strip() == "":
        gr.Warning("请填写用户提示词(User Prompt)!", duration=5)
        return False
    if response_format == 'json_schema':
        try:
            json_schema = json.loads(json_schema)
        except:
            gr.Warning("请填写正确的json_schema!", duration=5)
            return False
    if response_format == 'json_object' and 'json' not in user_prompt.lower():
        gr.Warning(message="使用json_object格式，User Promp必须包含关键词：json!", duration=5)
        return False
    return True

def request_batch(api_key, model_name, temperature, top_p, frequency_penalty, presence_penalty, system_prompt, user_prompt, df, output_column, response_format, json_schema=None):
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
    
    if not validate_data(user_prompt, response_format, json_schema):
        return
    
    if response_format == 'json_schema':
        json_schema = json.loads(json_schema)
    else:
        json_schema = None

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
    if not to_batch_jsonl(df, jsonl_request_file, system_prompt, user_prompt, model_name, URL, options, response_format=response_format, json_schema=json_schema):
        return gr.Error(f"写入Jsonl文件失败")
    gr.Info(f"请求Jsonl文件已存至: {jsonl_request_file}")
    batch_id = start_batch_job(jsonl_request_file, client)
    if not batch_id:
        return
    jsonl_response_file = file_path['jsonl'] + file_name + '_res.jsonl'
    jsonl_response_file =  get_batch_results(batch_id, jsonl_response_file, client)
    if not jsonl_response_file:
        return
    excel_file_out = file_path['output'] + file_name + '.xlsx'
    if format_batch_results(df, jsonl_response_file, excel_file_out, output_column, json_mode=response_format!="text"):
        return excel_file_out
    return
    
def check_response_format(response_format):
    if response_format != 'json_schema':
        return gr.Row(visible=False)
    return gr.Row(visible=True)

def check_json(json_schema):
    try:
        json.loads(json_schema.strip())
        return "解析成功"
    except Exception as e:
        return "解析失败: " + str(e)

with gr.Blocks(theme=gr.themes.Default()) as demo:
    update_by_code = gr.State(False)

    gr.Markdown("## OpenAI 批处理 UI")
    with gr.Row():
        select_template = gr.Dropdown(label="选择模板", choices=["-"]+[f.split(".")[0] for f in os.listdir(file_path['templates'])], value="-", interactive=True, scale=4)
        delete_current_template = gr.Button("删除当前模板", variant='stop', interactive=False)

    with gr.Row():
        api_key = gr.Textbox(label="OpenAI API Key", value=os.getenv("OPENAI_API_KEY"), placeholder="输入你的OpenAI API Key", scale=3)
        model_name = gr.Textbox(label="Model Name", placeholder=DEFAULT_MODEL)
        response_format = gr.Dropdown(label="Response Format", choices=["text", "json_object", "json_schema"], value="text", interactive=True)
        api_key.change(save_api_key, inputs=api_key)
    
    with gr.Row():
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=DEFAULT_TEMPERATURE, interactive=True)
        top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.1, value=DEFAULT_TOP_P, interactive=True)
        frequency_penalty = gr.Slider(label="Frequency Penalty", minimum=0.0, maximum=1.0, step=0.1, value=0.0, interactive=True)
        presence_penalty = gr.Slider(label="Presence Penalty", minimum=0.0, maximum=1.0, step=0.1, value=0.0, interactive=True)

    with gr.Row(visible=(response_format == 'json_schema')) as jsons:
        json_schema_code = gr.Code(language="json", label="json_schema", interactive=True, scale=2)
        json_schema_parsed = gr.Textbox(label="json_schema解析状态", value="待填写", interactive=False)

    json_schema_code.change(check_json, inputs=json_schema_code, outputs=json_schema_parsed)
    response_format.change(check_response_format, inputs=response_format, outputs=jsons)

    with gr.Row():
        with gr.Column(scale=2):
            system_prompt = gr.Textbox(label="System Prompt", placeholder="输入系统提示词（用{ }表示参数）（选填）")
            user_prompt = gr.Textbox(label="User Prompt", placeholder="输入用户提示词（用{ }表示参数）", lines=10)
        with gr.Column():
            with gr.Blocks():
                params = gr.Textbox(label="输入参数", placeholder="在左侧输入提示词自动识别参数名", interactive=False, lines=2)
                output_column = gr.Textbox(label="GPT输出列表头", placeholder="gpt_response")
            with gr.Column(visible=True) as save_template_col:
                template_name = gr.Textbox(label="模板名", placeholder="输入模板名称（选填）", interactive=True)
                save_template_btn = gr.Button("保存模板", variant='secondary')


    file_in = gr.File(label="上传原始数据表")
    df = gr.Dataframe(label="原始数据表", interactive=False, visible=False)
    file_in.upload(check_file, inputs=file_in, outputs=[file_in, df])
    file_in.clear(lambda x: None, inputs=file_in, outputs=df)

    request_btn = gr.Button("请求批处理🚀", variant='primary')
    progress = gr.Progress(track_tqdm=True)
    file_out = gr.File(label="下载处理后数据表")
    request_btn.click(request_batch, inputs=[api_key, model_name, temperature, top_p, frequency_penalty, presence_penalty, system_prompt, user_prompt, df, output_column, response_format, json_schema_code], outputs=file_out)

    # 输入提示词后后显示参数
    def get_params_from_prompt(sys_prompt, user_prompt):
        prompt = sys_prompt + user_prompt
        keys = re.findall(r'\{(\w+)\}', prompt)
        if len(keys) == 0:
            return "提示词中没有参数"
        return ", ".join(keys)
    user_prompt.change(fn=get_params_from_prompt, inputs=[system_prompt, user_prompt], outputs=params)
    system_prompt.change(fn=get_params_from_prompt, inputs=[system_prompt, user_prompt], outputs=params)

    # 删除模板
    @delete_current_template.click(inputs=[select_template], outputs=[select_template, delete_current_template])
    def delete_template(template):
        if template == "-":
            gr.Warning("请选择模板!", duration=5)
            return
        try:
            os.remove(file_path['templates'] + f"{template}.json")
            gr.Info(f"模板 {template} 已删除!")
            return [gr.Dropdown(choices=["-"]+[f.split(".")[0] for f in os.listdir(file_path['templates'])], value="-"), gr.Button(interactive=False)]
        except:
            gr.Error(f"删除模板失败: {template}")
            return

    # 保存模板
    @save_template_btn.click(inputs=[template_name, system_prompt, user_prompt, model_name, response_format, output_column, json_schema_code, temperature, top_p, frequency_penalty, presence_penalty, select_template], outputs=[save_template_col, select_template, delete_current_template])
    def save_template(template_name, system_prompt, user_prompt, model_name, response_format, output_column, json_schema, temperature, top_p, frequency_penalty, presence_penalty, select_template):
        if not model_name:
            model_name = DEFAULT_MODEL
        if not output_column:
            output_column = DEFAULT_OUTPUT_COLUMN

        if not template_name:
            gr.Warning("请填写模板名!", duration=5)
            return
        
        if template_name == "-":
            gr.Warning("\"-\"不能作为模板名模板名!", duration=5)
            return
        
        for existing_template in os.listdir(file_path['templates']):
            if existing_template == f"{template_name}.json":
                gr.Warning(f"模板名 {template_name} 已存在，请更换!", duration=5)
                return
        
        if not validate_data(user_prompt, response_format, json_schema):
            return
        
        template = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_name": model_name,
            "response_format": response_format,
            "output_column": output_column,
            "json_schema": json.loads(json_schema) if response_format == 'json_schema' else None,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        with open(file_path['templates'] + f"{template_name}.json", 'w') as f:
            json.dump(template, f, ensure_ascii=False) 
        gr.Info(f"模板已保存至: {file_path['templates']}{template_name}.json")

        return [gr.Column(visible=False), gr.Dropdown(choices=["-"]+[f.split(".")[0] for f in os.listdir(file_path['templates'])], value=template_name), gr.Button(interactive=False)]

    # 选择模板后自动填充参数
    @select_template.input(inputs=[select_template], outputs=[system_prompt, user_prompt, model_name, response_format, output_column, json_schema_code, temperature, top_p, frequency_penalty, presence_penalty])
    def load_template(template):
        if template == "-":
            return [None, None, None, None, None, None, None, None, None, None]
        try:
            with open(file_path['templates'] + f"{template}.json", 'r') as f:
                template = json.load(f)
        except:
            gr.Error(f"读取模板失败: {template}")
            return [None, None, None, None, None, None, None, None, None, None]
        return [template['system_prompt'], template['user_prompt'], template['model_name'], template['response_format'], template['output_column'], json.dumps(template['json_schema']) if template['response_format'] == 'json_schema' else None, template['temperature'], template['top_p'], template['frequency_penalty'], template['presence_penalty'], True]

    # 选择模板后启用删除按钮，隐藏保存模板栏;
    @select_template.change(inputs=select_template, outputs=[delete_current_template, save_template_col])
    def on_select_template(template):
        if template == "-":
            return [gr.update(interactive=False), gr.Column(visible=True)]
        return [gr.update(interactive=True), gr.Column(visible=False)]

    # 参数修改后隐藏删除按钮，显示保存模板栏；
    for element in [system_prompt, user_prompt, model_name, response_format, output_column, json_schema_code, temperature, top_p, frequency_penalty, presence_penalty]:
        element.input(lambda: gr.Dropdown(value="-"), outputs=select_template)

    

if __name__ == "__main__":
    dotenv.load_dotenv()
    demo.launch()