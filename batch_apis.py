import time
import json
import pandas as pd
import re
import gradio as gr
import numpy as np
import os
from tqdm import tqdm

current_dir = os.path.dirname(__file__)

def get_params_from_prompt_list(prompt):
    keys = re.findall(r'\{(\w+)\}', prompt)
    return keys

# jsonl 格式 batch 请求文件
def to_batch_jsonl(df: pd.DataFrame, file_out: str, sys_prompt_template: str, user_prompt_template: str, model: str, url: str, options: object, response_format: str, json_schema: str):
    file_out = os.path.join(current_dir, file_out)
    print("current_direcrtory", current_dir)
    print("file_out", file_out)
    try:
        with open(file_out, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                
                user_args = {param: row[param] for param in get_params_from_prompt_list(user_prompt_template)}
                if sys_prompt_template: 
                    sys_args = {param: row[param] for param in get_params_from_prompt_list(sys_prompt_template)}
                    prompt = [
                        {"role": "system", "content": sys_prompt_template.format(**sys_args)},
                        {"role": "user", "content": user_prompt_template.format(**user_args)}
                    ]
                else: 
                    prompt = [{"role": "user", "content": user_prompt_template.format(**user_args)}]

                data = get_batch_line(id=index, messages=prompt, model=model, url=url, options=options, response_format=response_format, json_schema=json_schema)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print("请求Jsonl文件已存至:", file_out)
        gr.Info(f"请求Jsonl文件已存至: {file_out}")
        return True
    except Exception as e:
        print(e)
        gr.Error(f"写入Jsonl文件失败: {e}")
        return False


# 获取单行 json 数据
def get_batch_line(id, messages, model, url, options, response_format, json_schema):
    data = {
        'custom_id': f"request-{id}",
        'method': 'POST',
        'url': url,
        'body': {
            'model': model,
            'temperature': options['temperature'],
            'top_p': options['top_p'],
            'frequency_penalty': options['frequency_penalty'],
            'presence_penalty': options['presence_penalty'],
            'messages': messages,
        }
    }
    if response_format == 'json_object':
        data['body']['response_format'] = {
            "type": "json_object"
        }
    elif response_format == 'json_schema':
        data['body']['response_format'] = {
            "type": "json_schema",
            "json_schema": json_schema
        }
    return data

# 批量请求
def start_batch_job(file_in, client):
    print(file_in)
    try:
        batch_input_file = client.files.create(
            file=open(file_in, "rb"),
            purpose="batch"
        )
        print("Batch input file uploaded. File ID:", batch_input_file.id)
        batch_process = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            }
        )

        while batch_process.status == "validating":
            time.sleep(2)
            batch_process = client.batches.retrieve(batch_process.id)

        if batch_process.status == "failed":
            gr.Error("批处理请求失败: " + batch_process.error)
            return None
        gr.Info(f"批处理请求运行中。")
        print("Batch process started. Batch ID:", batch_process.id)
        return batch_process.id
    except Exception as e:
        gr.Error(f"批处理请求失败: {e}")
        return

# 获取批量请求结果
def get_batch_results(batch_id, file_out, client):
    batch = client.batches.retrieve(batch_id)
    while batch.status == "validating":
        time.sleep(2)
        batch = client.batches.retrieve(batch_id)
    print("Batch process status:", batch.status)
    total = batch.request_counts.total
    current = batch.request_counts.completed
    print("Total requests:", total)

    progress = gr.Progress(track_tqdm=True)
    progress(current/total, desc="Starting")
    while batch.status != "completed":
        batch = client.batches.retrieve(batch_id)
        if batch.status == "failed":
            print("Batch failed")
            gr.Error("批处理请求失败: " + batch.error)
            return None
 
        # print("Batch process status:", batch.status)
        if batch.request_counts.completed != current:
            current = batch.request_counts.completed
            print(f"Completed requests: {current}/{total}")
            if current == total:
                progress(0.9, "Processing")
            else:
                progress(current / total, "Processing")
        time.sleep(2)

    if batch.status == "completed":
        print("Batch process completed.")

        print("Downloading batch output file...")
        if not batch.output_file_id:
            print("Batch output file not found.")
            batch_error_file = client.files.content(batch.error_file_id)
            file_out = file_out.replace(".json", "_error.json")
            with open(file_out, "w", encoding="utf-8") as f:
                f.write(batch_error_file.text)
        else:
            batch_output_file = client.files.content(batch.output_file_id)
            with open(file_out, "w", encoding="utf-8") as f:
                f.write(batch_output_file.text)
        progress(1, "Finish")

        print("Batch output file saved to", file_out)
        gr.Info(f"批处理结果已存至: {file_out}")
        return file_out
    else:
        print("Batch process not completed yet.")
        gr.Error("批处理请求失败: " + batch.error)
        return None

# Format batch results into a csv file
def format_batch_results(input_df, batch_result_jsonl, file_out, output_column, json_mode=False):
    try:
        df = pd.read_json(batch_result_jsonl, lines=True)
        df_raw = input_df
        df_raw['id'] = np.arange(0, len(df))

        print("Formatting batch results...")
        if batch_result_jsonl.endswith('_error.jsonl'):
            df[output_column] = df['response'].apply(lambda x: x['body']['error']['message'])
        else:
            df[output_column] = df['response'].apply(lambda x: x['body']['choices'][0]['message']['content'])

            if json_mode:
                json_keys = json.loads(df.iloc[0][output_column]).keys()
                for key in json_keys:
                    df[key] = df[output_column].apply(lambda x: json.loads(x)[key])

        df['id'] = df['custom_id'].apply(lambda x: int(x.split('-')[-1]))
        df.drop(columns=['response'], inplace=True)
        df.drop(columns=['custom_id'], inplace=True)
        df.drop(columns=['error'], inplace=True)

        combined_df = pd.merge(df_raw, df, on='id', how='left')
        combined_df.to_excel(file_out, index=False)
        print("Data saved to", file_out)
    except Exception as e:
        print("batch_formatting_error:", e)
        gr.Error(f"写入Excel文件失败: {e}")
        return False

    print("Data saved to", file_out)
    gr.Info(f"批处理任务已完成, 数据已存至: {file_out}")
    return True
