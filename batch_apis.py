import time
import json
import os
import pandas as pd
import re
import gradio as gr
import numpy as np
from openai import OpenAI


client = OpenAI()
os.chdir(os.path.dirname(__file__))

def get_params_from_prompt_list(prompt):
    keys = re.findall(r'\{(\w+)\}', prompt)
    return keys

# jsonl 格式 batch 请求文件
def to_batch_jsonl(df: pd.DataFrame, file_out: str, sys_prompt_template: str, user_prompt_template: str, model: str, url: str, options: object, json_mode: bool = False):
    try:
        with open(file_out, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():

                sys_args = {param: row[param] for param in get_params_from_prompt_list(sys_prompt_template)}
                user_args = {param: row[param] for param in get_params_from_prompt_list(user_prompt_template)}
                prompt = [
                    {"role": "system", "content": sys_prompt_template.format(**sys_args)},
                    {"role": "user", "content": user_prompt_template.format(**user_args)}
                ]
                data = get_batch_line(id=index, messages=prompt, model=model, url=url, options=options, json_output=json_mode)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
        gr.Info(f"请求Jsonl文件已存至: {file_out}")
        return True
    except Exception as e:
        gr.Error(f"写入Jsonl文件失败: {e}")
        return False


# 获取单行 json 数据
def get_batch_line(id, messages, model, url, options, json_output=False):
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
    return data

# 批量请求
def start_batch_job(file_in):
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

# 获取批量请求结果
def get_batch_results(batch_id, file_out, progress=gr.Progress()):
    batch = client.batches.retrieve(batch_id)
    while batch.status == "validating":
        time.sleep(2)
        batch = client.batches.retrieve(batch_id)
    print("Batch process status:", batch.status)
    total = batch.request_counts.total
    current = batch.request_counts.completed
    print("Total requests:", total)

    while batch.status != "completed":
        batch = client.batches.retrieve(batch_id)
        if batch.status == "failed":
            print("Batch failed")
            gr.Error("批处理请求失败: " + batch.error)
            return False
 
        # print("Batch process status:", batch.status)
        if batch.request_counts.completed != current:
            current = batch.request_counts.completed
            progress(current / total)
        time.sleep(2)

    if batch.status == "completed":
        print("Batch process completed.")
        batch_output_file = client.files.content(batch.output_file_id)

        print("Downloading batch output file...")
        with open(file_out, "w", encoding="utf-8") as f:
            f.write(batch_output_file.text)

        gr.Info(f"批处理结果已存至: {file_out}")
        return True
    else:
        print("Batch process not completed yet.")
        gr.Error("批处理请求失败: " + batch.error)
        return False

# Format batch results into a csv file
def format_batch_results(input_df, batch_result_jsonl, file_out, output_column):
    try:
        df = pd.read_json(batch_result_jsonl, lines=True)
        df_raw = input_df
        df_raw['id'] = np.arange(0, len(df))

        df[output_column] = df['response'].apply( 
        lambda x: x['body']['choices'][0]['message']['content'])
        df['id'] = df['custom_id'].apply(lambda x: int(x.split('-')[-1]))
        df.drop(columns=['custom_id'], inplace=True)
        df.drop(columns=['error'], inplace=True)

        combined_df = pd.merge(df_raw, df, on='id', how='left')
        combined_df.to_excel(file_out, index=False)
    except Exception as e:
        gr.Error(f"写入Excel文件失败: {e}")
        return False

    print("Data saved to", file_out)
    gr.Info(f"批处理任务已完成, 数据已存至: {file_out}")
    return True


if __name__ == "__main__":
    # MODEL = 'gpt-4o'
    MODEL = "ft:gpt-4o-mini-2024-07-18:personal::9yGNDJOv"
    TEMPERATURE = 0.3
    URL = '/v1/chat/completions'

    file_in = "data/train_c1_spark_raw.xlsx"
    file_out = "SFT/0820/train_c1_spark_raw.csv"

    file_name = file_in.split('/')[-1].split('.')[0]

    input_jsonl_file = "output/metas/" + file_name + '_batch.jsonl'
    output_jsonl_file = "output/metas/" + file_name + '_batch_output.jsonl'
    
    to_batch_jsonl(file_in, input_jsonl_file, get_prompt_ai_reply, MODEL, URL, TEMPERATURE, json_mode=False)
    batch_id = start_batch_job(input_jsonl_file)
    print("Batch ID:", batch_id)

    # batch_id = "batch_2AZz1v5tW4Vfb7H2IWJpyFsQ"
    time.sleep(1)
    get_batch_results(batch_id, output_jsonl_file)
    format_batch_results(file_in, output_jsonl_file, file_out)
