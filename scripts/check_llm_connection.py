# -*- coding: utf-8 -*-
"""测试 DeepSeek API 连接"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
    base_url='https://api.deepseek.com/v1'
)

try:
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[{'role': 'user', 'content': '你好'}],
        max_tokens=10
    )
    print('调用成功:', response.choices[0].message.content)
except Exception as e:
    print('调用失败:', type(e).__name__, e)
