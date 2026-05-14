# -*- coding: utf-8 -*-
"""测试 DeepSeek API 连接"""
from openai import OpenAI

client = OpenAI(
    api_key='sk-f4114da6625442b2b3013c1a6fb365b2',
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
