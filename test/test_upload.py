# -*- coding: utf-8 -*-
"""
测试上传功能
"""
import requests
import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = str(Path(__file__).parent.parent)

# 测试文件路径
TEST_FILE = r'C:\语音识别大模型\Whisper-Finetune\AS_Process--data 1000条（教育与学习）\AS_4.wav'

def test_upload_first():
    """测试1: 第一次上传文件"""
    print(f'测试1: 第一次上传文件: AS_4.wav')
    print(f'源文件存在: {os.path.exists(TEST_FILE)}')
    print(f'源文件大小: {os.path.getsize(TEST_FILE)} 字节')

    with open(TEST_FILE, 'rb') as f:
        response = requests.post('http://localhost:8336/audio/upload/', files={'file': ('AS_4.wav', f, 'audio/wav')})

    print(f'状态码: {response.status_code}')
    try:
        print(f'响应: {response.json()}')
    except:
        print(f'响应文本: {response.text[:500]}')
    print()
    return response

def test_upload_again():
    """测试2: 重复上传同一文件（已存在于data/uploads和数据库）"""
    print(f'测试2: 重复上传同一文件: AS_4.wav')
    print(f'文件在uploads目录存在: {os.path.exists(os.path.join(PROJECT_ROOT, "data", "uploads", "AS_4.wav"))}')

    with open(TEST_FILE, 'rb') as f:
        response = requests.post('http://localhost:8336/audio/upload/', files={'file': ('AS_4.wav', f, 'audio/wav')})

    print(f'状态码: {response.status_code}')
    try:
        print(f'响应: {response.json()}')
    except:
        print(f'响应文本: {response.text[:500]}')
    print()
    return response

def test_upload_after_delete():
    """测试3: 删除文件后重新上传（数据库有记录，磁盘无文件）"""
    upload_path = os.path.join(PROJECT_ROOT, 'data', 'uploads', 'AS_4.wav')
    print(f'测试3: 删除文件后重新上传: AS_4.wav')

    # 删除磁盘文件
    if os.path.exists(upload_path):
        os.remove(upload_path)
        print(f'已删除磁盘文件: {upload_path}')
    else:
        print(f'磁盘文件不存在，跳过删除')

    print(f'文件在uploads目录存在: {os.path.exists(upload_path)}')

    with open(TEST_FILE, 'rb') as f:
        response = requests.post('http://localhost:8336/audio/upload/', files={'file': ('AS_4.wav', f, 'audio/wav')})

    print(f'状态码: {response.status_code}')
    try:
        print(f'响应: {response.json()}')
    except:
        print(f'响应文本: {response.text[:500]}')
    print()
    return response

def test_transcribe(audio_id=None):
    """测试4: 转录音频"""
    if audio_id is None:
        print('测试4: 跳过转录（无audio_id）')
        return None

    print(f'测试4: 转录音频 AS_4.wav (id={audio_id})')

    response = requests.post(f'http://localhost:8336/audio/transcribe/{audio_id}')
    print(f'状态码: {response.status_code}')
    try:
        print(f'响应: {response.json()}')
    except:
        print(f'响应文本: {response.text[:500]}')
    print()
    return response

if __name__ == '__main__':
    print('=' * 60)
    print('开始测试')
    print('=' * 60)
    print()

    try:
        # 测试1: 第一次上传
        r1 = test_upload_first()
        audio_id = r1.json().get('audio_id') if r1 and r1.status_code == 200 else None

        # 测试2: 重复上传
        r2 = test_upload_again()

        # 测试3: 删除后重新上传
        r3 = test_upload_after_delete()

        # 测试4: 转录
        r4 = test_transcribe(audio_id)

        print('=' * 60)
        print('测试结果汇总:')
        print(f'  第一次上传: {"成功" if r1 and r1.status_code == 200 else "失败"}')
        print(f'  重复上传: {"成功" if r2 and r2.status_code == 200 else "失败"}')
        print(f'  删除后重新上传: {"成功" if r3 and r3.status_code == 200 else "失败"}')
        print(f'  转录: {"成功" if r4 and r4.status_code == 200 else "失败"}')
        print('=' * 60)

    except Exception as e:
        print(f'测试异常: {e}')
        import traceback
        traceback.print_exc()
