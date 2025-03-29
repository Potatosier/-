#!/usr/bin/env python
# coding: utf-8

!pip install plotly

import math
import streamlit as st
import plotly.graph_objects as go
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# DeepSeek API 密钥
API_KEY = "sk-c7672043a2a748d996489d7cc4852578"  # 请替换成你的密钥

# 风险计算函数
def calculate_fall_risk(TUG_time, gait_speed, stride_time_std, test_surface, BMI):
    x = 0.296 * TUG_time + 0.271 * gait_speed + 0.209 * stride_time_std + 0.08 * test_surface + 0.077 * BMI
    risk = 1 / (1 + math.exp(-x))
    return risk

# 构造 DeepSeek 提示词
def build_prompt(TUG_time, gait_speed, stride_time_std, ts_value, BMI, level):
    ts_label = '室外' if ts_value == 1 else '室内'
    prompt = f"""你是一个老年健康管理专家，请根据以下老年人数据提供个性化的跌倒风险干预建议：
- TUG时间：{TUG_time} 秒
- 步态速度：{gait_speed} m/s
- 步态时间标准差：{stride_time_std}
- 测试表面：{ts_label}
- BMI：{BMI}
系统当前评估风险等级为：{level}

请结合这些数据，生成简洁、具体、可操作的干预建议，帮助其预防跌倒。建议请用中文输出。"""
    return prompt

# 调用 DeepSeek API 获取个性化建议
def get_deepseek_response(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",  # 使用 deepseek-chat 模型
        "messages": [
            {"role": "system", "content": "你是一个专业的老年健康顾问，擅长提供跌倒风险预防建议。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"❌ API调用失败，状态码：{response.status_code}\n{response.text}"

# 仪表盘图
def draw_gauge_chart(risk):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "跌倒风险概率"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.4], 'color': "green"},
                {'range': [0.4, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}],
        }))
    return fig

# 雷达图
def draw_radar_chart(features):
    labels = ['TUG时间', '步态速度', '步态时间标准差', '测试表面', 'BMI']
    raw_values = features

    # 手动归一化 —— 每个特征按经验值设置合理的 min-max 区间
    mins = [0, 0, 0, 0, 10]  # TUG, gait, stride_std, surface, BMI
    maxs = [60, 2, 2, 1, 50]

    # 逐项归一化
    scaled_values = [(v - min_) / (max_ - min_) for v, min_, max_ in zip(raw_values, mins, maxs)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scaled_values + [scaled_values[0]],  # 闭合雷达图
        theta=labels + [labels[0]],            # 闭合雷达图
        fill='toself',
        name='用户指标',
        line_color='blue'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title='跌倒风险影响因素雷达图'
    )
    return fig

# 总体交互函数
def display_risk(TUG_time, gait_speed, stride_time_std, test_surface, BMI):
    # 测试表面转换
    ts_value = 0 if test_surface == "室内" else 1

    # 计算风险
    risk = calculate_fall_risk(TUG_time, gait_speed, stride_time_std, ts_value, BMI)
    st.write(f"预测跌倒风险概率: {risk:.4f}")

    # 风险等级
    if risk < 0.4:
        level = "低风险"
    elif risk < 0.7:
        level = "中风险"
    else:
        level = "高风险"
    st.write(f"风险等级: {level}")

    # 构造 DeepSeek 提示词
    prompt = build_prompt(TUG_time, gait_speed, stride_time_std, ts_value, BMI, level)
    ai_reply = get_deepseek_response(prompt)
    st.write("\n💡 DeepSeek AI建议：")
    st.write(ai_reply)

    # 展示仪表盘和雷达图
    fig_gauge = draw_gauge_chart(risk)
    fig_radar = draw_radar_chart([TUG_time, gait_speed, stride_time_std, ts_value, BMI])

    st.plotly_chart(fig_gauge)
    st.plotly_chart(fig_radar)

# 页面标题
st.title('老年人跌倒风险评估系统')

TUG_time_widget = st.slider('TUG时间(秒)', min_value=0.0, max_value=60.0, step=0.1, value=10.0)
gait_speed_widget = st.slider('步态速度(m/s)', min_value=0.0, max_value=2.0, step=0.01, value=1.0)
stride_time_std_widget = st.slider('步态时间STD(秒)', min_value=0.0, max_value=2.0, step=0.01, value=0.1)
BMI_widget = st.slider('BMI', min_value=10.0, max_value=50.0, step=0.1, value=25.0)

# 为 test_surface_widget 添加一个选择框
test_surface_widget = st.selectbox('测试表面', ['室内', '室外'])

# 计算按钮
if st.button('计算跌倒风险'):
    display_risk(TUG_time_widget, gait_speed_widget, stride_time_std_widget, test_surface_widget, BMI_widget)
