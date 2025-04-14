# %%
import math
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
import requests
import numpy as np
from functools import lru_cache

# DeepSeek API 密钥（请替换成你的密钥）
API_KEY = "sk-c7672043a2a748d996489d7cc4852578"

# 加载已训练好的 XGBoost 模型
booster = xgb.Booster()
booster.load_model("xgb_model_tuned.json")

# ----------------------------
# 风险计算函数（注意特征顺序必须与训练时一致）
def calculate_fall_risk(living_nursing, weight_loss, BMI, GDS, gait_speed, TUG_time,
                        balance, frailty, stride_std, cadence_std, heel_strike, test_surface):
    features = np.array([[living_nursing, weight_loss, BMI, GDS, gait_speed, TUG_time,
                           balance, frailty, stride_std, cadence_std, heel_strike, test_surface]])
    dmatrix = xgb.DMatrix(features)
    risk = booster.predict(dmatrix)[0]
    return risk

# ----------------------------
# 构造 DeepSeek 提示词
def build_prompt(living_nursing, weight_loss, BMI, GDS, gait_speed, TUG_time,
                 balance, frailty, stride_std, cadence_std, heel_strike, ts_value, level):
    ln_label = "YES" if living_nursing == 1 else "NO"
    wl_label = "YES" if weight_loss == 1 else "NO"
    ts_label = "Varied" if ts_value == 1 else "Plane"
    prompt = f"""你是一个老年健康管理专家，请根据以下老年人数据提供个性化的跌倒风险干预建议：
- 是否住在养老院: {ln_label}
- 意外体重减轻: {wl_label}
- BMI: {BMI}
- Global Deterioration Scale指数: {GDS}
- 步态速度: {gait_speed} m/s
- TUG测试时间: {TUG_time} 秒
- 平衡测试指数: {balance}
- 衰弱评估指数: {frailty}
- 步态时间标准差: {stride_std} 秒
- 步频标准差: {cadence_std} strides/min
- Heel-Strike角度: {heel_strike} °
- 测试表面: {ts_label}

系统当前评估风险等级为：{level}

请结合这些数据，生成简洁、具体、可操作的干预建议，帮助预防跌倒。建议请用中文输出。"""
    return prompt

# ----------------------------
# 使用缓存机制缓存相同提示词的API响应
@lru_cache(maxsize=32)
def get_deepseek_response(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
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

# ----------------------------
# 绘制仪表盘图（转换为 0～100 分制）
def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_gauge_chart(risk_score):
    # 将风险概率转换为 0～100 分
    score = risk_score * 100

    # 定义三个颜色节点对应的RGB值
    color_low = (60, 179, 113)    # mediumseagreen
    color_mid = (255, 165, 0)       # orange
    color_high = (255, 99, 71)      # tomato

    steps = []
    n_steps = 100
    for i in range(n_steps):
        low = i / n_steps * 100
        high = (i + 1) / n_steps * 100
        # 计算颜色时转换回 0～1 的比例
        mid_value = ((low + high) / 2) / 100
        if mid_value <= 0.7:
            t = mid_value / 0.7
            color_rgb = interpolate_color(color_low, color_mid, t)
        else:
            t = (mid_value - 0.7) / (1 - 0.7)
            color_rgb = interpolate_color(color_mid, color_high, t)
        color_hex = rgb_to_hex(color_rgb)
        steps.append({'range': [low, high], 'color': color_hex})
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "跌倒风险评分 (0-100)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "royalblue", 'thickness': 0.1},
            'steps': steps,
        }
    ))
    return fig

# ----------------------------
# 绘制雷达图（展示所有12个特征的归一化结果）
def draw_radar_chart(features):
    labels = [
        "养老院(0/1)",
        "体重减轻(0/1)",
        "BMI",
        "GDS指数",
        "步态速度(m/s)",
        "TUG时间(s)",
        "平衡指数",
        "衰弱指数",
        "步态STD(s)",
        "步频STD",
        "Heel-Strike(°)",
        "测试表面(0/1)"
    ]
    raw_values = features
    # 根据经验设置最小值和最大值（请根据实际情况调整）
    mins = [0, 0, 15, 1, 0, 5, 0, 0, 0, 0, 0, 0]
    maxs = [1, 1, 40, 7, 2, 30, 4, 5, 0.25, 10, 40, 1]
    scaled_values = [(v - mn) / (mx - mn) if mx != mn else 0 for v, mn, mx in zip(raw_values, mins, maxs)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scaled_values + [scaled_values[0]],
        theta=labels + [labels[0]],
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

# ----------------------------
# 总体交互函数（使用所有12个特征）
def display_risk(living_nursing, weight_loss, BMI, GDS, gait_speed, TUG_time, 
                 balance, frailty, stride_time_std, cadence_std, heel_strike, test_surface):
    ln_value = 1 if living_nursing == "YES" else 0
    wl_value = 1 if weight_loss == "YES" else 0
    ts_value = 1 if test_surface == "Varied" else 0

    # 计算风险（输入特征顺序与训练时一致）
    risk = calculate_fall_risk(ln_value, wl_value, BMI, GDS, gait_speed, TUG_time, 
                                balance, frailty, stride_time_std, cadence_std, heel_strike, ts_value)
    # 将风险转换为百分制
    score = risk * 100
    st.write(f"预测跌倒风险评分: {score:.2f} 分")
    
    if score < 40:
        level = "✅风险较低，请保持良好习惯"
    elif score < 70:
        level = "⚠️中等跌倒风险，建议加强锻炼和定期检查"
    else:
        level = "🔴高跌倒风险，请注意安全"
    st.write(f"风险等级: {level}")

    fig_gauge = draw_gauge_chart(risk)
    fig_radar = draw_radar_chart([ln_value, wl_value, BMI, GDS, gait_speed, TUG_time,
                                  balance, frailty, stride_time_std, cadence_std, heel_strike, ts_value])
    st.plotly_chart(fig_gauge)
    st.plotly_chart(fig_radar)
    
    prompt = build_prompt(ln_value, wl_value, BMI, GDS, gait_speed, TUG_time,
                          balance, frailty, stride_time_std, cadence_std, heel_strike, ts_value, level)
    ai_reply = get_deepseek_response(prompt)
    st.write("💡DeepSeek AI建议：")
    st.write(ai_reply)

# ----------------------------
# Streamlit 页面布局
st.title("老年人跌倒风险评估系统")

# 控件：请确保输入的数值范围与训练数据一致
living_nursing_input = st.selectbox("是否居住在养老院", options=["是", "否"], index=1)
weight_loss_input = st.selectbox("是否经历意外体重减轻", options=["是", "否"], index=1)
BMI_input = st.slider("BMI（体质指数，单位：kg/m^2）", min_value=15.0, max_value=40.0, value=15.0, step=0.01)
GDS_input = st.slider("GDS评分（总体衰退量表，1-7分）", min_value=1, max_value=7, value=1, step=1)
gait_speed_input = st.slider("步行速度(4米步行测试，单位：米/秒)", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
TUG_time_input = st.slider("TUG测试耗时(起立行走时间，单位：秒)", min_value=5.0, max_value=30.0, value=5.0, step=0.01)
balance_input = st.slider("平衡能力评分（0-4分）", min_value=0, max_value=4, value=0, step=1)
frailty_input = st.slider("衰弱评估分数（0-5分）", min_value=0, max_value=5, value=0, step=1)
stride_time_std_input = st.slider("步态时间标准差（单位：秒）", min_value=0.0, max_value=0.25, value=0.0, step=0.001)
cadence_std_input = st.slider("步频标准差（单位：步/分钟）", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
heel_strike_input = st.slider("脚跟着地角度(脚与地面夹角，单位：度)", min_value=0.0, max_value=40.0, value=0.0, step=0.01)
test_surface_input = st.selectbox("测试表面", options=["室外", "室内"], index=0)

if st.button("计算跌倒风险"):
    display_risk(living_nursing_input, weight_loss_input, BMI_input, GDS_input,
                 gait_speed_input, TUG_time_input, balance_input, frailty_input,
                 stride_time_std_input, cadence_std_input, heel_strike_input, test_surface_input)

# %%

