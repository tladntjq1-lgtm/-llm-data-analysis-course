import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 페이지 설정
st.set_page_config(
    page_title="주문 취소 예측 대시보드",
    page_icon="📦",
    layout="wide"
)

# 1. 모델 및 설정 불러오기
@st.cache_resource
def load_model():
    model_path = 'models/cancel_model_pipeline.pkl'
    config_path = 'models/threshold_config.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()
        
    pipeline = joblib.load(model_path)
    threshold_config = joblib.load(config_path) if os.path.exists(config_path) else {'selected_threshold': 0.5}
    return pipeline, threshold_config.get('selected_threshold', 0.5)

pipeline, default_threshold = load_model()

# 메인 타이틀
st.title("📦 고객 주문 취소 위험도 예측 시스템")
st.markdown("주문 정보를 입력하거나 슬라이더를 조정하여 **취소 가능성(Probability)**을 실시간으로 측정합니다.")

st.divider()

# 레이아웃 구성
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 주문 입력 데이터")
    
    # 1) 수치형 변수
    item_count = st.number_input("주문 상품 종류 수 (item_count)", min_value=1, max_value=50, value=2)
    total_quantity = st.number_input("총 주문 수량 (total_quantity)", min_value=1, max_value=100, value=3)
    order_amount = st.number_input("총 결제 금액 (order_amount)", min_value=1.0, max_value=1000000.0, value=19000.0, step=1000.0)
    age = st.number_input("고객 나이 (age)", min_value=10, max_value=100, value=30)
    
    # 2) 범주형 변수
    gender = st.selectbox("성별 (gender)", ["M", "F", "Other"])
    city = st.selectbox("도시 (city)", ["Seoul", "Busan", "Incheon", "Daegu", "Gwangju", "Other"])
    customer_state = st.selectbox("주/지역 (customer_state)", ["CA", "NY", "TX", "FL", "Other"])
    
    # 💳 결제 수단 (카카오페이, 네이버페이, 토스페이 등 추가)
    payment_method = st.selectbox(
        "결제 수단 (payment_method)", 
        ["kakaopay", "naverpay", "tosspay", "credit_card", "bank_transfer", "vbank", "paypal"]
    )
    payment_type = st.selectbox(
        "결제 유형 (payment_type)", 
        ["simple_payment", "credit_card", "debit_card", "bank_transfer", "voucher"]
    )

    st.markdown("---")
    threshold = st.slider(
        "⚙️ Decision Threshold (임계값 조정)",
        min_value=0.1,
        max_value=0.9,
        value=float(default_threshold),
        step=0.05,
        help=f"Validation 최적 임계값은 {default_threshold} 입니다."
    )

with col2:
    st.subheader("🎯 예측 결과 (Inference)")
    
    # 모델 학습 시 요구되는 모든 컬럼 구성
    input_data = pd.DataFrame([{
        'item_count': item_count,
        'total_quantity': total_quantity,
        'order_amount': order_amount,
        'age': age,
        'gender': gender,
        'city': city,
        'customer_state': customer_state,
        'payment_method': payment_method,
        'payment_type': payment_type
    }])
    
    try:
        # 확률 예측
        prob = pipeline.predict_proba(input_data)[0][1]
        is_cancelled = prob >= threshold
        
        # 1) 게이지 및 수치 표기
        st.metric(
            label="주문 취소 예측 확률",
            value=f"{prob * 100:.1f}%",
            delta=f"기준 Threshold: {threshold:.2f}",
            delta_color="off"
        )
        st.progress(float(prob))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2) 취소 위험도 예측 결과 배너 (상자)
        if is_cancelled:
            st.error(f"🚨 **[위험] 취소 가능성 높음** (확률 {prob*100:.1f}% ≥ Threshold {threshold})")
            st.warning("💡 **운영 권장 조치**: 출고 일시 중지 후 고객 확인 SMS 발송 또는 배송 지연 여부를 점검하세요.")
        else:
            st.success(f"✅ **[정상] 정상 완료 예상** (확률 {prob*100:.1f}% < Threshold {threshold})")
            st.info("💡 **운영 권장 조치**: 정상 출고 프로세스를 진행하시면 됩니다.")
            
    except Exception as e:
        st.error(f"❌ 예측 도중 오류가 발생했습니다: {e}")