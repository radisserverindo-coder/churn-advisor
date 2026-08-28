import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from knowledge.rag_engine import retrieve_chunks, ask_openrouter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Telecom Churn Advisor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA AND MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df_raw = load_data()
model = load_model()

# Prepare active customers (we only want to predict for those who haven't churned yet if we are a retention team,
# but for demonstration with this dataset, let's predict for everyone or just use the current features.
# To make it realistic, we'll pretend all customers in the dataset are 'current' for the sake of the dashboard 
# and use the model to find the riskiest ones.)
# Actually, the dataset already has a "Churn" column indicating historical churn.
# We will evaluate risk for active customers (Churn == 'No') to prevent them from churning.
df_active = df_raw[df_raw['Churn'] == 'No'].copy()

# Add Churn Probability column
# Drop customerID and Churn for prediction
X_active = df_active.drop(columns=['customerID', 'Churn'])
probabilities = model.predict_proba(X_active)[:, 1] # Probability of Churn (Class 1)
df_active['Churn Probability'] = probabilities
df_active['Risk Level'] = pd.cut(df_active['Churn Probability'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])

# ============================================================
# PROMPT INJECTION DEFENSE
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?system\s+prompt",
    r"reveal\s+(the\s+)?api\s+key",
    r"show\s+(me\s+)?(the\s+)?api\s+key",
    r"print\s+(the\s+)?api\s+key",
    r"reveal\s+(secret|secrets)",
    r"show\s+(secret|secrets)",
]


def is_prompt_injection(text):
    """Detect common prompt injection attempts."""
    text_lower = text.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    return False

# --- SIDEBAR ---
st.sidebar.title("📡 Telecom Churn Advisor")
st.sidebar.markdown("Internal CRM Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview & KPIs",
        "🎯 Retention Action List",
        "👤 Customer Profile & Simulation",
        "🤖 AI Retention Advisor"
    ]
)
# --- PAGE 1: OVERVIEW & KPIs ---
if page == "📊 Overview & KPIs":
    st.title("📊 Business Overview & KPIs")
    
    # KPIs
    total_customers = len(df_raw)
    churned_customers = len(df_raw[df_raw['Churn'] == 'Yes'])
    churn_rate = churned_customers / total_customers * 100
    monthly_revenue = df_raw['MonthlyCharges'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Monthly Revenue", f"${monthly_revenue:,.2f}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Churn by Contract Type
        churn_contract = df_raw.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig1 = px.bar(churn_contract, x='Contract', y='Count', color='Churn', barmode='group',
                      title="Churn vs Retention by Contract Type",
                      color_discrete_map={'Yes': '#EF553B', 'No': '#00CC96'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        # Churn by Internet Service
        churn_internet = df_raw.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
        fig2 = px.bar(churn_internet, x='InternetService', y='Count', color='Churn', barmode='group',
                      title="Churn by Internet Service",
                      color_discrete_map={'Yes': '#EF553B', 'No': '#00CC96'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("### 🔍 Risk Distribution (Active Customers)")
    risk_counts = df_active['Risk Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    fig3 = px.pie(risk_counts, values='Count', names='Risk Level', 
                  title='Active Customers by Churn Risk',
                  color='Risk Level',
                  color_discrete_map={'Low': '#00CC96', 'Medium': '#FFA15A', 'High': '#EF553B'},
                  hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)

    # --- CHURN PREDICTION MODEL ---
    st.markdown("### 🎯 Churn Prediction Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Model", "Logistic Regression")

    with col2:
        st.metric("F1 Score", "0.6136")

    with col3:
        st.metric("ROC-AUC", "0.8242")


# --- PAGE 2: RETENTION ACTION LIST ---

# --- PAGE 2: RETENTION ACTION LIST ---
elif page == "🎯 Retention Action List":
    st.title("🎯 Retention Action List")
    st.markdown("Identify high-risk customers and deploy targeted retention strategies.")
    
    # Filter for High/Medium risk
    min_prob = st.slider("Minimum Churn Probability Filter", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    df_risky = df_active[df_active['Churn Probability'] >= min_prob].sort_values(by='Churn Probability', ascending=False)
    
    # Generate Recommendations
    def get_recommendation(row):
        if row['Contract'] == 'Month-to-month':
            return "Offer 1-Year Contract with 10% Discount"
        elif row['InternetService'] == 'Fiber optic':
            return "Provide Free Tech Support Add-on"
        else:
            return "Offer $10 Monthly Credit"
            
    df_risky['Recommendation'] = df_risky.apply(get_recommendation, axis=1)
    
    display_cols = ['customerID', 'Churn Probability', 'Risk Level', 'Contract', 'MonthlyCharges', 'Recommendation']
    
    st.write(f"Showing **{len(df_risky)}** customers with Churn Probability >= {min_prob:.0%}")
    
    st.dataframe(
        df_risky[display_cols].style.format({'Churn Probability': '{:.1%}', 'MonthlyCharges': '${:.2f}'})\
        .map(lambda x: 'background-color: #ffcccc' if x == 'High' else ('background-color: #fff2cc' if x == 'Medium' else ''), subset=['Risk Level']),
        use_container_width=True,
        hide_index=True
    )

# --- PAGE 3: CUSTOMER PROFILE & SIMULATION ---
elif page == "👤 Customer Profile & Simulation":
    st.title("👤 Customer Profile & Simulation")
    st.markdown("Search for a customer and simulate how different offers impact their churn probability.")
    
    customer_ids = df_active['customerID'].tolist()
    selected_id = st.selectbox("Search Customer ID", options=[""] + customer_ids, index=0)
    
    if selected_id:
        cust_data = df_active[df_active['customerID'] == selected_id].iloc[0]
        
        st.subheader(f"Customer: {selected_id}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Contract", cust_data['Contract'])
        col2.metric("Monthly Charges", f"${cust_data['MonthlyCharges']:.2f}")
        col3.metric("Current Churn Risk", f"{cust_data['Churn Probability']:.1%}")
        
        st.markdown("### 🛠️ Retention Offer Simulator")
        st.info("Adjust the parameters below to see how a retention offer affects the customer's churn probability.")
        
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            new_contract = st.selectbox("Simulate New Contract", 
                                        options=['Month-to-month', 'One year', 'Two year'], 
                                        index=['Month-to-month', 'One year', 'Two year'].index(cust_data['Contract']))
            
        with sim_col2:
            discount = st.slider("Simulate Monthly Discount (%)", min_value=0, max_value=50, value=0, step=5)
            
        # Recalculate Probability
        if st.button("Run Simulation"):
            # Create a copy of the customer's features
            sim_data = cust_data.drop(['customerID', 'Churn', 'Churn Probability', 'Risk Level']).to_frame().T
            
            # Apply simulations
            sim_data['Contract'] = new_contract
            sim_data['MonthlyCharges'] = sim_data['MonthlyCharges'] * (1 - (discount/100))
            
            # Predict
            new_prob = model.predict_proba(sim_data)[:, 1][0]
            prob_diff = new_prob - cust_data['Churn Probability']
            
            st.markdown("#### Simulation Results")
            
            r_col1, r_col2 = st.columns(2)
            r_col1.metric("Simulated Monthly Charges", f"${sim_data['MonthlyCharges'].values[0]:.2f}", f"-${(cust_data['MonthlyCharges'] * discount/100):.2f}")
            r_col2.metric("New Churn Risk", f"{new_prob:.1%}", f"{prob_diff*100:.1f}%", delta_color="inverse")
            
            if prob_diff < 0:
                st.success(f"✅ This offer successfully reduces the churn risk by {abs(prob_diff)*100:.1f}%.")
            else:
                st.warning("⚠️ This offer does not significantly reduce the churn risk.")
# --- PAGE 4: AI RETENTION ADVISOR ---
elif page == "🤖 AI Retention Advisor":

    st.title("🤖 AI Retention Advisor")

    st.markdown(
        "Ask the AI advisor about customer retention strategies, "
        "churn risk, contracts, discounts, and recommended actions."
    )

    st.markdown("---")

    question = st.text_area(
        "💬 Ask your question",
        placeholder=(
            "Example: What strategy should we use for a "
            "high-risk month-to-month customer?"
        ),
        height=120
    )

    if st.button("🚀 Ask AI Advisor"):

        if not question.strip():
            st.warning("Please enter a question first.")

        elif is_prompt_injection(question):
            st.error(
                "🛡️ Prompt injection detected. "
                "The request was blocked because it attempts "
                "to override instructions or access protected information."
            )

            st.info(
                "Please ask a question related to customer churn, "
                "retention, contracts, discounts, or customer strategy."
            )

        else:
            # OpenRouter API key
            try:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
            except Exception:
                api_key = None

            if not api_key:
                api_key = os.getenv("OPENROUTER_API_KEY")

            if not api_key:
                st.error(
                    "OPENROUTER_API_KEY belum tersedia. "
                    "Tambahkan API key melalui Streamlit Secrets."
                )

            else:
                # ------------------------------------------------
                # RAG ROUTER
                # ------------------------------------------------

                rag_keywords = [
                    "retention",
                    "retain",
                    "churn",
                    "contract",
                    "discount",
                    "customer",
                    "fiber",
                    "internet",
                    "month-to-month",
                    "one year",
                    "two year",
                    "strategy",
                    "offer",
                    "risk",
                    "customer service"
                ]

                question_lower = question.lower()

                needs_rag = any(
                    keyword in question_lower
                    for keyword in rag_keywords
                )

                retrieved_chunks = []

                if needs_rag:
                    retrieved_chunks = retrieve_chunks(
                        question,
                        top_k=3
                    )

                context = "\n\n".join(
                    chunk["text"]
                    for chunk in retrieved_chunks
                )

                # ------------------------------------------------
                # CALL LLM
                # ------------------------------------------------

                with st.spinner("🤖 AI sedang menganalisis..."):
                    try:
                        result = ask_openrouter(
                            question=question,
                            context=context,
                            model="openai/gpt-4o-mini",
                            api_key=api_key
                        )

                        # ------------------------------------------------
                        # RESULT
                        # ------------------------------------------------

                        st.markdown("### 💡 AI Recommendation")
                        st.success(result["answer"])

                        st.markdown("---")
                        st.markdown("### 🔎 AI Execution Details")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "RAG Used",
                            "Yes" if needs_rag else "No"
                        )

                        col2.metric(
                            "Chunks Retrieved",
                            len(retrieved_chunks)
                        )

                        col3.metric(
                            "Model Used",
                            result["model"]
                        )

                        # ------------------------------------------------
                        # TOKEN USAGE
                        # ------------------------------------------------

                        usage = result.get("usage", {})

                        if usage:
                            st.markdown("### 🪙 Token Usage")

                            token_col1, token_col2, token_col3 = st.columns(3)

                            token_col1.metric(
                                "Prompt Tokens",
                                usage.get("prompt_tokens", 0)
                            )

                            token_col2.metric(
                                "Completion Tokens",
                                usage.get("completion_tokens", 0)
                            )

                            token_col3.metric(
                                "Total Tokens",
                                usage.get("total_tokens", 0)
                            )

                        # ------------------------------------------------
                        # SOURCES
                        # ------------------------------------------------

                        if retrieved_chunks:
                            st.markdown("### 📚 Retrieved Knowledge")

                            for i, chunk in enumerate(
                                retrieved_chunks,
                                start=1
                            ):
                                with st.expander(
                                    f"Knowledge Chunk {i}"
                                ):
                                    st.write(chunk["text"])
                                    st.caption(
                                        f"Relevance score: {chunk['score']}"
                                    )
                        else:
                            st.info(
                                "ℹ️ RAG tidak digunakan karena "
                                "pertanyaan tidak membutuhkan knowledge base."
                            )

                    except Exception as e:
                        st.error(
                            f"Terjadi error saat menghubungi AI: {e}"
                        )

