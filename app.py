#Necessary Libraries
import streamlit as st
import plotly.graph_objects as go
from config import stock_symbols,valid_periods,intervals,APP_NAME,APP_VERSION,DEVELOPER,COPYRIGHT
from data import get_live_data
from models import train_models,model_performance_report

st.set_page_config(page_title=APP_NAME, layout="wide",page_icon="assets/nextclose_logo.png")

#sidebar --> inputs
with st.sidebar:
    st.image("assets/nextclose_logo.png",width=100)
    company = st.selectbox("Company: ",options=stock_symbols)
    ticker = stock_symbols[company]
    period = st.selectbox("Period: ",options=valid_periods)
    interval = st.selectbox("Interval: ",options=intervals)
    st.caption(f"""App Version: {APP_VERSION} 
               
Copyright: {COPYRIGHT}

Developer: {DEVELOPER}""")

t1,t2,t3,t4 = st.tabs(["🏠 Home","💲Next Price","📊 Analytics","🚀 Model Performance Report"])

with st.spinner("Training the model..."):
    df = get_live_data(ticker,period,interval)
    #Divide the data
    X = df.drop("Target",axis=1)
    y = df.Target
    results = train_models(X,y)
with t1:
    st.subheader(f"{company}  Live Stock Analysis")
    c1,c2 = st.columns(2)
    with c1.container(border=True,height=340):
        st.write(f"**Data source:** yahoo finance API")
        st.write(f"**Company:** {company}")
        st.write(f"**Period:** {period}")
        st.write(f"**Interval:** {interval}")
    with c2.container(border=True,height=340):
        st.write("##### About app:")
        st.markdown("""
                    NextClose is a real-time stock analysis and prediction platform built using Yahoo Finance live API, Python, Machine Learning, and Streamlit.
                    The goal of this application is to help users:
                    
                    🔹 Track live stock price movements

                    🔹 Analyze historical performance

                    🔹 Compare indicators like Open, Close, High, Low & Volume

                    🔹 Train regression models (Linear, Ridge & Lasso)

                    🔹 Generate near-term stock price predictions

                    🔹 Visualize trends through interactive charts
                    """)
with t4:
    st.write("📊 **Model Performance Evaluation Report:**")
    per_df = model_performance_report(results)
    st.table(per_df)
    best_model = max(results,key=lambda x: results[x]['r2'])
    st.write(f"#### **Best model:** &nbsp;&nbsp;{best_model}")
    best_model = results[best_model]["model_obj"]
    df["prediction"] = best_model.predict(X)

with t2:
    c1,c2 = st.columns([2,1])
    c1.write("Sample data: ")
    c1.dataframe(df.sample(50),height=300)
    next_day_price = best_model.predict(X.tail(1))
    c2.subheader('   ')
    c2.subheader('   ')
    c2.metric("💲**Next Day Price**",value=round(next_day_price[0],ndigits=2),border=True)

with t3:
    data = df.reset_index()
    # st.line_chart(df[["Close","prediction"]])
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],mode='lines',name="Actal Close"))

    fig.add_trace(go.Scatter(x=df.index,y=df["prediction"],mode='lines',name="predicted close"))
    
    fig.update_layout(
        title=f"{ticker}  Actual vs Predicted Close",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig)

        

