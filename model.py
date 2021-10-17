import pickle as pkl
import yfinance as yf
from datetime import date, timedelta, datetime
import time
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
import streamlit as st
from plotly import graph_objs as go

start = time.time()

path_2020 = "IPO-2020.xlsx"
path_2000 = "IPO-2000.xlsx"
pickle_file = open("arima_values_2020.pkl", "rb")
arma_dict = (pkl.load(pickle_file))
error_color = 'rgba(0, 114, 178, 0.2)'
arima_sum = pd.DataFrame()
fbp_sum = pd.DataFrame()
df_2020 = pd.read_excel(path_2020)
df_2000_sum = pd.DataFrame()
df_2020_sum = pd.DataFrame()

fig_compare = go.Figure()



def insert_offer_price(df, offer_price, start_date):
    day_before = start_date + timedelta(days=-1)
    new_row = pd.DataFrame({"Date": day_before, "Close": offer_price}, index=[0])
    return pd.concat([new_row, df]).reset_index(drop=True)


def change_x_value(df, x_name):
    df[x_name] = [x for x in range(len(df))]
    return df


def get_end_date(len_of_data):
    today = date.today()
    if (today.isoweekday()<=5):
        return date.today() + pd.tseries.offsets.BDay(364 - len_of_data)
    else:
        return date.today() + pd.tseries.offsets.BDay(365 - len_of_data)


def plottings(x, y, title, show_year, df):
    fig_plot = go.Figure()
    fig_plot.add_trace(go.Scatter(x=x, y=y))
    fig_plot.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_plot)
    show_df(show_year, df)


def show_df(data_name, df):
    with st.beta_expander("View {} data".format(data_name)):
        st.write(df)


def arima_plot_and_predict(values, data, ticker, offer_price, start_date):
    end_date = get_end_date(len(data))
    print(len(data))
    predictions = ARIMA(data, order=(int(values[0]), int(values[1]), int(values[2]))).fit().predict(start=len(data) + 1,
                                                                                                  end=365)
    df = pd.DataFrame({"Date": (pd.bdate_range(start=date.today(), end=end_date, freq="B")), "Close": predictions})
    data = data.to_frame()
    data.reset_index(inplace=True)
    data = insert_offer_price(data, float(offer_price), start_date)
    df = pd.concat([data, df])
    arima_sum[ticker] = df["Close"][:365]
    print(arima_sum[ticker])
    return df


def fbprophet_plot_and_predict(data, ticker, offer_price, start_date):
    data.reset_index(inplace=True)
    train_data = data
    train_data = train_data.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet(interval_width=0.95)
    m.fit(train_data)
    future = m.make_future_dataframe(periods=366 - len(train_data))  # yeah not a clue here
    forecast = m.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]][len(train_data):int(365)]
    end_date = get_end_date(len(train_data))
    forecast["ds"] = pd.bdate_range(start=date.today(), end=end_date, freq="B")
    forecast = forecast[1:]
    data = insert_offer_price(data, float(offer_price), start_date)
    data = data.rename(columns={"Close": "yhat", "Date": "ds"})
    df = pd.concat([data, forecast])  # why is th is here - you tell me
    fbp_sum[ticker] = df["yhat"]
    print(fbp_sum[ticker])
    return forecast


def get_start_date(date):
    if isinstance(date, datetime):
        return date
    else:
        return datetime.strptime(date, "%m/%d/%Y")


def forecast_2000():
    if(view_dot_com):
        st.header("2000 Stock Data")
        progress_2000 = st.progress(0)
        df_2000 = pd.read_excel(path_2000)
        fig_2000 = go.Figure()
        for x in range(len(df_2000)):
            ticker = df_2000["Ticker"][x]
            start_date = get_start_date(df_2000["Date"][x])
            end_date = start_date + pd.tseries.offsets.BDay(400)
            data = yf.download(ticker, start_date, end_date)["Close"].to_frame()
            if ((len(data) >= 365)):
                print(len(data))
                data.reset_index(inplace=True)
                data = insert_offer_price(data, df_2000["Price"][x], start_date)[:365]
                fig_2000.add_trace(go.Scatter(x=[x for x in range(365)], y=data["Close"], name=ticker))
                df_2000_sum[ticker] = data["Close"][:365]
            progress_2000.progress(((100 / len(df_2000)) * (x + 1) / 100))
        fig_2000.layout.update(title_text="2000 Stock Trend", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_2000)
        show_df("2000",df_2000)
        st.balloons()


def forecast_2020():
    if (non_forecasted_2020 or forecast_arima_2020 or forecast_fbp_2020):
        st.header("2020 Stock Data")
        progress_2020 = st.progress(0)
        for x in range(len(df_2020)):
            print("----------------------------------")
            ticker = df_2020["Ticker"][x]
            start_date = get_start_date(df_2020["Start"][x])
            print(ticker)
            data = yf.download(ticker, start_date, datetime.today())["Close"]
            if len(data) < 365 and (forecast_arima_2020 or forecast_fbp_2020):
                if ticker in arma_dict:
                    order_values = arma_dict[ticker][0]
                    offer_price = arma_dict[ticker][2]
                    if (forecast_arima_2020):
                        arma = arima_plot_and_predict(order_values, data, ticker, offer_price, start_date)
                        fig_arima.add_trace(go.Scatter(x=arma["Date"], y=arma["Close"], name=ticker))
                    if (forecast_fbp_2020):
                        fbp = fbprophet_plot_and_predict(data.to_frame(), ticker, offer_price, start_date)
                        fig_fb.add_trace(go.Scatter(
                            name=ticker,
                            x=fbp['ds'],
                            y=fbp['yhat'],
                            mode='lines',
                            line=dict(color='#0072B2', width=2),
                            fillcolor=error_color,
                        ))
            else:
                if (len(data) >= 365 and non_forecasted_2020):
                    data = data.to_frame()
                    data.reset_index(inplace=True)
                    data = insert_offer_price(data, float(df_2020["Median/Offer Price "][x]), start_date)
                    fbp_sum[ticker] = data["Close"][:365]
                    arima_sum[ticker] = data["Close"][:365]
                    df_2020_sum[ticker] = data["Close"][:365]
                    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name=ticker))
                # heres all the big boys for fbprophet cause it likes to be a pain - these just make it more confusing visually
                # fig.add_trace(go.Scatter(x=fbp['ds'],y=fbp['yhat_upper'], mode='lines',line=dict(width=0),fillcolor=error_color,fill='tonexty',hoverinfo='skip'))
                # fig.add_trace(go.Scatter( x=fbp['ds'],y=fbp['yhat_lower'], mode='lines',line=dict(width=0),fillcolor=error_color,fill='tonexty',hoverinfo='skip'))))
            progress_2020.progress(((100 / len(df_2020)) * (x + 1) / 100))
    if (non_forecasted_2020):
        fig.layout.update(title_text="2020 Stock Trend", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        show_df("2020 Stock Trend", df_2020_sum)
    if (forecast_fbp_2020):
        fig_fb.layout.update(title_text="2020 FBP Trend", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_fb)
        show_df("2020 FBP Trend", fbp_sum)
    if (forecast_arima_2020):
        fig_arima.layout.update(title_text="2020 Arima Trend", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_arima)
        show_df("2020 ARIMA Trend", arima_sum)
    st.balloons()


st.title(
    "EPQ: How successful is statistical forecasting to identify the similar patterns and trends between 2020 IPOs and the 2000 Dot Com Bubble?")

fig_fb = go.Figure()
fig_arima = go.Figure()
fig = go.Figure()

options_container = st.empty()
with options_container:
    column_1, column_2, column_3,column_4 = st.beta_columns(4)
    with column_1:
        non_forecasted_2020 = st.checkbox("Non Forecasted 2020 Stocks", value=True)
    with column_2:
        forecast_arima_2020 = st.checkbox("Forecasted 2020 Stocks with ARIMA")
    with column_3:
        forecast_fbp_2020 = st.checkbox("Forecasted 2020 Stocks with FBProphet")
    with column_4:
        view_dot_com = st.checkbox("Actual 2000 Dot Com Stocks",value=True)

# wait for user input:
element = st.empty()
for i in range(5, 0, -1):
    time.sleep(1)
    element.write("Waiting for selections to be made. {} seconds left! ⌛".format(i))
element.empty()
options_container.empty()

forecast_2020()
forecast_2000()

# plot funds
list_365 = [x for x in range(365)]
diffs = pd.DataFrame()
diffs["days"] = list_365[1:]
if (non_forecasted_2020):
    df_2020_sum["sum"] = df_2020_sum.sum(axis=1)
    df_2020_sum["days"] = list_365
    plottings(list_365, df_2020_sum["sum"], "2020 Fund (Actual)", "2020", df_2020_sum[["days", "sum"]])
    diffs["2020"] = df_2020_sum["sum"].pct_change()[1:]
    fig_compare.add_trace(go.Scatter(x=list_365[1:], y=diffs["2020"], name="2020"))
if (forecast_arima_2020):
    arima_sum["sum"] = arima_sum.sum(axis=1)
    arima_sum["days"] = list_365
    plottings(list_365, arima_sum["sum"], "2020 Fund (ARIMA)", "2020", arima_sum[["days", "sum"]])
    diffs["ARIMA"] = arima_sum["sum"].pct_change()[1:]
    fig_compare.add_trace(go.Scatter(x=list_365[1:], y=diffs["ARIMA"], name="ARIMA"))
if (forecast_fbp_2020):
    fbp_sum["sum"] = fbp_sum.sum(axis=1)
    fbp_sum["days"] = list_365
    plottings(list_365, fbp_sum["sum"], "2020 Fund (FBP)", "2020", fbp_sum[["days", "sum"]])
    diffs["FBP"] = fbp_sum["sum"].pct_change()[1:]
    fig_compare.add_trace(go.Scatter(x=list_365[1:], y=diffs["FBP"], name="FBP"))
if(view_dot_com):
    df_2000_sum["sum"] = df_2000_sum.sum(axis=1)
    df_2000_sum["days"] = list_365
    plottings(list_365, df_2000_sum["sum"], "2000 Fund (Actual)", "2000", df_2000_sum[["days", "sum"]])
    diffs["2000"] = df_2000_sum["sum"].pct_change()[1:]
    fig_compare.add_trace(go.Scatter(x=list_365[1:], y=diffs["2000"], name="Dot Com"))

# FINAL GRAPH?
fig_compare.layout.update(title_text="Comparison Graphs", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_compare)
show_df("Differentials", diffs)

# get prediction length values so sum = 365 - DONE
# do the same with fbprohet
# collect 2000 dot com tickers and get their 365 movement - DONE
# streamlit the bitch - DONE

col_left, col_right = st.beta_columns(2)
with col_left:
    if(view_dot_com):
        plottings(list_365, df_2000_sum["sum"], "2000 Fund (Actual)", "2000", df_2000_sum[["days", "sum"]])
with col_right:
    if(forecast_arima_2020):
        plottings(list_365, arima_sum["sum"], "2020 Fund (ARIMA)", "2020", arima_sum[["days", "sum"]])

print(time.time() - start)
