
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import yfinance as yf


# Streamlit 앱 제목
st.title("세븐스플릿")

# "Hello World" 출력


# 화면 1 : 세븐스플릿 소개
st.header("1. 세븐스플릿을 소개합니다.")

c1, c2 = st.columns(2)

with c1 :
    button1 = st.button("다음")
    if button1 :
        st.write("세븐스플릿 전략의 백테스팅 페이지입니다.")
            
    else :
        st.write("세븐스플릿 전략의 백테스팅 페이지입니다.")
        st.write("조회가능한 종목의 목록은 다음과 같습니다.")
        # 상장사 목록조회
        cl_needed_at_list = ["Code", "Name", "Market", "Marcap"]
        listed_stock_list = fdr.StockListing("KRX")[cl_needed_at_list]
        listed_stock_list.reindex()
        st.write(listed_stock_list)     

with c2 :
    st.write("조회가능한 종목의 목록은 다음과 같습니다.")
    # 상장사 목록조회
    cl_needed_at_list = ["Code", "Name", "Market", "Marcap"]
    listed_stock_list = fdr.StockListing("KRX")[cl_needed_at_list]
    listed_stock_list = listed_stock_list[listed_stock_list["Market"]!="KONEX"]
    listed_stock_list.reindex()
    st.write(listed_stock_list)  



# 화면2-1 : 데이터 수집 방식 선택


st.header("2. 데이터를 수집합니다.")
way_to_crawling = st.radio(
        "",
        ["여기서 다운로드하겠습니다.", "직접 엑셀파일을 업로드하겠습니다."])

col1, col2 = st.columns(2)

if way_to_crawling == "직접 엑셀파일을 업로드하겠습니다." :
    with col1 :
        uploaded_data = st.file_uploader("")
    with col2 :
        st.write("업로드한 데이터입니다.")
        df = pd.read_csv(uploaded_data)
        st.write(df)
        
        
       

else :         
    with col1 :
        #select_interval = ['1분', '2분', '5분', '15분', '30분', '1시간', '1일']
        st.write("*조회할 데이터의 정보를 입력해주세요.")
        select_interval = ['2분', '1일']    
        interval = st.selectbox("주가데이터 빈도", select_interval).replace('분',"m").replace('일','d')
        stock_code = st.text_input("종목코드","005930")        
        import datetime
        now = datetime.date.today()
        two_m_ago = now - datetime.timedelta(days=50)
        start_date, end_date = st.date_input("시작날짜-종료날짜", (two_m_ago,now), min_value=two_m_ago)
        market = listed_stock_list[listed_stock_list["Code"]==stock_code]["Market"].values[0]
        stock_name = listed_stock_list[listed_stock_list["Code"]==stock_code]["Name"].values[0]


    with col2 : 
        # 화면 2-2 : 주가 데이터 조회
        def from_yf(stock_code, market,start_date="", end_date="", interval="2m") :  
            
            #yfinace의 시장이름 포맷에 맞도록 수정
            market = market.replace("KOSPI", "KS")
            market = market.replace("KOSDAQ", "KQ")    
            # 주가조회
            ticker = yf.Ticker(f'{stock_code}.{market}')    
            try :
                df = ticker.history(
                    interval=interval,
                    start=start_date,
                    end=end_date)    
            # 결측치 제거
                if df.isna().any().any() :
                    df.fillna(method="ffill")            
                # 변화율 포함
                df["Change"] = np.log(df["Close"]) - np.log(df["Close"].shift(-1))
                df = df.iloc[:-1]
            except Exception as e :
                print(e)
            return df[["Close", "Change"]]

        df = from_yf(stock_code, market, start_date, end_date, interval)  

        st.write("* 조회한 데이터입니다.", df)


# 화면 3 : 투자전략

# input 데이터프레임 작성
c1,c2,c3 = st.columns(3)

st.header("3. 거래전략을 수립합니다.")
with c1:
    st.write("스플릿 갯수")    
    Controling_Number = st.number_input("최대 스플릿의 갯수 : ", 1,50, 7)
    initial_balance = st.number_input("시작금액",min_value=10000, step=10000, value=1000000)  # 10억들고 시작 가정
    st.write(f"{initial_balance:,.0f}원으로 시작합니다.")
with c2:
    st.write("스플릿 조건")    
    up = st.number_input("u% 오르면 익절", min_value=1.,max_value=5.,value=2.,step=0.1)
    down = st.number_input("d% 오르면 다음회차 매수", min_value=1.,max_value=5.,value=2.,step=0.1)
    u = [up / 100] * Controling_Number # 일정하게 5% 상승 시 익절 가정
    d = [down / 100] * Controling_Number # 일정하게 5% 하락 시 추매 가정
with c3 :
    st.write("거래비용")
    transaction_fee = st.number_input("증권사 거래수수료(%)", min_value=0.,max_value=1.,value=0.015,step=0.001) / 100
    tax = st.number_input("증권거래세(%)", min_value=0.,max_value=1.,value=0.05,step=0.01) / 100
    tax2 = st.number_input("농어촌특별세(%)", min_value=0.,max_value=1.,value=0.15,step=0.01) / 100

st.write("손님의 거래전략은 다음과 같습니다.")
st.write(f"1. {stock_name} ({stock_code})에 {interval}간격으로 {start_date}부터 {end_date} 기간동안 투자합니다.")
st.write("2. 스플릿별 관리 전략은 아래표와 같습니다.")
input_cl = [f"{i+1}th" for i in range(Controling_Number)]
input_idx = ["할당 주식 수", "u(%)", "d(%)"]
df_input = pd.DataFrame(columns=input_cl, index=input_idx)


balance=[int(initial_balance // df["Close"][0] // Controling_Number)] * Controling_Number # 매 회차 동일수량 매수 가정

df_input.loc["할당 주식 수"] = balance
df_input.loc["u(%)"] = u
df_input.loc["d(%)"] = d

st.write(df_input)
st.write("* 회차 별 할당 주식 수 = ( 시작금액 / 현재 주가 ) / 스플릿 갯수")


# 화면 4 (숨김) : 백테스팅



# 관리 회차 부여
for i in range(Controling_Number+1):
    df[f"{i}th"] = [np.NAN]*len(df)

df["buy"]=[np.NAN]*len(df)
df["sell"]=[np.NAN]*len(df)
df["position"]=[np.NAN]*len(df)
df["sell"]=[np.NAN]*len(df)
df["ith"] = [np.NAN]*len(df)
df["id"] = [np.NAN]*len(df)

# 첫날 수동거래
df["1th"][0]=df["Close"][0] # 조회시점부터 구매가정.
df["buy"][0]=df["Close"][0]
df["position"][0] = -1
df["ith"][0] = 1.
df["id"][0] = 1


# 반복
indexer = list(df.columns).index("1th") # 1th 칼럼의 위치 
indexer_end = list(df.columns).index(f"{Controling_Number}th")
controlling = 0
going1 = [np.nan] * Controling_Number
going1[0] = df["Close"][0]
going2 = df["Close"][0]

for i in range(len(df)) :    


    going2 = np.min(going1[:controlling+1])
    up = going2 * (1+u[controlling])    
    down = going2 * (1-d[controlling])

    #print(i, controlling, going2)

    # 매수
    if df["Close"][i] <= down :
        df["position"][i] = -1
        df.iat[i,indexer+controlling+1] = df["Close"][i]
        df["buy"][i] = df["Close"][i]
        df["ith"][i] = controlling+2
        
        going1[controlling+1] = df["Close"][i]     
        controlling += 1
    
    # 매도
    elif df["Close"][i] >= up :
        if controlling != 0 :
                
            df["position"][i] = 1
            df.iat[i,indexer+controlling] = df["Close"][i]
            df["sell"][i] = df["Close"][i]
            df["ith"][i] = controlling+1
            going1[controlling] = np.nan   

            controlling -= 1   


# 화면 5 : 투자이력 표시

st.header("4. 투자이력을 표시합니다.")  
st.write("* 투자지평의 마지막날 남은 보유분에서는 1분간격으로 청산을 가정합니다.")


# 청산가정
from datetime import timedelta

df_settled = df.copy()
agg =[]
a = 1

while controlling >= 0 :
    print(controlling)
    fron = pd.DataFrame(df.iloc[-1,:]).T
    fron["position"] = 1 # 매도
    fron[f"{controlling+1}th"] = fron["Close"]
    fron["sell"] = fron["Close"]
    fron["ith"] = controlling+1
    
    fron.index = [fron.index[0] + timedelta(minutes=a)]
    agg.append(fron)
    controlling -= 1
    a += 1

df_settled = pd.concat([df_settled, *agg])
df_settled.tail()
        
# 청산가정 _ history 채우기
fron = pd.melt(df_settled, "ith",("buy","sell"))
fron = fron[fron["ith"].notna()]
fron = fron[fron['value'].notna()]
fron.sort_values('ith').head(10)

idx = []
buy_price = []
sell_price = []
ith = []

for i in fron['ith'].unique():
    temp= fron[fron['ith']==i]

    a = temp[temp['variable']=='buy']
    b = temp[temp['variable']=='sell']
    ith += [i]*len(a)


    idx += list(a.index)
    buy_price += list(a["value"].values)
    sell_price += list(b["value"].values)


df_history = pd.DataFrame()
df_history["Buy At"] = buy_price
df_history["Sold At"] = sell_price
df_history.index = df_settled.index[idx]
df_history['ith'] = ith
df_history["Amount"] = [df_input[f"{int(i)}th"]["할당 주식 수"] for i in ith]
df_history['fee'] = (df_history["Buy At"]*transaction_fee + df_history["Sold At"]*(tax+tax2+transaction_fee)) * df_history["Amount"]
df_history["Money Earned"] = (df_history["Sold At"] -df_history["Buy At"] )  * df_history["Amount"] - df_history['fee']
df_history["Money Earned(%)"] = (np.log(df_history["Sold At"] - df_history['fee']) - np.log(df_history["Buy At"]))

st.write(df_history)



# 화면 6 : 시각화합니다.

# plotly 그래프 그리기
st.header("5. 시각화합니다.")
import plotly
import plotly.graph_objects as go
df_settled_ = df_settled.copy()
name = listed_stock_list[listed_stock_list["Code"]==stock_code]["Name"].values[0]
layout = go.Layout(title=f'{name}({stock_code})')
fig = go.Figure(layout=layout)
fig.add_trace(
    go.Scatter(x=df_settled_.index, y=df_settled_['Close'], line = dict(color='white', width=0.7), name='Close'))
fig.add_trace(
    go.Scatter(x=df_settled_.index, y=df_settled_['buy'], mode='markers', name='Buy', marker = dict(color='red', size=10)))
fig.add_trace(
    go.Scatter(x=df_settled_.index, y=df_settled_['sell'], mode='markers', name='Sell', marker = dict(color='blue', size=10)))
# x축 옵션 추가
fig.update_xaxes(
    type='category')
st.write(fig)




# 화면 7 : 전략평가
st.header("6. 전략을 평가합니다.")

# 표 정의
cl = ["Initial Balance", "Final Balance", "CAGR", "Return(%)", "Buy&Hold return", "KOSPI200 Return", "KOSDAQ Return", 
      "Standard Deviation"]
df_res = pd.DataFrame(columns=cl)




# Initial Balance
df_res["Initial Balance"] = [f"{initial_balance:,.3f}"]

# Finanl Balance
final_balance = initial_balance + df_history.loc[:,'Money Earned'].sum()
df_res["Final Balance"] = f"{final_balance:,.3f}"

# CAGR 계산
from datetime import datetime
try :
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
except Exception as e :
    pass
period = (end_date - start_date)
seconds_per_year = 365.25 * 24 * 3600  # 1년의 초 수 (대략적으로 계산)
period = period.total_seconds() / seconds_per_year
CAGR = (np.log(final_balance)-np.log(initial_balance) +1) ** (1/period) -1  
df_res["CAGR"] = CAGR

# Strategy Return 계산
strategy_return = np.log(final_balance) - np.log(initial_balance)
df_res["Return(%)"] = f"{strategy_return*100:.3f}%"

# Buy&Hold 전략 return 계산
buy_hold_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
df_res["Buy&Hold return"] = f"{buy_hold_return*100:.3f}%"

# 지수 return 계산
#KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
#KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
df_res["Standard Deviation"] = f"{df['Change'].std()*100:.3f}%"

# 표시
st.write(df_res)
st.write("* 거래비용을 포함하고 있습니다.")
st.write("* kospi, kosdaq, standard deviation (daily), MDD 구하는데 자료 빈도가 안 맞아서 향후 업데이트 예정")
st.write(f"* 주당 가치가 계속 바뀌기 때문에 initial balance와 실제 매수금액은 차이가 있다. (매 회 {balance[0]}주 매매)")




# 화면8 : 전략수정
st.header("7. 전략을 수정합니다.")
st.write("업데이트 예정")


