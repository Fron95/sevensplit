
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import yfinance as yf
from streamlit_extras.switch_page_button import switch_page



    

    
    













# 화면 2 : 자료수집



#!
def start() :
    st.write("*조회할 데이터의 정보를 입력해주세요.")
    
    # 코드 (수동)
    stock_code = st.text_input("종목코드를 입력해주십시오", value='005930')
    st.caption("예시)삼성전자일 경우 '005930' 입력")
    # 수집방식 (수동)
    way_to_crawling = st.radio("",["여기서 다운로드하겠습니다.", "직접 엑셀파일을 업로드하겠습니다."])
    # 시장 (자동)
    market = listed_stock_list[listed_stock_list["Code"]==stock_code]["Market"].values[0]
    # 종목명 (자동)
    stock_name = listed_stock_list[listed_stock_list["Code"]==stock_code]["Name"].values[0]


    input_for_collecting = {       
        'stock_code' : stock_code,
        'stock_name' : stock_name,
        'way_to_crawling' : way_to_crawling,
        'market' : market,   
        }
    
    return input_for_collecting


#@
#st.divider()
#st.header("2. 데이터를 수집합니다.")
#col1, col2 = st.columns(2)
#with col1 :
#    input_for_collecting = start()




































# 화면2-1 : 데이터 수집 방식 선택
# 자료 listed_stock_list, df
# 변수 way_to_crawling,sotck_code, start_date, end_date, market, stock_name

#!
def sampling(input_for_collecting, display=True) :
    # 방식구분
    
    # 직접 업로드
    if input_for_collecting['way_to_crawling'] == "직접 엑셀파일을 업로드하겠습니다." :
        with col1 :
            uploaded_data = st.file_uploader("")
            if uploaded_data is not None :
                # 함수정의
                def handling_kiwoom(uploaded_data) : 
                    df=pd.read_excel(uploaded_data, engine='xlrd' )
                    kiwoom = df.copy().iloc[:,:6]                
                    kiwoom.columns = ['Date','Time','Open','High','Low','Close']
                    kiwoom["Date"] = pd.to_datetime(kiwoom["Date"], format="%Y-%m-%d %H:%M:%S")
                    kiwoom["Time"] = pd.to_datetime(kiwoom["Time"], format="%H:%M:%S")
                    DateTime = [i + (j - pd.Timestamp(j.date())) for i,j in zip(kiwoom["Date"], kiwoom["Time"])]
                    kiwoom["DateTime"] = DateTime
                    kiwoom.drop(["Date","Time",'Open','High','Low'],inplace=True, axis=1)
                    kiwoom["Change"] = np.log(kiwoom["Close"]) - np.log(kiwoom["Close"].shift(-1))
                    kiwoom.set_index("DateTime", inplace=True)
                    df = df.sort_index()
                    kiwoom.dropna(inplace=True)                
                    return kiwoom
                # 함수사용
                df = handling_kiwoom(uploaded_data)
                
                # 그 외 변수 
                input_for_collecting['interval'] = np.min([df.index[1] - df.index[0], df.index[2] - df.index[1]])
                input_for_collecting['start_date'] = df.index[0]
                input_for_collecting['end_date'] = df.index[-1]
                
                with col2 :
                    if display :
                        st.write("-업로드한 데이터입니다.")
                        st.write(df)

                st.write(input_for_collecting)
        











    # 패키지 이용 다운로드
    else :         
        with col1 :
            #시작 / 종료
            import datetime
            now = datetime.date.today()
            two_m_ago = now - datetime.timedelta(days=50)
            start_date, end_date = st.date_input("시작날짜-종료날짜", (two_m_ago,now))
            input_for_collecting['start_date'] = start_date
            input_for_collecting['end_date'] = end_date
            start_date = start_date.strftime('%Y-%m-%d')

            # 인터벌
            #select_interval = ['1분', '2분', '5분', '15분', '30분', '1시간', '1일']
            select_interval = ['2분', '1시간']    
            interval = st.selectbox("주가데이터 빈도", select_interval).replace('분',"m").replace('시간','h')    
            input_for_collecting['interval'] = interval

        with col2 : 
            # 화면 2-2 : 주가 데이터 조회
            # 함수정의
            def from_yf() :  
                
                #yfinace의 시장이름 포맷에 맞도록 수정
                input_for_collecting["market"] = input_for_collecting["market"].replace("KOSPI", "KS")
                input_for_collecting["market"] = input_for_collecting["market"].replace("KOSDAQ", "KQ")    
                # 주가조회
                ticker = yf.Ticker(f'{input_for_collecting["stock_code"]}.{input_for_collecting["market"]}')    
                import datetime
                df = ticker.history(
                    interval=input_for_collecting['interval'],
                    start=input_for_collecting["start_date"],
                    end=end_date)
                # 결측치 제거
                
                if df.isna().any().any() :
                    df.fillna(method="ffill")            
                # 변화율 포함
                df["Change"] = np.log(df["Close"]) - np.log(df["Close"].shift(-1))
                df = df.iloc[:-1]

                # 날짜 형식 변경
                df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
                df.index = pd.to_datetime(df.index,format="%Y-%m-%d %H:%M:%S")
                return df[["Close", "Change"]]
            # 함수사용
            df = from_yf()  
            if len(df)==0 : st.warning("조회된 데이터가 없습니다. 조회시작일자를 더 최근으로 조정하세요.")
            if display : st.write("* 조회한 데이터입니다.", df)

    return input_for_collecting, df


#@
#input_for_collecting, df = sampling(input_for_collecting)  


















































# 화면 3 : 투자전략 작성
# input 데이터프레임 작성


def investing_strategy(df, input_for_collecting, display=True) :
    if display :
        st.divider()
        st.header("3. 거래전략을 수립합니다.")
    c1,c2,c3 = st.columns(3)
    with c1:
        # 스플릿 갯수
        st.subheader("1)스플릿 갯수")    
        Controling_Number = st.number_input("최대 스플릿의 갯수 : ", 1,50, 7)
        
        # 회차별 자산배분
        select_balance = st.selectbox("회차별 자산비중", ["균등분배","1회차만 다르게" ,"직접입력"])
        st.caption(f"-현재 **{input_for_collecting['stock_name']}의 주가**는 **{df['Close'][0]:,.0f}**입니다.")
        if select_balance == "직접입력" :
            st.caption("아래에 회차별 자산배중 입력해주세요")
            balance = np.zeros(Controling_Number)
            co0, co1, co2 = st.columns(3)
            ind = 0
            for i in range(Controling_Number//3+1) :                
                with co0 : 
                    if 3*i+1 <= Controling_Number :
                        b = st.number_input(f"{3*i+1}회차 매수주식수",1, value=1)
                        balance[3*i] = b
                with co1 : 
                    if 3*i+2 <= Controling_Number :
                        b = st.number_input(f"{3*i+2}회차 매수주식수",1, value=1)
                        balance[3*i+1] = b
                with co2 : 
                    if 3*i+3 <= Controling_Number :
                        b = st.number_input(f"{3*i+3}회차 매수주식수",1, value=1)
                        balance[3*i+2] = b
                initial_balance = np.sum(balance) * df["Close"][-1]
            st.write(f"{initial_balance}로 투자금액을 설정합니다.")
            st.caption("(설정한 회차별 주식수 * 조회시작일 종가)")
        elif select_balance == "1회차만 다르게" :
            
            initial_balance = st.number_input("시작금액",min_value=10000, step=10000, value=1000000)            
            c_a = st.columns(2)        
            with c_a[0] :
                b1 = st.number_input("1회차 비중(%)", value=10)                                
            with c_a[1] :
                b2 = st.number_input("2회~ 비중(%)", value=(100-b1)) /100
                b1 /= 100

            balance = [initial_balance*b1//df['Close'][0]]   \
                    + [initial_balance*b2/Controling_Number//df['Close'][0]] * (Controling_Number-1)
            
            if balance[0] < 1 or balance[1] < 1 :
                st.warning("회차당 매수 주식 수가 0주 입니다. 시작금액을 높이거나 비중을 조절하세요")  
            else : 
                st.caption(f"-1회차에 {balance[0]}주를 매수합니다.")
                st.caption(f"-그 외 회차마다 {balance[1]}주를 매수합니다.")

        else : 
            st.caption("- 시작금액을 설정하면 회차별 매수주식수가 자동설정됩니다.")
            initial_balance = st.number_input("시작금액",min_value=10000, step=10000, value=1000000)  # 10억들고 시작 가정
            balance=[int(initial_balance // df["Close"][0] // Controling_Number)] * Controling_Number # 매 회차 동일수량 매수 가정
            if balance[0] < 1  :
                st.warning("회차당 매수 주식 수가 0주 입니다. 시작금액을 높이세요.")  
            else : 
                st.caption(f"- 회차마다 {balance[0]}주를 매매합니다.")   


















    
        
        #스플릿조건
    with c2:
        st.subheader("2)스플릿 조건")    
        select_split = st.selectbox("스플릿 조건부여방법", ["모든 회차 일괄적용", "점진적 변화", "1회차만 다르게"])
        
        if select_split == "모든 회차 일괄적용" :    
            st.divider()
            up = st.number_input("u% 오르면 익절", min_value=1.,max_value=500.,value=2.,step=0.1)
            down = st.number_input("d% 떨어지면 다음회차 매수", min_value=1.,max_value=500.,value=2.,step=0.1)
            u = [up / 100] * Controling_Number # 일정하게 5% 상승 시 익절 가정
            d = [down / 100] * Controling_Number # 일정하게 5% 하락 시 추매 가정

        elif select_split == "점진적 변화" :
            st.divider()
            up = st.number_input("1회차 u% 오르면 2회차 익절", min_value=1.,max_value=500.,value=2.,step=0.1) / 100
            up_step = st.number_input("회차마다 u% 변화분", value=0.5,step=0.1) / 100
            down = st.number_input("1회차 d% 떨어지면 3회차 매수", min_value=1.,max_value=500.,value=2.,step=0.1) / 100
            down_step = st.number_input("회차마다 d% 변화분", value=0.5,step=0.1) / 100
            u = [up + up_step*i for i in range(Controling_Number)]
            d = [down + down_step*i for i in range(Controling_Number)]

        else :
            st.divider()
            st.write("1회차")
            up_1th = st.number_input("1회차 u% 오르면 1회차 익절", min_value=1.,max_value=500.,value=2.,step=0.1) / 100
            down_1th = st.number_input("1회차 d% 떨어지면 2회차 매수", min_value=1.,max_value=500.,value=2.,step=0.1) / 100
            
            st.write("그 외 회차")
            select_split = st.selectbox("스플릿 조건부여방법 :", ["일괄적용", "점진적 변화"])
            
            if select_split == "일괄적용" :    
                up = st.number_input("u% 오르면 익절", min_value=1.,max_value=5.,value=2.,step=0.1)
                down = st.number_input("d% 떨어지면 다음회차 매수", min_value=1.,max_value=5.,value=2.,step=0.1)
                u = [up_1th] + [up / 100] * (Controling_Number-1) # 일정하게 5% 상승 시 익절 가정
                d = [down_1th] + [down / 100] * (Controling_Number-1) # 일정하게 5% 하락 시 추매 가정

            elif select_split == "점진적 변화" :
                up = st.number_input("2회차 u% 오르면 3회차 익절", min_value=1.,max_value=500.,value=2.,step=0.1) / 100
                up_step = st.number_input("회차마다 u% 변화분", value=0.5,step=0.1) / 100
                down = st.number_input("2회차 d% 떨어지면 3회차 매수", min_value=1.,max_value=5.,value=2.,step=0.1) / 100
                down_step = st.number_input("회차마다 d% 변화분", value=0.5,step=0.1) / 100
                u = [up_1th] + [up + up_step*i for i in range(Controling_Number-1)]
                d = [down_1th] + [down + down_step*i for i in range(Controling_Number-1)]


























    with c3 :
        # 거래비용
        st.subheader("3)거래비용")
        select_transaction_fee = st.toggle("거래비용포함", value=True)
        if select_transaction_fee :
            transaction_fee = st.number_input("증권사 거래수수료(%)", min_value=0.,max_value=1.,value=0.015,step=0.001) / 100
            tax = st.number_input("증권거래세(%)", min_value=0.,max_value=1.,value=0.05,step=0.01) / 100
            tax2 = st.number_input("농어촌특별세(%)", min_value=0.,max_value=1.,value=0.15,step=0.01) / 100
        else : 
            transaction_fee, tax, tax2 = 0, 0, 0

        # 1회차 수동매매
        st.subheader("4)1회차 수동매매")
        select_manual_trading = st.radio("", ["1회차매도 시 거래종료", "1회차도 스플릿"]) 
        if select_manual_trading == "1회차매도 시 거래종료" :
            st.caption("1회차가 매도될 시 더 이상 해당종목 거래하지 않습니다.")    

        elif select_manual_trading == "1회차도 스플릿"  :
            st.caption("1회차 매도가로부터 d% 되면은 1회차 재매수합니다.")    
        
        else :
            st.caption("백테스팅 조회기간 처음에 매수해서 끝에 매도합니다.")

    input_strategy = { 
    'Controling_Number' : Controling_Number,
    'select_balance' : select_balance,
    'select_split' : select_split,
    'select_transaction_fee' : select_transaction_fee,
    'select_manual_trading' : select_manual_trading,
    'balance' : balance,
    'initial_balance' : initial_balance,
    'u' : u,
    'd' : d,
    'tax' : tax,
    'tax2' : tax2,
    'transaction_fee' : transaction_fee
    }

    if display : 
        
        
        st.subheader("* 손님의 거래전략은 다음과 같습니다.")
        st.write(f"1. {input_for_collecting['stock_name']} ({input_for_collecting['stock_code']})에\
                {input_for_collecting['interval']}간격으로\
                {input_for_collecting['start_date']}부터 {input_for_collecting['end_date']} 기간동안 투자합니다.")
        st.write("2. 스플릿별 관리 전략은 아래표와 같습니다.")
        input_cl = [f"{i+1}th" for i in range(input_strategy['Controling_Number'])]
        input_idx = ["할당 주식 수", "u", "d"]

        # 전략표작성
        #def write_input_df()
        df_input = pd.DataFrame(columns=input_cl, index=input_idx)
        df_input.loc["할당 주식 수"] = input_strategy['balance']
        df_input.loc["u"] = input_strategy['u']
        df_input.loc["d"] = input_strategy['d']

        st.write(df_input)
        st.write("* 회차 별 할당 주식 수 = ( 시작금액 / 현재 주가 ) / 스플릿 갯수")

    return input_strategy

#@
#input_strategy = investing_strategy(df,input_for_collecting, display=False)

    




























































# 화면 4 (숨김) : 백테스팅
#!
def backtesting_settled(df, input_for_collecting, input_strategy) :
    # 필요칼럼 추가생산
    for i in ["buy","sell","position","sell","ith"] : df[i] = [np.NAN]*len(df)
    for i in range(input_strategy["Controling_Number"]+1):
        df[f"{i}th"] = [np.NAN]*len(df)
    

    # 첫날 수동거래
    df["1th"][0]=df["Close"][0] # 조회시점부터 구매가정.
    df["buy"][0]=df["Close"][0] # 구매이력
    df["position"][0] = -1 # 매수 = -1, 매도 = 1
    df["ith"][0] = 1. # 1회차


    # 반복문 준비
    indexer = list(df.columns).index("1th") # 1th 칼럼의 위치 
    #indexer_end = list(df.columns).index(f"{input_strategy["Controling_Number"]}th")
    controlling =  0 # 회차관리 (1회차만 매수상태는 0이다.)
    going1 = [np.nan] * input_strategy["Controling_Number"]
    going1[0] = df["Close"][0] # 회차별 매수가 관리
    going2 = df["Close"][0] # 가장 높은 회차 매수가

    for i in range(len(df)) :    

        if controlling == -1 :
            up = np.inf # 매도사인이 없도록 만든다.
            down = going2 * (1-input_strategy["d"][controlling])  
        else : 
            going2 = np.min(going1[:controlling+1])
            up = going2 * (1+input_strategy["u"][controlling])    
            down = going2 * (1-input_strategy["d"][controlling]) 
        

        #print(i, controlling, going2)

        # 매수
        if (controlling+1 < input_strategy["Controling_Number"] and df["Close"][i] <= down) :
            df["position"][i] = -1
            df.iat[i,indexer+controlling+1] = df["Close"][i]
            df["buy"][i] = df["Close"][i]
            df["ith"][i] = controlling+2        
            going1[controlling+1] = df["Close"][i]     
            controlling += 1  
            
        
        # 매도
        elif df["Close"][i] >= up :
            df["position"][i] = 1
            df.iat[i,indexer+controlling] = df["Close"][i]
            df["sell"][i] = df["Close"][i]
            df["ith"][i] = controlling+1
            going1[controlling] = np.nan   
            controlling -= 1
            if (input_strategy["select_manual_trading"] == "1회차도 스플릿" and controlling == -1) : going2 = df["Close"][i]
            if (input_strategy["select_manual_trading"] == "1회차매도 시 거래종료" and controlling == -1) : break
                
    df_settled = df.copy()
    agg =[]
    a = 1

    while controlling >= 0 :
        fron = pd.DataFrame(df.iloc[-1,:]).T
        fron["position"] = 1 # 매도
        fron[f"{controlling+1}th"] = fron["Close"]
        fron["sell"] = fron["Close"]
        fron["ith"] = controlling+1

        fron.index = [fron.index[0] + timedelta(seconds=a)]
        agg.append(fron)
        controlling -= 1
        a += 1

    df_settled = pd.concat([df_settled, *agg])
    return df, df_settled

#@
#df, df_settled = backtesting_settled(df, input_for_collecting, input_strategy)






























# 화면 5 : 투자이력 표시        
# 청산가정 _ history 채우기
#!
def settled_history(df_settled, input_strategy, display=True)    :
    # 변수꺼내기    
    balance = input_strategy["balance"]
    tax = input_strategy["tax"]
    tax2 = input_strategy["tax2"]
    transaction_fee = input_strategy["transaction_fee"]
    
    # 재정리
    fron = pd.melt(df_settled, "ith",("buy","sell"))
    fron = fron[fron["ith"].notna()]
    fron = fron[fron['value'].notna()]
    fron.sort_values('ith', inplace=True)

    # 작성
    df_history = pd.DataFrame()
    df_history["Buy At"] = fron[fron["variable"]=="buy"]["value"].values
    df_history["Sold At"] = fron[fron["variable"]=="sell"]["value"].values
    df_history["ith"] = fron[fron["variable"]=="buy"]["ith"].values
    df_history.index = df_settled.index[fron[fron["variable"]=="buy"].index]

    df_history["Amount"] = [balance[int(i)-1] for i in df_history["ith"].values]
    df_history['fee'] = (df_history["Buy At"]*transaction_fee + \
                        df_history["Sold At"]*(tax+tax2+transaction_fee)) * df_history["Amount"]
    df_history["Money Earned"] = (df_history["Sold At"] -df_history["Buy At"] )  *\
        df_history["Amount"] - df_history['fee']
    df_history["Money Earned(%)"] = (np.log(df_history["Sold At"] - df_history['fee']) \
                                    - np.log(df_history["Buy At"]))
    
    if display : 
        st.write(df_history)
        st.divider()    
        st.header("4. 투자이력을 표시합니다.")  
        st.write(f"* 투자지평의 마지막날 남은 보유분에서는 {input_for_collecting['interval']}간격으로 청산을 가정합니다.")


    return df_history
#@
#df_history = settled_history(df_settled, input_strategy)



























# 화면 5 - 2) : 참고화면 
#with st.expander("구체사항이 알고싶다면 열어보세요.", expanded=False):
#    st.write(df_settled[df_settled["ith"].notna()])
#    st.write("buy : 매수가격 / sell : 매도가격 / position : 포지션(-1:매수,1:매도)")
#    st.write("ith : 매매실행회차 / 각 ~th : 회차별 매매가격(주당)")



































# 화면 6 : 시각화합니다.

# plotly 그래프 그리기

import plotly
import plotly.graph_objects as go
# #!
def visualization(df_settled, input_for_collecting, display=True) :
    if display :
        st.divider()
        st.header("5. 시각화합니다.")
        stock_code = input_for_collecting['stock_code']

        df_settled_ = df_settled.copy()
        name = listed_stock_list[listed_stock_list["Code"]==stock_code]["Name"].values[0]
        layout = go.Layout(title=f'{name}({stock_code})')
        fig = go.Figure(layout=layout)
        fig.add_trace(
            go.Scatter(x=df_settled_.index, y=df_settled_['Close'], line = dict(color='white', width=0.7), name='Close'))
        fig.add_trace(
            go.Scatter(x=df_settled_.index, y=df_settled_['buy'], mode='markers', name='Buy', marker = dict(color='red', size=7)))
        fig.add_trace(
            go.Scatter(x=df_settled_.index, y=df_settled_['sell'], mode='markers', name='Sell', marker = dict(color='blue', size=7)))
        # x축 옵션 추가
        fig.update_xaxes(
            type='category')
        st.write(fig)
    
    return fig
#@
#fig = visualization(df_settled, input_for_collecting)







































# 화면 7 : 전략평가



#!
def valuation_strategy(df, df_history, input_for_collecting, input_strategy, display=True) :
    
    # 변수 꺼내기
    start_date = input_for_collecting['start_date']
    end_date = input_for_collecting['end_date']
    initial_balance = input_strategy['initial_balance']
    interval = input_for_collecting['interval']    
    tax = input_strategy["tax"]
    tax2 = input_strategy["tax2"]
    transaction_fee = input_strategy["transaction_fee"]
    balance = input_strategy["balance"]
    select_manual_trading = input_strategy['select_manual_trading']
    

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
    #from datetime import datetime
    #try :
    #    start_date = datetime.strtime(start_date_, "%Y-%m-%d")
    #    end_date = datetime.strptime(end_date_, "%Y-%m-%d")
    #except Exception as e :
    #    print(e)
    period = (end_date - start_date)
    seconds_per_year = 365.25 * 24 * 3600  # 1년의 초 수 (대략적으로 계산)
    period = period.total_seconds() / seconds_per_year
    CAGR = (np.log(final_balance)-np.log(initial_balance) +1) ** (1/period) -1  
    df_res["CAGR"] = CAGR

    # Strategy Return 계산
    strategy_return = np.log(final_balance) - np.log(initial_balance)
    df_res["Return(%)"] = f"{strategy_return*100:.3f}%"

    
    # Buy&Hold 전략 return 계산
    if select_manual_trading == "1회차매도 시 거래종료" :
        try :
            last_trade_date = df[df["sell"].notna()]['sell'].index[-1]
        except Exception as e :
            last_trade_date = df.index[-1]
    else :
        last_trade_date = df['sell'].index[-1]

    buy_hold_return = (np.log(df["Close"][last_trade_date] - df["Close"][last_trade_date]*(transaction_fee+tax+tax2))\
                        - np.log(df["Close"][0] + df["Close"][0]*transaction_fee))
    df_res["Buy&Hold return"] = f"{buy_hold_return*100:.3f}%"

    # 지수 return 계산
    #KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
    #KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
    df_res[f"Standard Deviation({interval})"] = f"{df['Change'].std()*100:.3f}%"

    if display : 
        st.divider()
        st.header("6. 전략을 평가합니다.")
        st.write(df_res)
        st.write("* 거래비용을 포함하고 있습니다.")
        st.write("* kospi, kosdaq, standard deviation (daily), MDD 구하는데 자료 빈도가 안 맞아서 향후 업데이트 예정")
        st.write(f"* 주당 가치가 계속 바뀌기 때문에 initial balance와 실제 매수금액은 차이가 있다. (매 회 {input_strategy['balance'][0]}주 매매)")

    return df_res
#@
#df_res = valuation_strategy(df, df_history, input_for_collecting, input_strategy)

# 표시






## 화면8 : 전략수정
#st.header("7. 전략을 수정합니다.")
#st.subheader(f"7-1. {stock_name} ({stock_code}) 최적 비율 추천")
#st.write("스플릿 조건을 제외한 다른 투자전략은 계승합니다.")
#
#
#ops_start = st.button("그리드탐색 시작") 
#if ops_start :
#        
#
#    grid = np.arange(1., 5.5, 0.5)
#
#    from itertools import product
#    # 가능한 조합을 생성할 숫자들을 정의합니다.
#    grid = np.arange(1., 5.5, 0.5)
#    # 조합의 길이를 지정합니다. 예를 들어, 길이가 3인 모든 조합을 생성하려면 3으로 설정합니다.
#    combination_length = input_strategy["Controling_Number"]
#    # 가능한 모든 조합을 생성합니다.
#    combinations = list(product(grid, repeat=combination_length))
#
#
#    suppose_u = combinations
#    suppose_d = combinations
#
#    # 화면2-1 : 데이터 수집 , 방식
#    # -1직접입력
#    # df = handling_kiwoom(uploaded_data)
#    # input_for_collecting
#
#    # -2 다운로드
#    # input_for_collecting
#    #df = from_yf(input_for_collecting)  
#
#    # 화면 3 : 투자전략
#    def sevensplit(df, input_strategy) :
#        
#
#        # 화면 4 (숨김) : 백테스팅
#        df, df_settled = backtesting_settled(df, input_strategy)
#        # 화면 5 : 투자이력 표시
#        df_history = settled_history(df_settled, input_strategy)
#        # 화면 6 : 시각화합니다.
#        fig = visualization(df_settled)    
#        # 화면 7 : 전략평가
#        df_res = valuation_strategy(df, df_history)
#
#        return df, df_settled, df_history, fig, df_res
#
#    return_agg = []
#    return_u = []
#    return_d = []
#
#    for i in suppose_u :
#        for j in suppose_d :
#            input_strategy["u"] = i 
#            input_strategy["d"] = j
#
#            try :
#                df, df_settled, df_history, fig, df_res = sevensplit(df, input_strategy)
#
#                return_agg.append(df_res["Return(%)"].values[0])
#                return_u += i
#                return_d += j
#
#            except Exception as  e :
#                pass    
#
#    df_ops = pd.DataFrame({'u' : return_u,
#                'd' : return_d,               
#                'return' : return_agg})
#
#    st.write(df_ops.sort_values('return', ascending=False ))
#
#
#
#
#
#



# 화면 1 : 세븐스플릿 소개

st.title("구성하신 포트폴리오를 평가합니다.")

            



st.divider()
st.header("1. 세븐스플릿을 소개합니다.")
c1, c2 = st.columns(2)
with c1 :
    st.write("세븐스플릿 전략의 백테스팅 페이지입니다.")
    st.caption("향후 업데이트 예정사항 입니다.")
    st.caption("- 개선사항3. 분봉자료 얻기")
    st.caption("- 개선사항4. 최적 퍼센트 추천")
    st.caption("- 개선사항5. 직접 업로드 기능")
    st.caption("- 개선사항8. 비청산 상황도 가정가능하도록")
    st.caption("- 개선사항10. 인풋데이터를 리스트나 딕트로 만들어놔야해 ")
    st.caption("- 개선사항11. 표에서 단위표시 화폐단위나 %형태로 ")
    st.caption("- 개선사항11. 1회차 매도시 %상승할 때 매수 할 수 있도록 조건부여 ")
with c2 :           
    st.write("조회가능한 종목의 목록은 다음과 같습니다.")
    # 상장사 목록조회
    cl_needed_at_list = ["Code", "Name", "Market", "Marcap"]
    listed_stock_list = pd.read_csv("listed_stock_list.csv")
    listed_stock_list = listed_stock_list.drop(listed_stock_list.columns[0],axis=1)
    st.write(listed_stock_list) 




#@
st.divider()
st.header("2. 데이터를 수집합니다.")
col1, col2 = st.columns(2)
with col1 :
    input_for_collecting = start()
#@
input_for_collecting, df = sampling(input_for_collecting)   
#@
input_strategy = investing_strategy(df,input_for_collecting, display=False)
#@
df, df_settled = backtesting_settled(df, input_for_collecting, input_strategy)
#@
df_history = settled_history(df_settled, input_strategy)
#@
fig = visualization(df_settled, input_for_collecting)
#@
df_res = valuation_strategy(df, df_history, input_for_collecting, input_strategy)