
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import yfinance as yf
from streamlit_extras.switch_page_button import switch_page
import random
import matplotlib.pyplot as plt

# 상장사 목록조회
cl_needed_at_list = ["Code", "Name", "Market", "Marcap"]
listed_stock_list = pd.read_csv("listed_stock_list.csv")
listed_stock_list = listed_stock_list.drop(listed_stock_list.columns[0],axis=1)



    
    







# 화면 2 : 자료수집

#!
def start_portfolio() :
    cols = st.columns(2)
    way_to_crawling = st.radio("",["여기서 다운로드하겠습니다.", "직접 엑셀파일을 업로드하겠습니다."])
    cols = st.columns(3)
    if way_to_crawling == "여기서 다운로드하겠습니다." :        
        with cols[0]:
            st.subheader("1. 주가데이터 빈도")
            interval = st.selectbox("", ['2분', '1시간', '1일']).replace('분',"m").replace('시간','h').replace('일','d')
            st.caption("매매빈도가 높은 세븐스플릿 특성상 분단위 정보가 백테스팅에 적합하지만, 오픈소스에서 분단위 자료는 과거 50일만 조회가능한 점 양해부탁드려요")
            st.caption("주가데이처 출처 : 야후파이낸스")
        with cols[1] :
            #시작 / 종료
            st.subheader("2. 시작-종료날짜")
            
            import datetime        
            now = datetime.date.today()

            two_m_ago = now - datetime.timedelta(days=50)
            start_date, end_date = st.date_input("", (two_m_ago,now), min_value=two_m_ago if interval=="2m" else None)
        with cols[2] :
            st.subheader("3. 포트폴리오 포함 종목 데이터")
            stock_names = []
            stock_markets = []
            stock_codes = []
            stock_weights = []
            k = st.number_input("**몇 개 종목이 포함되나요?**", value=2, step=1, format="%d")
            initial_balance_portfolio = st.number_input("**시작금액은 얼마인가요?**", value=10000000, step=10000, format="%d")
            st.caption(f"{initial_balance_portfolio:,.0f}원으로 시작합니다.")
            
    else:
        st.write("업데이트 예정입니다.")    
    
    
    
    # 시장 (자동)
    firms = ['005930','086520','068270','005490','051910']
    #balance = np.zeros(k)
    co = st.columns(3)
    ind = 0
    for i in range(k//3+1):
        for j in range(len(co))   :  
            with co[j] : 
                c = st.columns(2)
                if 3*i+j+1 <= k :
                    with c[0] :
                        stock_code = st.text_input(f"{3*i+j+1}번 종목코드",value=firms[0], key=3*i+j+1)
                        try :
                            stock_name = listed_stock_list[listed_stock_list['Code']==stock_code]['Name'].values[0]
                            stock_market = listed_stock_list[listed_stock_list['Code']==stock_code]['Market'].values[0]
                            stock_codes.append(stock_code)
                            stock_names.append(stock_name)
                            stock_markets.append(stock_market)
                            st.caption(stock_name)
                        except Exception as e: 
                            st.warning("종목코드를 확인하세요. 조회할 수 없습니다.")
                    with c[1] :
                        if 3*i+j+1 ==  k :                        
                            with c[1] :
                                fixed_weight = 100- np.sum(stock_weights)*100
                                stock_weight =st.number_input(f"{3*i+j+1}번 비중(%)", min_value=fixed_weight, max_value=fixed_weight, value=fixed_weight, format='%g')
                                st.caption(stock_name)
                                stock_weights.append(stock_weight/100)                                
                                if fixed_weight < 0 : st.write("비중은 (-)가 될 수 없습니다.")
                        else :
                            with c[1] :
                                stock_weight = st.number_input(f"{3*i+j+1}번 비중(%)",min_value=0, value=int(100/k))
                                st.caption(stock_name)
                                stock_weights.append(stock_weight/100) 

                
        
    stock_initial_balance = [i*initial_balance_portfolio for i in stock_weights]


            
    input_for_collecting_portfolio ={        
        'stock_code' : stock_codes,
        'stock_name' : stock_names,
        'market' : stock_markets,   
        'stock_weight' : stock_weights,
        'way_to_crawling' : way_to_crawling,
        'interval' : interval,
        'start_date' : start_date,
        'end_date' : end_date,
        'initial_balance_portfolio' : initial_balance_portfolio,
        'stock_initial_balance' : stock_initial_balance
        }   

    return input_for_collecting_portfolio



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
def sampling_portfolio(input_for_collecting_portfolio, display=True) :
    
    dfs = []
    iter = len(input_for_collecting_portfolio["stock_code"])


    for i in range(iter) :

        input_for_collecting = {        
        'stock_code' : input_for_collecting_portfolio["stock_code"][i],
        'stock_name' : input_for_collecting_portfolio["stock_name"][i],
        'market' : input_for_collecting_portfolio["market"][i],   
        'stock_weight' : input_for_collecting_portfolio["stock_weight"][i],
        'way_to_crawling' : input_for_collecting_portfolio["way_to_crawling"],
        'interval' : input_for_collecting_portfolio['interval'],
        'start_date' : input_for_collecting_portfolio['start_date'],
        'end_date' : input_for_collecting_portfolio['end_date']
        }   

        #!
        def sampling(input_for_collecting, display=display) :
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
                    #now = datetime.date.today()
                    #two_m_ago = now - datetime.timedelta(days=50)
                    #start_date, end_date = st.date_input("시작날짜-종료날짜", (two_m_ago,now))
                    #input_for_collecting['start_date'] = start_date
                    #input_for_collecting['end_date'] = end_date
                    #start_date = start_date.strftime('%Y-%m-%d')

                    # 인터벌
                    #select_interval = ['1분', '2분', '5분', '15분', '30분', '1시간', '1일']
                    #select_interval = ['2분', '1시간']    
                    #interval = st.selectbox("주가데이터 빈도", select_interval).replace('분',"m").replace('시간','h')    
                    #input_for_collecting['interval'] = interval

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
                            end=input_for_collecting['end_date'])
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
        
        _, df = sampling(input_for_collecting)        
        
        dfs.append(df)
    
    return dfs


#@
#input_for_collecting, df = sampling(input_for_collecting)  


















































# 화면 3 : 투자전략 작성
# input 데이터프레임 작성

#!
def investing_strategy_portfolio(dfs, input_for_collecting_portfolio, display=True) :

    if display :
        st.divider()
        st.header("3. 거래전략을 수립합니다.")


    
    
    c1,c2,c3 = st.columns(3)
    
    balances = []
    initial_balances = []
    

    with c1:
        # 스플릿 갯수
        st.subheader("1)스플릿 갯수")    
        Controling_Number = st.number_input("최대 스플릿의 갯수 : ", 1,50, 7)
        
        # 회차별 자산배분
        
        balances = []
        stock_initial_balances = []

        select_balance = st.selectbox("회차별 자산비중", ["균등분배","1회차만 다르게"])
        #st.caption(f"-현재 **{input_for_collecting['stock_name']}의 주가**는 **{df['Close'][0]:,.0f}**입니다.")

                    
        # 1회차만 다르게
        if select_balance == "1회차만 다르게" :
            
            
            c_a = st.columns(2)        
            with c_a[0] :
                b1 = st.number_input("1회차 비중(%)", value=10)                                
            with c_a[1] :
                b2 = st.number_input("2회~ 비중(%)", value=(100-b1)) /100
                b1 /= 100

            for k in range(len(dfs)) :
                initial_balance = input_for_collecting_portfolio["stock_initial_balance"][k]
                df=dfs[k]

                balance = [initial_balance*b1//df['Close'][0]]   \
                        + [initial_balance*b2/Controling_Number//df['Close'][0]] * (Controling_Number-1)
                
                if (balance[0] < 1 or balance[1] < 1) :
                    st.warning(f"{input_for_collecting_portfolio['stock_name'][k]}의 회차당 매수 주식 수가 0주 입니다. 해당 종목에 대한 시작금액을 높이세요. 현재 할당된 금액은 {input_for_collecting_portfolio['stock_initial_balance'][k]:,.0f}원입니다.")
                
                balances.append(balance)
            
            #if balance[0] < 1 or balance[1] < 1 :
            #    st.warning("회차당 매수 주식 수가 0주 입니다. 시작금액을 높이거나 비중을 조절하세요")  
            #else : 
            #    st.caption(f"-1회차에 {balance[0]}주를 매수합니다.")
            #    st.caption(f"-그 외 회차마다 {balance[1]}주를 매수합니다.")

        # 균등분배
        else : 
            #st.caption("- 시작금액을 설정하면 회차별 매수주식수가 자동설정됩니다.")
            #initial_balance = st.number_input("시작금액",min_value=10000, step=10000, value=1000000)  # 10억들고 시작 가정
            
            for k in range(len(dfs)) :
                df = dfs[k]
                initial_balance = input_for_collecting_portfolio['stock_initial_balance'][k]
                balance=[int(initial_balance // df["Close"][0] // Controling_Number)] * Controling_Number # 매 회차 동일수량 매수 가정
                if balance[0] < 1  :
                    st.warning(f"{input_for_collecting_portfolio['stock_name'][k]}의 회차당 매수 주식 수가 0주 입니다. 해당 종목에 대한 시작금액을 높이세요. 현재 할당된 금액은 {input_for_collecting_portfolio['stock_initial_balance'][k]}원입니다.")
                    #else : 
                    #    st.caption(f"- 회차마다 {balance[0]}주를 매매합니다.")   
                balances.append(balance)
                

        
        stock_initial_balances=[np.sum(np.multiply(balances[i],  dfs[i]["Close"][0])) for i in range(len(balances))]
        






        



















    
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

    input_strategy_portfolio = { 
    'Controling_Number' : Controling_Number,
    'select_balance' : select_balance,
    'select_split' : select_split,
    'select_transaction_fee' : select_transaction_fee,
    'select_manual_trading' : select_manual_trading,
    'balance' : balances,
    'stock_initial_balance' : stock_initial_balances,
    'u' : u,
    'd' : d,
    'tax' : tax,
    'tax2' : tax2,
    'transaction_fee' : transaction_fee
    }

    input_for_collecting_portfolio['stock_initial_balance'] = input_strategy_portfolio['stock_initial_balance']

    #if display : 
    #            
    #    
    #    st.subheader("* 손님의 거래전략은 다음과 같습니다.")
    #    st.write(f"1. {input_for_collecting_portfolio['stock_name']} ({input_for_collecting_portfolio['stock_code']})에\
    #            {input_for_collecting_portfolio['interval']}간격으로\
    #            {input_for_collecting_portfolio['start_date']}부터 {input_for_collecting_portfolio['end_date']} 기간동안 투자합니다.")
    #    st.write("2. 스플릿별 관리 전략은 아래표와 같습니다.")
    #    input_cl = [f"{i+1}th" for i in range(input_strategy['Controling_Number'])]
    #    input_idx = ["할당 주식 수", "u", "d"]
#
    #    # 전략표작성
    #    #def write_input_df()
    #    df_input = pd.DataFrame(columns=input_cl, index=input_idx)
    #    df_input.loc["할당 주식 수"] = input_strategy['balance']
    #    df_input.loc["u"] = input_strategy['u']
    #    df_input.loc["d"] = input_strategy['d']
#
    #    st.write(df_input)
    #    st.write("* 회차 별 할당 주식 수 = ( 시작금액 / 현재 주가 ) / 스플릿 갯수")

    return input_strategy_portfolio



def display_strategy(input_strategy_portfolio, iter, display=True) :
    if display :
        st.subheader("* 손님의 거래전략은 다음과 같습니다.")
        st.write(f"1. {input_for_collecting_portfolio['stock_name'][iter]} ({input_for_collecting_portfolio['stock_code'][iter]})에\
                {input_for_collecting_portfolio['interval']}간격으로\
                {input_for_collecting_portfolio['start_date']}부터 {input_for_collecting_portfolio['end_date']} 기간동안 투자합니다.")
        st.write("2. 스플릿별 관리 전략은 아래표와 같습니다.")
        input_cl = [f"{i+1}th" for i in range(input_strategy_portfolio['Controling_Number'])]
        input_idx = ["할당 주식 수", "u", "d"]
        #
        # 전략표작성
        #def write_input_df()
        df_input = pd.DataFrame(columns=input_cl, index=input_idx)
        df_input.loc["할당 주식 수"] = input_strategy_portfolio['balance'][iter]
        df_input.loc["u"] = input_strategy_portfolio['u']
        df_input.loc["d"] = input_strategy_portfolio['d']
        #
        st.write(df_input)
        st.write("* 회차 별 할당 주식 수 = ( 시작금액 / 현재 주가 ) / 스플릿 갯수")








#return input_strategy_portfolio
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
def settled_history(df_settled, input_for_collecting, input_strategy, display=True)    :
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
    df_history["Money Earned"] = (df_history["Sold At"] -df_history["Buy At"] )*df_history["Amount"]\
          - df_history['fee']
    df_history["Money Earned(%)"] = (np.log(df_history["Money Earned"]) \
                                    - np.log(df_history["Buy At"]))
    
    if display : 
        st.divider()    
        st.header("4. 투자이력을 표시합니다.")  
        st.write(f"* 투자지평의 마지막날 남은 보유분에서는 {input_for_collecting['interval']}간격으로 청산을 가정합니다.")
        st.write(f"* 매매 주식수가 0이더라도 매매가능했던 시점은 표시합니다.")
        st.write(df_history)

        with st.expander("구체사항이 알고싶다면 열어보세요.", expanded=False):
            st.write(df_settled[df_settled["ith"].notna()])
            st.write("buy : 매수가격 / sell : 매도가격 / position : 포지션(-1:매수,1:매도)")
            st.write("ith : 매매실행회차 / 각 ~th : 회차별 매매가격(주당)")
            


    return df_history
#@
#df_history = settled_history(df_settled, input_strategy)



























# 화면 5 - 2) : 참고화면 







































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
            go.Scatter(x=df_settled_.index, y=df_settled_['Close'], line = dict(color='yellow', width=0.7), name='Close'))
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
































def num_db(num) :
    return f"{num*100:,.3f}%"







# 화면 7 : 전략평가



#! 개별종목.
def valuation_strategy(df, df_history, input_for_collecting, input_strategy, display=True) :
    
    # 변수 꺼내기
    start_date = input_for_collecting['start_date']
    end_date = input_for_collecting['end_date']
    initial_balance = input_strategy['stock_initial_balance']
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


    
    #  거래 마지막 날 설정 / 경과기간 계산
    if select_manual_trading == "1회차매도 시 거래종료" :
        try :
            last_trade_date = df[df["sell"].notna()]['sell'].index[-1]
        except Exception as e :
            last_trade_date = df.index[-1]
    else :
        last_trade_date = df['sell'].index[-1]

    last_trade_date = last_trade_date.date()
    period = (last_trade_date - start_date)
    seconds_per_year = 365.25 * 24 * 60 * 60  # 1년의 초 수 (대략적으로 계산)
    period = period.total_seconds() / seconds_per_year # 경과기간

    # Strategy Return 계산
    hpr_strategy = np.log(final_balance) - np.log(initial_balance)
    df_res["Return(%)"] = f"{hpr_strategy*100:.3f}"


    # Annualised Strategy Return 계산
    annual_return_strategy = hpr_strategy * (1/period)
    df_res["Annual Return(%)"] = f"{annual_return_strategy*100:.3f}"

    # CAGR (기하수익률)
    CAGR = (np.log(final_balance)-np.log(initial_balance) +1) ** (1/period) -1  
    df_res["CAGR"] = num_db(CAGR)

    
    # Buy&Hold 전략 return 계산
    # 마지막날 설정
    if select_manual_trading == "1회차매도 시 거래종료" :
        try :
            last_trade_date = df[df["sell"].notna()]['sell'].index[-1]
        except Exception as e :
            last_trade_date = df.index[-1]
    else :
        last_trade_date = df['sell'].index[-1]

    # hpr_buy_hold
    hpr_buy_hold = (np.log(df["Close"][last_trade_date] - df["Close"][last_trade_date]*(transaction_fee+tax+tax2))\
                        - np.log(df["Close"][0] + df["Close"][0]*transaction_fee))
    df_res["Buy&Hold return"] = f"{hpr_buy_hold*100:.3f}%"

    # annulalized hpr_buy_hold
    annual_return_buy_hold = hpr_buy_hold * (1/period)




    # 지수 return 계산
    #KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
    #KOSPI200_return = (np.log(df["Close"][-1]) - np.log(df["Close"][0]))
    std = df['Change'].std()
    df_res[f"Standard Deviation({interval})"] = f"{std*100:.3f}%"


    outputs_stock = {
        'start_date' : start_date,
        'last_trade_date' : last_trade_date,
        'trading_period' : period,
        'final_balance' : final_balance,
        'hpr_strategy' : hpr_strategy,
        'CAGR' : CAGR,
        'hpr_buy_hold' : hpr_buy_hold,
        'standard_deviation' : std,
        'annual_return_strategy' : annual_return_strategy,
        'annual_return_buy_hold' : annual_return_buy_hold

    }


    if display : 
        st.divider()
        st.header("6. 전략을 평가합니다.")
        st.write(df_res)
        st.write("* 거래비용을 포함하고 있습니다.")
        st.write("* kospi, kosdaq, standard deviation (daily), MDD 구하는데 자료 빈도가 안 맞아서 향후 업데이트 예정")

    return df_res, outputs_stock
#@
#df_res = valuation_strategy(df, df_history, input_for_collecting, input_strategy)



#! 포트폴리오 input_for_collecting_portfolio, input_strategy_portfolio, iter) 을 종목 하나껄로 
def portfolio_to_stock(input_for_collecting_portfolio, input_strategy_portfolio, iter) :
    
    input_for_collecting = {        
        'stock_code' : input_for_collecting_portfolio["stock_code"][iter],
        'stock_name' : input_for_collecting_portfolio["stock_name"][iter],
        'market' : input_for_collecting_portfolio["market"][iter],   
        'stock_weight' : input_for_collecting_portfolio["stock_weight"][iter],
        'way_to_crawling' : input_for_collecting_portfolio["way_to_crawling"],
        'interval' : input_for_collecting_portfolio['interval'],
        'start_date' : input_for_collecting_portfolio['start_date'],
        'end_date' : input_for_collecting_portfolio['end_date']
        }
    
    input_strategy = { 
    'Controling_Number' : input_strategy_portfolio['Controling_Number'],
    'select_balance' : input_strategy_portfolio['select_balance'],
    'select_split' : input_strategy_portfolio['select_split'],
    'select_transaction_fee' : input_strategy_portfolio['select_transaction_fee'],
    'select_manual_trading' : input_strategy_portfolio['select_manual_trading'],
    'balance' : input_strategy_portfolio['balance'][iter],
    'stock_initial_balance' : input_strategy_portfolio['stock_initial_balance'][iter],
    'u' : input_strategy_portfolio['u'],
    'd' : input_strategy_portfolio['d'],
    'tax' : input_strategy_portfolio['tax'],
    'tax2' : input_strategy_portfolio['tax2'],
    'transaction_fee' : input_strategy_portfolio['transaction_fee']
    }

    
    return input_for_collecting, input_strategy




# 종목별 결과물을 가지고 portfolio결과물 만들기

def valuation_portfolio(input_for_collecting_portfolio, input_strategy_portfolio, dfs, output_stock,display=True) :
    hpr_portfolio = np.sum([i*j for i,j in zip(output_stock['returns'], input_for_collecting_portfolio['stock_weight'])])
        
    # 연수익률
    annual_return_portfolio = np.sum([i*j for i,j in zip(output_stock['annual_return_strategy'], input_for_collecting_portfolio['stock_weight'])])
    
    # 포트폴리오 분산
    agg = pd.DataFrame( [dfs[i]["Change"].values for i in range(len(dfs))], 
                    index= input_for_collecting_portfolio['stock_name']).T
    cov_matrix = agg.cov()
    weight_vector = input_for_collecting_portfolio['stock_weight']
    portfolio_std = np.sqrt(np.dot(np.dot(weight_vector,cov_matrix),weight_vector))
    #st.write('agg',agg) 
    
    # CAGR
    CAGR_portfolio = (1+hpr_portfolio) ** (1/output_stock['trading_period']) -1
    
    #hpr_buy_hold hpr
    hpr_buy_hold = np.sum([i*j for i,j in zip(output_stock['buy_hold_returns'], input_for_collecting_portfolio['stock_weight'])])
    
    #buy_hold_ annual return
    annual_return_buy_hold = np.sum([i*j for i,j in zip(output_stock['annual_return_buy_hold'], input_for_collecting_portfolio['stock_weight'])])
    
    #hpr_buy_hold CAGR
    CAGR_buy_hold = (1+hpr_buy_hold) ** (1/(output_stock['trading_period'])) -1
    
    if display :
        st.write("**- trading period** :" f"{output_stock['trading_period']:.3f}year ({output_stock['start_date']}~{output_stock['last_trade_date']})")
        st.write("**- (%) hpr_portfolio** :", num_db(hpr_portfolio))
        st.write("**- (%) annual_return_portfolio** : ", num_db(annual_return_portfolio))
        st.write('**- (%) portfolio_std** : ', num_db(portfolio_std))
        st.write('**- (%) CAGR_portfolio** : ', num_db(CAGR_portfolio))
        st.write('**- (%) hpr_buy_hold** : ', num_db(hpr_buy_hold))
        st.write('**- (%) annual_return_buy_hold** : ', num_db(annual_return_buy_hold))
        st.write('**- (%) CAGR_buy_hold** :', num_db(CAGR_buy_hold))

    #returns.append(annual_return_portfolio
    #stds.append(portfolio_std
    #final_balances.append(np.sum(final_balances)
    #buy_hold_returns.append(hpr_buy_hold)
    #hpr.append(hpr_portfolio

    outputs_portfolio = {
        'start_date' : output_stock['start_date'],
        'last_trade_date' : output_stock['last_trade_date'],
        'trading_period' : output_stock['trading_period'],
        'hpr_portfolio' : hpr_portfolio,
        'annual_return_portfolio' : annual_return_portfolio,
        'portfolio_std' : portfolio_std,
        'CAGR_portfolio' : CAGR_portfolio,
        'hpr_buy_hold' : hpr_buy_hold,
        'annual_return_buy_hold' : annual_return_buy_hold,
        'CAGR_buy_hold' : CAGR_buy_hold
    }

    return outputs_portfolio


















#! 랩핑
def sevensplit_portfolio(input_for_collecting_portfolio, input_strategy_portfolio, dfs, display=True) :
    st.divider()
    st.subheader("아래에서 백테스팅 결과를 확인하세요. 탭을 선택하세요")
    st.divider()
    # 레이아웃  
    tabs = st.tabs (['포트폴리오']+input_for_collecting_portfolio["stock_name"]) 

    # 값 모으기
    returns = []
    annual_returns = []
    stds = []
    final_balances = []
    buy_hold_returns = []
    hpr = []
    annual_buy_hold_returns = []


    for i in range(len(input_for_collecting_portfolio["stock_name"])) :
        with tabs[i+1] :        
            display_strategy(input_strategy_portfolio, iter=i, display=display)
            input_for_collecting, input_strategy = \
                portfolio_to_stock(input_for_collecting_portfolio, input_strategy_portfolio, i) 
            df = dfs[i]
            #@
            df, df_settled = backtesting_settled(df, input_for_collecting, input_strategy)
            #@
            df_history = settled_history(df_settled, input_for_collecting,input_strategy,display=display)
            #@
            fig = visualization(df_settled, input_for_collecting, display=display)

            df_res, outputs_stock =  valuation_strategy(df, df_history, input_for_collecting, input_strategy, display=display)



            returns.append(outputs_stock["hpr_strategy"])
            stds.append(outputs_stock['standard_deviation'])
            final_balances.append(outputs_stock['final_balance'])
            buy_hold_returns.append(outputs_stock['hpr_buy_hold'])
            hpr.append(outputs_stock["hpr_strategy"])
            annual_returns.append(outputs_stock['annual_return_strategy'])
            annual_buy_hold_returns.append(outputs_stock['annual_return_buy_hold'])

    output_stock = {
        'start_date' : outputs_stock['start_date'],
        'last_trade_date' : outputs_stock['last_trade_date'],
        'trading_period' : outputs_stock['trading_period'],
        'returns' : returns,
        'stds': stds,
        'final_balances' : final_balances,
        'buy_hold_returns' : buy_hold_returns,
        'hpr' : hpr,
        'annual_return_strategy' : annual_returns,
        'annual_return_buy_hold' : annual_buy_hold_returns
    }



    with tabs[0] :
        outputs_portfolio = valuation_portfolio(input_for_collecting_portfolio, input_strategy_portfolio, dfs, output_stock,display=display)
    

    return output_stock, outputs_portfolio





    
    














# 화면 1 : 세븐스플릿 소개

st.title("구성하신 포트폴리오를 평가합니다.")
st.divider()
st.header("1. 세븐스플릿을 소개합니다.")
c1, c2 = st.columns(2)
with c1 :
    st.write("세븐스플릿 전략의 백테스팅 페이지입니다.")
    st.write("세븐스플릿의 간략묘사는, 매매 횟수를 n회로 분할하여 시간차를 두고 매매를 진행하는 겁니다.")
    st.write("앞 전의 거래로 부터 u% 주가가 상승할 경우 익절, 또는 d% 하락할 경우 추가매매를 진행합니다.")
    st.write("구불구불한 그래프의 모양을 이용하는 전략인 것입니다.")
    st.write("조악한 저의 도구가 전략을 이해하는 정도라도 도움 되었으면 좋겠습니다..")
    st.write("언제든지 편안하게 피드백 주세요..")
    st.write("성투하세요~")

    
with c2 :           
    st.subheader("조회가능한 종목 목록")
    ## 상장사 목록조회
    #cl_needed_at_list = ["Code", "Name", "Market", "Marcap"]
    #listed_stock_list = pd.read_csv("listed_stock_list.csv")
    #listed_stock_list = listed_stock_list.drop(listed_stock_list.columns[0],axis=1)
    st.write(listed_stock_list) 




#! 파트 1 : 사용자 직접 설정이 필요한 부분  
#@
st.divider()
st.header("2. 데이터를 수집합니다.")
#@
input_for_collecting_portfolio = start_portfolio()
#@
col1, col2 = st.columns(2)
dfs = sampling_portfolio(input_for_collecting_portfolio, display=False)   
#@
input_strategy_portfolio = investing_strategy_portfolio(dfs,input_for_collecting_portfolio)



#! 파트 2 :동작하기만 하면 되는 부분
output_stock, outputs_portfolio = sevensplit_portfolio(input_for_collecting_portfolio, input_strategy_portfolio, dfs)







#! 파트 3: 심화파트
st.divider()
st.header('추가시각화 : 업데이트 중입니다.')
iter = st.number_input("포트폴리오 갯수", value=500)
if st.button("효율적 경계 보기"):



    aggregated_return = []
    aggregated_std = []
    
    with st.empty():
        for complete in range(iter) :
            # 가중치 조정
            st.write(f"⏳ {complete/iter*100:.2f}% is completed")
            weights = np.random.rand(len(input_for_collecting_portfolio['stock_code']))
            wegihts = weights / np.sum(weights)
            input_for_collecting_portfolio["stock_weight"] = weights

            outputs_portfolio = \
                valuation_portfolio(input_for_collecting_portfolio, input_strategy_portfolio, dfs,output_stock, display=False)

            aggregated_return.append(outputs_portfolio['annual_return_portfolio'])
            aggregated_std.append(outputs_portfolio['portfolio_std'])

        fig = plt.figure()
        plt.scatter(y=aggregated_return, x=aggregated_std, alpha=0.5)
        plt.xlabel = ['표준편차']
        plt.ylabel = ['수익률']
        st.write(fig)

    st.write(aggregated_return[-1])
    st.write(aggregated_std[-1])
    st.write(input_for_collecting_portfolio['stock_weight'])
























































































    # IRR_portfolio
    #traded_dates = [dfs[i][dfs[i]["buy"].notna and dfs[i]['sell'].notna()].index for i in range(len(dfs))]
    #st.write(traded_dates)
    #df_history



    # test
    #for _ in range(300) :
    #    weights = np.random.rand(len(input_for_collecting_portfolio['stock_code']))
    #    wegihts = weights / np.sum(weights)
    #    input_for_collecting_portfolio["stock_weight"] = weights
#





    #st.write("세븐스플릿의 경우 중간기간에 현금유출이 없으니 재투자 수익을 무시하시면 안됩니다.")
    #st.write("세븐스플릿의 경우 거래횟수가 빈번하니 거래비용을 무시하시면 안됩니다.")
    #st.write("세븐스플릿의 경우 하락장에서는 자동매매가 이루어지지만 무자비한 상승장에서는 어떻게 운용할 지 고려가 필요합니다.")



    





    
    
    


    #outputs = {
    #    'final_balance' : final_balance,
    #    'hpr_strategy' : hpr_strategy,
    #    'CAGR' : CAGR,
    #    'hpr_buy_hold' : hpr_buy_hold,
    #    'standard_deviation' : std
    #}











    #portfolio_return = np.sum(returns) 

    # 바이앤홀드 리턴 HPR

    # 지수 리턴

    # 포트폴리오 변동성

    # NPA, IRR

    # 비중은 얼마가 최적이였을까 : CAPM 모형 시각화

    # Sharpe Ratio

    # CAGR


