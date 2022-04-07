"""
use streamlit to deploy them
"""
import pandas as pd
import streamlit as st
from typing import Tuple
from cleanrnns.fetchers import fetch_pipeline_for_classification
from cleanrnns.pipelines import PipelineForClassification


@st.cache(allow_output_mutation=True)
def cache_pipeline() -> Tuple[PipelineForClassification,
                              PipelineForClassification,
                              PipelineForClassification]:
    rnn = fetch_pipeline_for_classification("eubinecto", "rnn_for_classification")
    lstm = fetch_pipeline_for_classification("eubinecto", "lstm_for_classification")
    bilstm = fetch_pipeline_for_classification("eubinecto", "bilstm_for_classification")
    return rnn, lstm, bilstm


def main():
    # fetch a pre-trained model
    rnn, lstm, bilstm = cache_pipeline()
    st.title("The Clean Rnns - 긍 / 부정 감성분석")
    text = st.text_input("문장을 입력하세요", value="제목은 시선을 끌지만 줄거리가 애매모호하다")
    if st.button(label="분석하기"):
        with st.spinner("로딩중..."):
            # prediction with RNN
            table = list()
            pred, probs = rnn(text)
            sentiment = "🟢(긍정)" if pred else "🔴(부정)"
            probs = ["{:.4f}".format(prob) for prob in probs]
            table.append(["RNN", sentiment, str(probs)])
            # prediction with LSTM
            pred, probs = lstm(text)
            probs = ["{:.4f}".format(prob) for prob in probs]
            sentiment = "🟢(긍정)" if pred else "🔴(부정)"
            table.append(["LSTM", sentiment, str(probs)])
            pred, probs = bilstm(text)
            sentiment = "🟢(긍정)" if pred else "🔴(부정)"
            probs = ["{:.4f}".format(prob) for prob in probs]
            table.append(["RNN", sentiment, str(probs)])
            df = pd.DataFrame(table, columns=["모델", "예측", "확률분포 [부정, 긍정]"])
            st.markdown(df.to_markdown(index=False))


if __name__ == '__main__':
    main()
