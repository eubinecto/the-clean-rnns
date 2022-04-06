"""
use streamlit to deploy them
"""
from typing import Tuple
import streamlit as st
from cleanrnns.fetchers import fetch_pipeline_for_classification
from cleanrnns.pipelines import PipelineForClassification


@st.cache(allow_output_mutation=True)
def cache_pipeline() -> Tuple[PipelineForClassification, PipelineForClassification]:
    rnn = fetch_pipeline_for_classification("eubinecto", "rnn_for_classification")
    lstm = fetch_pipeline_for_classification("eubinecto", "lstm_for_classification")
    return rnn, lstm


def main():
    # fetch a pre-trained model
    rnn, lstm = cache_pipeline()
    st.title("The Clean Rnns - 긍 / 부정 감성분석")
    text = st.text_input("문장을 입력하세요", value="이 영화 너무 재미있어요! ㅎㅎ")
    if st.button(label="분석하기"):
        with st.spinner("Please wait..."):
            st.text("RNN: " + str(rnn(text)))
            st.text("LSTM: " + str(lstm(text)))


if __name__ == '__main__':
    main()
