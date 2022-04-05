"""
nsmc 데이터셋 구경해보기
https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/nsmc.html
"""
from Korpora import Korpora
from Korpora import NSMCKorpus


def main():
    Korpora.fetch("nsmc")  # 디렉토리를 명시하지 않을 경우, 그냥 ~/Korpora 경로에 다운로드 된다고 함.
    corpus = NSMCKorpus()
    # 훈련, 테스트만 존재, 검증 데이터는 없음.
    # 향후에 훈련 셋을 검증데이터로 나누는 작업이 필요할 것.
    print(len(corpus.train))
    print(len(corpus.test))

    for example in corpus.train:
        print(example.text)
        print(example.label)

    print(type(corpus.train))
    print(type(corpus.train[0]))


if __name__ == '__main__':
    main()
