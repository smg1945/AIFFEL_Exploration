# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이하영님
- 리뷰어 : 김용석님

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  ▶ 기본 노드에 충실하게 작성되어 코드는 문제없이 잘 동작하였습니다. 

- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
    ▶ 구두로 설명을 해주셨는데 내용에 대해 이해를 잘 하고 계셔서 많은 도움이 되었습니다. 

- [o] 3.코드가 에러를 유발할 가능성이 있나요?
    ▶ 에러 유발할 부분은 없는 것 같습니다. 

- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
    ▶ 네, 완벽하게 이해하고 있었습니다.

- [ ] 5.코드가 간결한가요?
    ▶ 클래스 형태로 불록화가 잘 진행되어 매우 이해하기 편했습니다. 


# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
▷ 하영님께서 공부하시면서 중요하다고 생각하신 부분에 대한 자세한 주석이 있는 코드를 올려놓았습니다. 

▶ 한글로 데이터 전처리 작업 부분
  def preprocess_sentence(sentence):
  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 예를 들어서 "I am a student." => "I am a student ."와 같이
  # student와 온점 사이에 거리를 만듭니다.
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  # (a-z, A-Z, ㄱ-ㅎ, ㅏ-ㅣ, 가-힣, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
  sentence = re.sub(r"[^a-zA-Zㄱ-ㅣ가-힣0-9?.!,]+", " ", sentence)
  sentence = sentence.strip()

# 불용어 제거
  sentence_list = []
  for s in sentence.split(' '):
    if s not in stopwords:
      sentence_list.append(s)

  sentence = " ".join([str(ele) for ele in sentence_list])
  return sentence

▶ # 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
 def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 이하인 경우에만 데이터셋으로 허용
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # 최대 길이로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs


# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
