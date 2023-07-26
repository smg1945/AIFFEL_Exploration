# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이하영님
- 리뷰어 : 최원석님

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

[X] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  ▶ 마지막 4개 모델의 accuracy와 loss 그래프에서 유의미한 모습은 보이지 않았습니다. 이를 해결하기 위해 각종 값들을 변경했지만, 유의미한 모습을 보이진 못했다고 합니다. Epoch를 15로 설정하셨는데, 혹시라도 Epoch를 늘리면 어떻게 될지에 대해서 여쭤보았습니다. 하지만 오늘 Epoch를 50 이상으로도 늘려서 진행해보았지만 원하는 모습이 나오진 않았다고 합니다. 메모리때문에 batch나 레이어 크기를 줄였다고 하시는데, 이것으로 인해 적절한 학습이 되지 않았나 싶습니다.
![download](https://github.com/smg1945/AIFFEL_Exploration/assets/126739777/87c86310-92b5-4bda-890a-ce72c9bca155)

[O] 2.주석을 보고 작성자의 코드가 이해되었나요?
    ▶ 넵! 추가적으로 구두로 설명도 자세하게 해주셔서 이해가 잘 되었습니다.

[X] 3.코드가 에러를 유발할 가능성이 있나요?
    ▶ 에러 유발할 부분은 없는 것 같습니다.
  하지만 아래처럼 중간중간 코드가 실행되면서 경고문이 나왔는데, 'lr'은 이제는 사용되지 않고, 대신 'learning_rate'로 대체되어야 한다고 나왔습니다. 에러는 아니지만 경고가 나와서 같이 언급했습니다.
  
코드
```
    resnet_50.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.),
        metrics=['accuracy'],
    )
```
경고문
```
c:\Users\ZAKAR\anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:111: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super().__init__(name, **kwargs)
```


- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
    ▶ 네, 완벽하게 이해하고 있었습니다.

- [O] 5.코드가 간결한가요?
    ▶ 넵 간결하게 되어 있습니다. 블록화가 잘 되어있어서 이해하기 편했고, 정리를 잘 하셔서 수월하게 코드를 봤습니다.
