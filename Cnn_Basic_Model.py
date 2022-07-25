# MLP의 경우 Hidden layer의 fully connected 방식만 사용되었다.
# CNN의 경우 Convolutional layer와 pooling layer를 추가로 구현해야 한다.
# 따라서 계층 추가를 쉽게 반영하는 프로그램의 툴이 필요하다.
#
# 새로운 계층을 지원하려면 계층 특성에 따른 파라미터 생성, 순전파, 역전파 처리 기능이 필요하다.
# 이들 세 기능을 수행할 메서드를 정의해주는 것만으로 새로운 계층 추가가 가능하도록 모델 프로그램을 확장

"""
method 이름을 이용한 hooking 기법


"""