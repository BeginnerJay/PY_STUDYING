A = input("직각을 낀 첫 번째 변의 길이를 입력하세요 : ") # input 함수로 입력받으면 자료형이 문자열(string)이다.
a = int(A) # int()로 자료형을 정수형으로 형변환한다.
B = input("직각을 낀 두 번째 변의 길이를 입력하세요 : ") # input 함수로 입력받으면 자료형이 문자열(string)이다.
b = int(B) # int()로 자료형을 정수형으로 형변환한다.
Result = '넓이는 {area}, 빗변의 길이는 {hypotenuse}입니다.'.format(area=a * b * 0.5, hypotenuse=(a ** 2 + b ** 2) ** 0.5)
# format() 메소드를 이용해 결과값을 출력한다. 변수명을 넣어 주는 것을 선호하여 넣어 주었다.
print(Result) # 입력된 값들을 이용해 계산된 직각삼각형의 넓이와 빗변의 길이를 출력한다.
