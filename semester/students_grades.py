# 자료 생성(각 학생들의 정보는 딕셔너리)
d1 = dict(번호 = '1', 학번 = '2018-1000', 이름 = 'Bob', 중간고사 = 76, 결석횟수 = 2)
d2 = dict(번호 = '2', 학번 = '2018-1001', 이름 = 'Chris', 중간고사 = 40, 결석횟수 = 5)
d3 = dict(번호 = '3', 학번 = '2018-1002', 이름 = 'Diana', 중간고사 = 95, 결석횟수 = 1)
d4 = dict(번호 = '4', 학번 = '2018-1003', 이름 = 'Irene', 중간고사 = 77, 결석횟수 = 0)
# 자료 생성(학생들의 목록은 리스트)
dd = [d1, d2, d3, d4]

#수정사항 반영 전 자료 출력(줄을 한 칸씩 띄워 주기 위해 반복문 사용)
def before_edit():
    print('===== 수정 전 학생 성적 정보 =====')
    for x in dd:
        print(x)
before_edit()

#for i in dd:
#    num = int(dd[i]['결석횟수'])
#      if num >= 5
#          del dd[i]
# 반복문을 써 보려고 했는데 'list indices must be integers or slices, not dict'라는 메세지와 함께 실패.

del(dd[1]) # 휴학생 삭제
# 기말고사 이후 수정사항 반영
d1.update({'기말고사' : 85})
d3.update({'기말고사' : 100, '중간고사' : 90})
d4.update({'기말고사' : 30, '결석횟수' : 1})

#수정사항 반영한 자료 출력
def after_edit():
    print('===== 수정 후 학생 성적 정보 =====')
    for i in dd:
        print(i)

after_edit()