from django.shortcuts import render, redirect, reverse

#http 출력위함
from django.http import HttpResponse, HttpResponseRedirect

from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout

#form(회원등록폼,) 불러올때
from .forms import LoginForm, JoinUserForm

# message불러올때
from django.contrib import messages

# decorator 불러들일때
from django.contrib.auth.decorators import login_required
# from main.decorators import user_is_entry_author

# Create your views here.

def index(request):

    if request.method == "POST":

        projectname = request.POST['projectname']

        return render(request,'home/index.html', {'projectname':projectname})
        # return HttpResponse("Welcome Home")
        # return render(request, 'main/index.html')

    else:

        return render(request, 'home/index.html')


## ajax를 이용한 로그인
def login(request):

    if request.method == "POST":

        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None :

            auth_login(request, user)

            # session 설정하여 나중에 활용
            m = User.objects.get(username=request.POST['username'])

            request.session['member_id'] = m.id
            request.session['member_username'] = m.username

            return HttpResponse("Welcome My Python Tutorial")   # ajax 리턴의 경우 String으로 리턴

        else:
            return HttpResponse('로그인 실퍠.  다시 시도하세요.')

    else:
        form = LoginForm()

        return render(request, 'home/login.html', {'form': form})


def logout(request):

    auth_logout(request)

    return render(request, 'home/index.html')


def join(request):

    if request.method == 'POST':

        if request.POST['password1'] == request.POST['password2']:

            user = User.objects.create_user(
                username=request.POST['username'],
                password=request.POST['password1'],
                email=request.POST['email'],)

            auth_login(request, user)

            return redirect('/')

        return render(request, 'home/join.html')

    return render(request, 'home/join.html')





# ajax를 이용한
# def join(request):
#     if request.method == "POST":
#
#         # create a form instance and populate it with data from the request:
#         print("test1")
#
#         form = JoinUserForm(request.POST)
#         print(form)
#
#         if form.is_valid():
#
#             if request.POST['password1'] == request.POST['password2']:
#                 form.save()
#             new_user = User.objects.create_user(**form.cleaned_data)  # User에 저장(create_user())
#
#             # 옵션 저장 다른 표현
#             # new_user=form.save(commit=False)  # 저장 객체 생성, 장장 실행하지 않음
#             # new_user.email=request.User.get_email()  # 추가적 User의 email을 이곳 이메일로 지정
#
#
#             # new_user.generate() # 새로운 User db 저장
#
#             print("test2")
#
#             login(request, new_user)  # login 함수 실행
#
#             return HttpResponse("Welcome My World")  #return redirect('index') #url의 name을 경로대신 입력한다.(여기는 ajax 때문에 문자열 리턴)
#     else:
#         form = JoinUserForm()  #forms.py의 JoinUserForm 클래스의 인스턴스
#         return render(request, 'base/join.html', {'form': form})   # 템플릿 파일 경로 지정, 데이터 전달


# ajax를 이용한
# def join(request):
#     if request.method == "POST":
#
#         # create a form instance and populate it with data from the request:
#         form = JoinUserForm(request.POST)
#         print(form)
#
#         if form.is_valid():
#
#             form.save()
#             new_user = User.objects.create_user(**form.cleaned_data)  # User에 저장(create_user())
#
#             # 옵션 저장 다른 표현
#             # new_user=form.save(commit=False)  # 저장 객체 생성, 장장 실행하지 않음
#             # new_user.email=request.User.get_email()  # 추가적 User의 email을 이곳 이메일로 지정
#             # new_user.generate() # 새로운 User db 저장
#
#             login(request, new_user)  # login 함수 실행
#
#             return HttpResponse("Welcome My World")  #return redirect('index') #url의 name을 경로대신 입력한다.(여기는 ajax 때문에 문자열 리턴)
#
#     else:
#         form = JoinUserForm()  #forms.py의 JoinUserForm 클래스의 인스턴스
#         return render(request, 'main/join.html', {'form': form})   # 템플릿 파일 경로 지정, 데이터 전달

# ajax를 이용한 회원가입
# def join(request):
#
#     if request.method == "POST":
#
#         # create a form instance and populate it with data from the request:
#         form = JoinForm(request.POST)
#         # form = UserCreationForm(request.POST)
#
#         if form.is_valid():
#
#
#
#             messages.success(request, "New BS was successfully added!")
#             # user.refresh_from_db()  # load the profile instance created by the signal
#
#             username = form.cleaned_data.get('username')
#             raw_password = form.cleaned_data.get('password')
#
#             print(raw_password)
#
#             user = authenticate(username=username, password=raw_password)
#
#             print(user)
#
#             auth_login(request, user)                         # login 함수 자동 연결 실행
#
#
#
#             # session 설정하여 나중에 활용
#             m = User.objects.get(username=request.POST['username'])
#
#             request.session['member_id'] = m.id
#             request.session['member_username'] = m.username
#
#             # return redirect('main:index')
#
#
#             return HttpResponse("Welcome My Python Tutorial")
#             # return redirect('index')       #url의 name을 경로대신 입력한다.(여기는 ajax 때문에 문자열 리턴)
#             # return render(request, 'main/index.html')
#
#     else:
#
#         form = JoinForm()                                               #forms.py의 JoinForm 클래스의 인스턴스
#         # form = UserCreationForm()
#
#         return render(request, 'base/join.html', {'form': form})   # 템플릿 파일 경로 지정, 데이터 전달
#
#
