from django import forms
from django.contrib.auth.models import User        # 'auth.User' table
from django.contrib.auth.forms import UserCreationForm

from .models import Home

# from ckeditor.widgets import CKEditorWidget                   # 일반적 처리가능
from ckeditor_uploader.widgets import CKEditorUploadingWidget  # image 포함 처리가능



# ModelForm활용한 로그인 접근
class LoginForm(forms.ModelForm):

    class Meta:
        model = User
        fields = ['username', 'password',]


class JoinUserForm(UserCreationForm):

    class Meta:
        model = User
        fields=['username', 'email', 'password1', 'password2',]


# Form을 이용할 경우
# class JoinForm(forms.Form):
    # username = forms.CharField()
    # last_name = forms.CharField()
    # first_name = forms.CharField()
    # password = forms.CharField()
    # email = forms.CharField()
    # address = forms.CharField()

    # class Meta:
    #     model = JoinMember
    #     fields = ('username', 'address', 'first_name', 'last_name', 'email', 'password',)


# ModelForm을 이용하여
# class JoinForm(forms.ModelForm):
#
#     class Meta:
#
#         model = User
#         # model = JoinMember
#
#         fields = ['username', 'email', 'password', ]
#         # fields = '__all__'

# class JoinUserForm(forms.ModelForm):
# class JoinUserForm(UserCreationForm):
#
#     class Meta:
#         model = User
#         fields = ('username', 'email', 'password1', 'password2',)



#  UserCreationForm을 통한 간편 회원 가입
# class JoinForm(UserCreationForm):
#
#     class Meta:
#         model = User
#         fields = ['username', 'email', 'password1', 'password2',]

#UserCreationForm을 통한 간편 회원 가입
# class JoinForm(UserCreationForm):
#
#     class Meta:
#         model = User
#         fields = ('username', 'password1', 'password2', 'first_name', 'last_name', 'email',)



# class MainForm(forms.ModelForm):
#
#     # content = forms.CharField(widget=CKEditorWidget())
#     content = forms.CharField(widget=CKEditorUploadingWidget())
#
#     class Meta:
#         model = Base
#         fields = ('subject', 'content', )
#         # fields = '__all__'
