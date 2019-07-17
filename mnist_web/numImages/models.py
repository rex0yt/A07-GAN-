from django.db import models

# Create your models here.


class num(models.Model):
    number = models.IntegerField('输入数字', null=False)
    gdata = models.DateTimeField('生成时间', auto_now_add=True)
    image = models.ImageField(upload_to='images', default='timg.gif')
    codeimage = models.ImageField(upload_to='images', default='code.png')


