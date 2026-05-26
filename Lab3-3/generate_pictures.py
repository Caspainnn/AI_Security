import os
import random
from captcha.image import ImageCaptcha
import time



# 用于生成图片
captcha_array=list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size=4

def generate_captcha_images(num_images,output_dir):
    image=ImageCaptcha()
    os.makedirs(output_dir,exist_ok=True)   # 确保目录存在
    for i in range(num_images):
        # 生成随机验证码字符
        image_val="".join(random.sample(captcha_array,captcha_size))
        # 生成唯一文件名
        image_name="{}_{}.png".format(image_val,int(time.time()*1000))
        image_path=os.path.join(output_dir,image_name)
        print(image_path)
        # 生成图片并保存
        try:
            image.write(image_val,image_path)
        except IOError as e:
            print(f"Error writing image {image_name}: {e}")
            
if __name__=="__main__":
    # generate_captcha_images(200,"./Lab3-3/datasets/test/")
    generate_captcha_images(10000,"./Lab3-3/datasets/train/")
    