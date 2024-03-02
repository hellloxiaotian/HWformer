import numpy as np
from PIL import Image
import os


# 按照指定尺寸进行图片裁剪
def crop_image(image, patch_w, patch_h, s=48, save_path='', img_name=''):
    width, height = image.size
    w_n = width//patch_w
    h_n = height//patch_h

    c_w = (w_n - 1) * patch_w
    c_h = (h_n - 1) * patch_h

    # 补丁计数
    cnt = 0
    for w in range(0, c_w, s):
        for h in range(0, c_h, s):
            cnt += 1
            # 指定原图片的左、上、右、下
            img = image.crop((w, h, w + patch_w, h + patch_h))

            # img.save("dog-%d.jpg" % cnt)
            img.save(save_path+'s_'+str(s)+'p_'+str(patch_h)+str(cnt)+img_name)

    for w in range(0, c_w, s):
        cnt += 1
        # 指定原图片的左、上、右、下
        img = image.crop((w, height - patch_h, w + patch_w, height))
        # img.save("dog-%d.jpg" % cnt)
        img.save(save_path + 's_' + str(s) + 'p_' + str(patch_h) + str(cnt) + img_name)

    for h in range(0, c_h, s):
        cnt += 1
        # 指定原图片的左、上、右、下
        img = image.crop((width-patch_w, h, width, h + patch_h))
        # img.save("dog-%d.jpg" % cnt)
        img.save(save_path + 's_' + str(s) + 'p_' + str(patch_h) + str(cnt) + img_name)

    print("图片补丁裁剪结束，共有{}张补丁".format(cnt))


def main(dataset_path, save_path):
    img_list = os.listdir(dataset_path)

    for img_name in img_list:
        img = Image.open(dataset_path + img_name)

        # 图片补丁裁剪
        crop_image(img, patch_w=96, patch_h=96, s=48, save_path=save_path, img_name=img_name)


if __name__ == '__main__':
    main('/home/zhengmenghua/Project/Denoising/data/images/test/BSD68/', '/home/zhengmenghua/Project/Denoising/data/images/test/S48_P96_BSD68/')