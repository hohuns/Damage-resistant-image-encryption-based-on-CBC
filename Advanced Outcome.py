# 2020-09-30 22:35:57
import numpy as np
import os
import random
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt


# URL method
import qrcode
import pyzbar
import cv2
# Noise.jpg
# qr_img = qrcode.make("https://i.loli.net/2020/09/20/cZNFDaLvfswoenp.jpg")
# Star.jpg
url = 'https://i.loli.net/2020/09/27/1zsu5vC7UWOmryk.jpg'
qr = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H,
)
qr.add_data(url)
qr_img = qr.make_image()
qr_img = qr_img.resize((128, 128))
qr_img.save('QRcode.jpg')
print('QRcode saved')

# base64 method
# import qrcode
# import base64
# import sys
# def ImageToQrcode():
#     with open("noise.jpg", "rb") as imageFile:
#         str = base64.b64encode(imageFile.read())
#     print(len(str))
#     print(sys.getsizeof(str))
#     qr = qrcode.QRCode(
#     version=40,
#     error_correction=qrcode.constants.ERROR_CORRECT_L,
#     box_size=2,
#     border=8,)
#     qr.add_data(str)
#     qr.make()
#     img = qr.make_image(fill_color="black", back_color="white")
#     # img = qrcode.make(str)
#     img.save('myQR.jpg')
#     return img

# Scrambling method: Arnold Transform
def AT(IMG,iter):
    h, w= IMG.shape #Height and width of image
    temp_IMG = IMG
    for i in range(iter):
        AN = np.zeros([h, w])    
        for y in range(h):       
            for x in range(w):           
                xx = (x+a*y)%N           
                yy = ((b*x)+(a*b+1)*y)%N           
                AN[yy][xx] =temp_IMG[y][x]
        temp_IMG = AN
    return temp_IMG #Return scrambled image

def IAT(AN,iter):
    h, w= AN.shape 
    temp_IMG = AN
    for i in range(iter):
        IMG = np.zeros([h, w])   
        for y in range(h):       
            for x in range(w):           
                xx = ((a*b+1)*x-a*y)%N           
                yy = (-b*x+y)%N           
                IMG[yy][xx] =temp_IMG[y][x]  
        temp_IMG = IMG
    return temp_IMG #Return descrambled image

# img = Image.open('star.jpg')
# img = Image.open('noise.jpg')
img = Image.open('QRcode.jpg')
img = img.resize((128,128))
print(img.size)
I0=np.asarray(img)
a,b,N = 5,3,img.width
for i in range(0,50,1):
    save_path = './generated/'
    file_path = os.path.join(save_path,str(i))+ '.jpg'
    I=AT(I0,i)
    cv2.imwrite(file_path, I)

print('arnold transform finished')
print('Next step is AES CBC, it may take a while')
# I=AT(I0,iter)
# img_dec=IAT(I,iter)
# plt.imshow(I)
# cv2.imwrite('scrambled.jpg', I)
# Image.open('scrambled.jpg').show()
# cv2.imwrite('descrambled.jpg', img_dec)
# Image.open('descrambled.jpg').show()


# AES: Cipher block chaining (CBC)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
key=os.urandom(16)
print('\nEncryption key is ',key,'\n') 
# key=b'\xfa(\x91>\xe2\xed\x1a\xc5\\,\xd6\xf9\x0e/\x1a\n'
aesCipher = Cipher( algorithms.AES(key),modes.ECB(),
                    backend = default_backend())
aesEncryptor = aesCipher.encryptor()
aesDecryptor = aesCipher.decryptor()

# percentage damage to different iteration (same percentage)
for i in range(0,25,1):
    save_path = './generated/'
    file_path = os.path.join(save_path,str(i))+ '.jpg'
    image = Image.open(file_path)
    image = image.resize((128,128))
    I=np.asarray(image)

    I_enc=np.zeros(I.size)
    I_dec=np.zeros(I.size)
    ROW=len(I); COL=int(I.size/ROW)

    I_flat = I.flatten()

    iv_enc=np.random.randint(256, size=16)
    iv_enc=np.asarray(iv_enc, dtype="uint8")
    iv_dec=iv_enc

    # percentage = 0.01 * i
    percentage = 0.2
    damge_upperbound = I.size * percentage
    print(damge_upperbound)

    # start=time.time()
    for ptr in range(0,I.size,16):
        blk=I_flat[ptr:ptr+16]
        blk=np.bitwise_xor(blk,iv_enc)
        t_enc=np.frombuffer(aesEncryptor.update(blk),dtype=np.uint8)
        tmp_t_enc = t_enc.copy()
        iv_enc=t_enc
        if ptr<=damge_upperbound:
            tmp_t_enc.setflags(write=1)
            tmp_t_enc[1]=0
        t_enc = tmp_t_enc
        t_dec=np.frombuffer(aesDecryptor.update(bytearray(t_enc)),dtype=np.uint8)
        t_dec=np.bitwise_xor(t_dec,iv_dec)
        iv_dec=t_enc
        I_enc[ptr:ptr+16]=t_enc
        I_dec[ptr:ptr+16]=t_dec
    # end=time.time()
    # print(i,' Encryption time with CBC= ',end-start,'seconds')
        
    img_enc=I_enc.reshape(ROW, COL)
    img_dec0=I_dec.reshape(ROW, COL)
    img_dec=IAT(img_dec0,i)
    save_path = './generated/aes/'
    # name = int(round(percentage * 100))
    name = i
    outfile = os.path.join(save_path,str(name))
    img_dec.astype(np.uint8)
    cv2.imwrite(outfile+'.jpg', img_dec)
    print('CBC image '+ str(i) + ' saved')
print('All image decrypted')
# cv2.imwrite('cbc_encfile.jpg', img_enc)
# cv2.imwrite('cbc_decfile.jpg', img_dec)
# Image.open('cbc_encfile.jpg').show()
# Image.open('cbc_decfile.jpg').show()


print('Using pyzbar library to decode the QRcode')
print('It have a very low chance to decode successfully')
from pyzbar.pyzbar import decode
for i in range(1,25):
    save_path = './generated/aes/'
    file_path = os.path.join(save_path,str(i))+ '.jpg'
    content = decode(Image.open(file_path))
    print(i, content)

