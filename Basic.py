from cryptography.hazmat.primitives.ciphers import Cipher, algorithms,modes
from cryptography.hazmat.backends import default_backend
import numpy as np
import os
from PIL import Image
import random
import cv2
import time

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

    
key=b'\xfa(\x91>\xe2\xed\x1a\xc5\\,\xd6\xf9\x0e/\x1a\n'
aesCipher = Cipher( algorithms.AES(key),modes.ECB(),
                    backend = default_backend())
aesEncryptor = aesCipher.encryptor()
aesDecryptor = aesCipher.decryptor()

iter=4 #Number of Arnold Transform iteration
img = Image.open('noise.jpg')
I0=np.asarray(img)
a,b,N = 5,3,img.width

I=AT(I0,iter)
cv2.imwrite('scrambled.jpg', I)
I=np.asarray(Image.open('scrambled.jpg'))

I_enc=np.zeros(I.size)
I_dec=np.zeros(I.size)
ROW=len(I); COL=int(I.size/ROW)

I_flat = I.flatten()

iv_enc=np.random.randint(256, size=16)
iv_enc=np.asarray(iv_enc, dtype="uint8")
iv_dec=iv_enc

start=time.time()
for ptr in range(0,I.size,16):
    blk=I_flat[ptr:ptr+16]
    blk=np.bitwise_xor(blk,iv_enc)
    t_enc=np.frombuffer(aesEncryptor.update(blk),dtype=np.uint8)
    tmp_t_enc = t_enc.copy()
    iv_enc=t_enc
    
    if ptr>=6400 and ptr<=8848:
        tmp_t_enc.setflags(write=1)
        tmp_t_enc[1]=0
        t_enc=tmp_t_enc
        
    t_dec=np.frombuffer(aesDecryptor.update(bytearray(t_enc)),dtype=np.uint8)
    t_dec=np.bitwise_xor(t_dec,iv_dec)
    iv_dec=t_enc
    I_enc[ptr:ptr+16]=t_enc
    I_dec[ptr:ptr+16]=t_dec
end=time.time()
print('Encryption time with CBC= ',end-start,'seconds')
    
img_enc=I_enc.reshape(ROW, COL)
img_dec0=I_dec.reshape(ROW, COL)

img_dec=IAT(img_dec0,iter)

cv2.imwrite('cbc_encfile.jpg', img_enc)
cv2.imwrite('cbc_decfile.jpg', img_dec)
Image.open('cbc_encfile.jpg').show()
Image.open('cbc_decfile.jpg').show()
