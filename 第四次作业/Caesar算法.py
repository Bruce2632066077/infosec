#caesar算法加密
s = input('请输入要加密的字符串：')
k = int(input('请输入移位值：'))
s_encrypt = ''
for word in s:
	if word == ' ':
		word_encrypt = ' '
	else:
		word_encrypt = chr((ord(word)-ord('a') + k) % 26 + ord('a'))
	s_encrypt += word_encrypt
print(s_encrypt)




